"""Train Beta PPO with disjoint validation and resumable checkpoints."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
from torch.distributions import Beta
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from evcharging.core import generate
from evcharging.models import network
from evcharging.simulation_schema import simulate


def validate(actor: torch.nn.Module, cfg: dict, beta: float, tolerance: float) -> dict:
    """Evaluate a deterministic policy on fixed independent scenarios.

    :param actor: Candidate actor.
    :param cfg: PPO configuration.
    :param beta: Fairness weight.
    :param tolerance: Target tolerance.
    :return: Aggregated validation metrics.
    """
    rows=[]
    for i in range(cfg['validation_count']):
        s=generate(cfg['validation_seed_start']+i,cfg['vehicles'], fixed_day=True)
        limit=cfg['capacity_per_ev_kw'][i%len(cfg['capacity_per_ev_kw'])]*cfg['vehicles']
        def policy(x: np.ndarray,k: int,urgency: np.ndarray) -> float:
            """Evaluate the beta-distribution mean.

            :param x: State.
            :param k: Vehicle index.
            :param urgency: Urgency values.
            :return: Normalized action.
            """
            del k,urgency
            with torch.no_grad():
                ab=torch.nn.functional.softplus(actor(torch.tensor(x,dtype=torch.float32)[None,:]))+1
                return float(ab[0,0]/ab.sum())
        traj=simulate(s, limit, policy, state_schema=cfg["state_schema"])
        gap=np.maximum(s.target-traj['soc'][s.td,np.arange(len(s.ta))],0)
        rows.append({'seed':cfg['validation_seed_start']+i,'limit':limit,'failed':int((gap>tolerance).sum()),'gap_kwh':float(gap@s.capacity),'cost':float(traj['cost'].sum()),'unfairness':float(traj['unfair'].sum())})
    return {'failed':sum(x['failed'] for x in rows),'gap_kwh':sum(x['gap_kwh'] for x in rows),'objective':sum((1-beta)*x['cost']+beta*x['unfairness'] for x in rows)/len(rows),'scenarios':rows}


def main() -> None:
    """Train one PPO model and persist optimizer, critic and random state.

    :return: None.
    """
    parser=argparse.ArgumentParser(); parser.add_argument('--config',default='configs/pressure21_v1.json'); parser.add_argument('--run-id',default='pressure21_v1/unspecified'); parser.add_argument('--resume',action='store_true'); args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]; config=json.loads((root/args.config).read_text(encoding='utf8')); cfg=config['ppo']; out=root/'artifacts/runs'/args.run_id/'ppo_model'; out.mkdir(parents=True,exist_ok=args.resume)
    torch.set_num_threads(1); torch.manual_seed(cfg['seed']); actor=network(2); critic=network(1); optim=torch.optim.Adam(list(actor.parameters())+list(critic.parameters()),lr=cfg['learning_rate']); start=0; history=[]; curves=[]
    if args.resume:
        saved=torch.load(out/'latest.pt',map_location='cpu',weights_only=False)
        if saved['config']!=cfg: raise RuntimeError('PPO configuration changed')
        actor.load_state_dict(saved['model']); critic.load_state_dict(saved['critic']); optim.load_state_dict(saved['optimizer']); torch.set_rng_state(saved['rng_state']); start=saved['episode']; history=saved['history']; curves=saved['curves']
    for episode in range(start,cfg['episodes']):
        s=generate(cfg['train_seed_start']+episode,cfg['vehicles'], fixed_day=True); limit=cfg['capacity_per_ev_kw'][episode%len(cfg['capacity_per_ev_kw'])]*cfg['vehicles']; acts=[]; logps=[]
        def policy(x: np.ndarray,k: int,urgency: np.ndarray) -> float:
            """Sample a training action.

            :param x: State.
            :param k: Vehicle index.
            :param urgency: Urgency values.
            :return: Normalized action.
            """
            del k,urgency
            with torch.no_grad():
                ab=torch.nn.functional.softplus(actor(torch.tensor(x,dtype=torch.float32)[None,:]))+1; dist=Beta(ab[:,0],ab[:,1]); sample=dist.sample().clamp(cfg['action_epsilon'],1-cfg['action_epsilon']); acts.append(float(sample)); logps.append(float(dist.log_prob(sample))); return float(sample)
        traj=simulate(s, limit, policy, state_schema=cfg["state_schema"]); x=torch.tensor(traj['states'],dtype=torch.float32); rec=traj['records']; reward=torch.tensor(-((1-s.beta)*traj['cost'][rec[:,0],rec[:,1]]+s.beta*traj['unfair'][rec[:,0],rec[:,1]]),dtype=torch.float32); nxt=torch.cat([x[1:],torch.zeros_like(x[:1])]); done=torch.zeros(len(x)); done[-1]=1
        action=torch.tensor(acts,dtype=torch.float32); old=torch.tensor(logps,dtype=torch.float32)
        with torch.no_grad():
            values=critic(x).squeeze(1); nv=critic(nxt).squeeze(1); delta=reward+cfg['gamma']*(1-done)*nv-values; advantage=torch.zeros_like(delta); acc=torch.tensor(0.0)
            for i in range(len(delta)-1,-1,-1):
                acc=delta[i]+cfg['gamma']*cfg['gae_lambda']*(1-done[i])*acc; advantage[i]=acc
            returns=advantage+values; advantage=(advantage-advantage.mean())/(advantage.std()+1e-8)
        for _ in range(cfg['update_epochs']):
            for ids in torch.randperm(len(x)).split(cfg['batch_size']):
                ab=torch.nn.functional.softplus(actor(x[ids]))+1; dist=Beta(ab[:,0],ab[:,1]); ratio=torch.exp(dist.log_prob(action[ids])-old[ids]); obj=torch.minimum(ratio*advantage[ids],ratio.clamp(1-cfg['clip'],1+cfg['clip'])*advantage[ids]); loss=-obj.mean()+cfg['value_coefficient']*((critic(x[ids]).squeeze(1)-returns[ids])**2).mean()-cfg['entropy_coefficient']*dist.entropy().mean(); optim.zero_grad(); loss.backward(); optim.step()
        curves.append({'episode':episode+1,'seed':cfg['train_seed_start']+episode,'limit':limit,'reward':float(reward.sum()),'loss':float(loss.detach())})
        if (episode+1)%cfg['checkpoint_interval']==0 or episode+1==cfg['episodes']:
            metrics=validate(actor,cfg,s.beta,config['evaluation']['soc_tolerance']); metrics['episode']=episode+1; history.append(metrics)
        saved={'episode':episode+1,'model':actor.state_dict(),'critic':critic.state_dict(),'optimizer':optim.state_dict(),'rng_state':torch.get_rng_state(),'config':cfg,'history':history,'curves':curves}; torch.save(saved,out/'latest.pt')
        if (episode+1)%cfg['checkpoint_interval']==0 or episode+1==cfg['episodes']:
            torch.save(saved,out/f'episode_{episode+1}.pt'); best=min(history,key=lambda z:(z['failed'],z['objective'])); (out/'selection.json').write_text(json.dumps(best,indent=2),encoding='utf8'); (out/'curve.json').write_text(json.dumps(curves,indent=2),encoding='utf8'); (out/'validation.json').write_text(json.dumps(history,indent=2),encoding='utf8'); print('episode',episode+1,'validation',metrics['failed'],metrics['objective'],flush=True)
    (out/'provenance.json').write_text(json.dumps({'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'config_sha256':hashlib.sha256((root/args.config).read_bytes()).hexdigest(),'torch':torch.__version__,'numpy':np.__version__},indent=2),encoding='utf8')


if __name__=='__main__': main()
