"""Render Pressure21 paper figures from audited frozen results."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from evcharging.core import generate

CONFIG = ROOT / 'configs/pressure21_evaluation.json'
OUT = ROOT / 'artifacts/runs/pressure21_v1/evaluation'
PPO_RUN = ROOT / 'artifacts/runs/ppo_half_formal_20261107'


def digest(path: Path) -> str:
    """Return the SHA-256 digest of a file.

    :param path: File to hash.
    :return: Hexadecimal digest.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> list[dict]:
    """Read scenario rows.

    :param path: CSV file.
    :return: Rows.
    """
    with path.open(newline='',encoding='utf8') as f:return list(csv.DictReader(f))


def main(dest: Path | None = None) -> None:
    """Draw the ten result panels for Figures 8–15.

    :param dest: Output directory for figures and their source inventory.
    :return: None.
    """
    ppo_small = PPO_RUN / 'evaluation/small'
    cfg=json.loads(CONFIG.read_text());base=ROOT/'artifacts/runs/ideal10h_v1'; li=ROOT/'artifacts/runs/li2023_r105/paper_33001'; dest=dest or ROOT/'outputs/pressure21_figures';dest.mkdir(parents=True,exist_ok=True)
    compact=ROOT/'results/figure_data/pressure21_plot_arrays.npz'
    validation=json.loads((compact.parent/'pressure21_full_validation.json').read_text(encoding='utf8'))
    if digest(compact)!=validation['compact_arrays_sha256']:raise RuntimeError('Compact plot arrays changed')
    sources=[Path(__file__), CONFIG, ROOT/'src/evcharging/core.py', compact, compact.parent/'pressure21_full_validation.json']
    small=read(OUT/'small/scenarios.csv');rl=read(OUT/'rl/small/scenarios.csv');old=read(base/'small_evaluation/scenarios.csv');rddpg=read(li/'scenarios.csv')
    scenario_paths = [OUT/'small/scenarios.csv', OUT/'rl/small/scenarios.csv',
                      OUT/'scale/scenarios.csv', base/'small_evaluation/scenarios.csv',
                      base/'scalability/scenarios.csv', li/'scenarios.csv',
                      ppo_small/'scenarios.csv', ROOT/'results/li2023_r105/scenarios.csv']
    for path in scenario_paths:
        expected = validation['sources_sha256'][path.relative_to(ROOT).as_posix()]
        if digest(path) != expected:
            raise RuntimeError(f'Plot source changed: {path}')
    sources.extend(scenario_paths)
    replacement=read(ppo_small/'scenarios.csv')
    rl=[r for r in rl if r['method']!='PPO']+replacement
    names=('IDEAL','IL','UC','DDQN','PPO','RDDPG');colors=dict(zip(names,('#1b7837','#d55e00','#555555','#0072b2','#9467bd','#cc79a7')))
    with np.load(compact) as arrays:
        trajectories={name:{key:arrays[f'{name}_{key}'].copy() for key in ('final','soc','cost','unfair')} for name in names}
    s=generate(76000,10,fixed_day=True);files=[]
    plt.rcParams.update({'font.size':7,'axes.labelsize':7,'axes.titlesize':8.4,'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,'pdf.fonttype':42,'font.family':'DejaVu Sans','mathtext.fontset':'dejavusans'})
    fig,axes=plt.subplots(2,2,figsize=(6,3.8),sharex=True,sharey=True,layout='constrained')
    for k,ax in enumerate(axes.flat):
        for name in ('IDEAL','IL'):ax.plot(trajectories[name]['final'][:,k],label='Ours' if name=='IL' else name,color=colors[name])
        ax.text(.98,.96,f'EV {k+1}',transform=ax.transAxes,ha='right',va='top',bbox={'facecolor':'white','edgecolor':'none','alpha':.8});ax.set(xticks=(0,20,40),yticks=(0,.5,1));ax.grid(alpha=.2)
    fig.supxlabel('Slot',fontsize=7);fig.supylabel('Action',fontsize=7);axes[0,0].legend(loc='upper left',frameon=False);fig.savefig(dest/'Detail_action.pdf');files.append('Detail_action.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(6,2.5),layout='constrained')
    for ax,key,label in zip(axes,('cost','unfair'),('Cost','Unfairness')):
        for name in ('IDEAL','IL'):ax.plot(trajectories[name][key].sum(axis=1),label='Ours' if name=='IL' else name,color=colors[name])
        ax.set(ylabel=label);ax.grid(alpha=.2)
    fig.supxlabel('Slot',fontsize=7);axes[0].legend(loc='upper left',frameon=False);fig.savefig(dest/'EV_10_COST_FAIR_PER_TIME.pdf');files.append('EV_10_COST_FAIR_PER_TIME.pdf');plt.close(fig)
    train_dir=ROOT/'artifacts/runs/pressure21_v1/il_11002/il_model';meta=json.loads((train_dir/'summary.json').read_text());teacher=json.loads((train_dir/'teacher_curve.json').read_text());continuation=json.loads((train_dir/'continuation_curve.json').read_text())
    training_provenance=json.loads((ROOT/'artifacts/runs/pressure21_v1/il_training_record_provenance.json').read_text())
    for path in (train_dir/'summary.json',train_dir/'teacher_curve.json',train_dir/'continuation_curve.json'):
        if digest(path)!=training_provenance[path.relative_to(ROOT).as_posix()]['sha256']:raise RuntimeError('Promoted training record hash mismatch')
    if min(teacher,key=lambda r:r['validation_mse'])['epoch']!=meta['teacher_best_epoch'] or min(continuation,key=lambda r:r['autoregressive_validation_mse'])['epoch']!=meta['continuation_best_epoch']:raise RuntimeError('Selected epochs do not match saved curves')
    sources += [train_dir/'summary.json',train_dir/'teacher_curve.json',train_dir/'continuation_curve.json']
    plt.rcParams.update({'font.size':8,'axes.labelsize':8,'axes.titlesize':8,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8})
    fig,axes=plt.subplots(1,2,figsize=(4,145.8/72),layout='constrained')
    axes[0].plot([r['epoch'] for r in teacher],[r['train_mse'] for r in teacher],label='Training',color='#0072b2');axes[0].plot([r['epoch'] for r in teacher],[r['validation_mse'] for r in teacher],label='Validation',color='#d55e00');axes[0].axvline(meta['teacher_best_epoch'],color='#555555',linestyle='--',linewidth=.9,label='Selected')
    axes[1].plot([r['epoch'] for r in continuation[1:]],[r['train_mse'] for r in continuation[1:]],label='Training',color='#0072b2');axes[1].plot([r['epoch'] for r in continuation],[r['autoregressive_validation_mse'] for r in continuation],label='Validation',color='#d55e00');axes[1].axvline(meta['continuation_best_epoch'],color='#555555',linestyle='--',linewidth=.9,label='Selected')
    for ax,label in zip(axes,('(a)','(b)')):ax.text(.97,.96,label,transform=ax.transAxes,ha='right',va='top');ax.grid(alpha=.2)
    fig.supxlabel('Epoch',fontsize=8);fig.supylabel('Mean squared error',fontsize=8);axes[0].legend(frameon=False,fontsize=7)
    fig.savefig(dest/'loss.pdf');files.append('loss.pdf');plt.close(fig)
    plt.rcParams.update({'font.size':7,'axes.labelsize':7,'axes.titlesize':8.4,'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7})
    fig,axes=plt.subplots(1,2,figsize=(6,2.6),layout='constrained')
    for ax,key,label in zip(axes,('cost','unfair'),('Cost','Unfairness')):
        ax.bar(['Ours' if name=='IL' else name for name in names],[trajectories[name][key].sum() for name in names],color=[colors[name] for name in names]);ax.set_ylabel(label);ax.tick_params(axis='x',rotation=30)
    fig.savefig(dest/'EV_10_CMP.pdf');files.append('EV_10_CMP.pdf');plt.close(fig)
    plt.rcParams.update({'font.size':6,'axes.labelsize':6,'xtick.labelsize':6,'ytick.labelsize':6,'legend.fontsize':6})
    for key,filename in (('cost','EVAL_COST_CMP.pdf'),('unfair','EVAL_FAIR_CMP.pdf')):
        fig,ax=plt.subplots(figsize=(3.25,1.615));fig.subplots_adjust(left=.14,right=.98,bottom=.21,top=.97)
        for name,style in zip(names,('-','-','--','-.',':','--')):
            r=trajectories[name];energy=(r['soc'][s.td,np.arange(10)]-s.arrival)*s.capacity/s.efficiency;unit=np.divide(r[key].sum(axis=0),energy,out=np.zeros(10),where=energy>0)
            ax.plot(np.arange(1,11),unit,label='Ours' if name=='IL' else name,color=colors[name],linestyle=style,marker='o',markersize=2.3,linewidth=1.1)
        ax.set_ylabel('Unit cost' if key=='cost' else 'Unit unfairness');ax.set_xticks(np.arange(1,11));ax.grid(alpha=.2);ax.text(.02,.98,'(a)' if key=='cost' else '(b)',transform=ax.transAxes,ha='left',va='top')
        if key=='cost':ax.tick_params(axis='x',bottom=False,labelbottom=False);fig.savefig(dest/filename,bbox_inches=Bbox.from_extents(0,18/72,3.25,1.615))
        else:ax.set_xlabel('EV');ax.legend(loc='upper right',ncol=2,frameon=False,columnspacing=.6,handlelength=1.2,handletextpad=.3);fig.savefig(dest/filename)
        files.append(filename);plt.close(fig)
    plt.rcParams.update({'font.size':10,'axes.labelsize':10,'axes.titlesize':12,'xtick.labelsize':10,'ytick.labelsize':10,'legend.fontsize':10})
    fig,axes=plt.subplots(1,2,figsize=(6,2.8),layout='constrained'); methods=('IDEAL','IL','DDQN','PPO','RDDPG')
    for ax,metric in zip(axes,('cost','unfairness')):
        values=[]
        for name in methods:
            records=small if name=='IL' else rl if name in ('DDQN','PPO') else rddpg if name=='RDDPG' else old
            values.append([100*(float(next(r[metric] for r in records if int(r['seed'])==seed and r['method']==name))/float(next(r[metric] for r in old if int(r['seed'])==seed and r['method']=='UC'))-1) for seed in cfg['small_seeds']])
        ax.boxplot(values,tick_labels=['Ours' if name=='IL' else name for name in methods],showmeans=True);ax.text(.03,.96,metric.capitalize(),transform=ax.transAxes,ha='left',va='top');ax.grid(alpha=.2)
    fig.supylabel('Change from UC (%)',fontsize=10)
    fig.savefig(dest/'EV-10-Multi.pdf');files.append('EV-10-Multi.pdf');plt.close(fig)
    scale=read(OUT/'scale/scenarios.csv'); prior=read(base/'scalability/scenarios.csv');li_results=ROOT/'results/li2023_r105/scenarios.csv';li_rows=read(li_results);li_result_manifest=json.loads((li_results.parent/'manifest.json').read_text());expected=li_result_manifest['results_sha256'][li_results.name]
    if digest(li_results)!=expected:raise RuntimeError('RDDPG scale results hash mismatch')
    plt.rcParams.update({'font.size':7,'axes.labelsize':7,'axes.titlesize':8.4,'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7})
    for key,filename,label in (('cost','EV-scalability-cost.pdf','Cost per EV'),('unfairness','EV-scalability-fair.pdf','Unfairness per EV')):
        fig,ax=plt.subplots(figsize=(4.3,1.87));fig.subplots_adjust(left=.16,right=.97,bottom=.24,top=.96);sizes=(100,500,1000,3000)
        for name in ('IL','UC','RDDPG'):
            values=[]
            for n in sizes:
                if name=='IL':rows=[r for r in scale if int(r['vehicles'])==n and int(r['training_seed'])==11002]
                elif name=='UC':
                    rows=[r for r in prior if int(r['vehicles'])==n and r['method']=='UC']
                else: rows=([r for r in rddpg if int(r['vehicles'])==n] if n in (100,500) else [r for r in li_rows if int(r['vehicles'])==n and r['training_seed']=='33001'])
                if len(rows)!=3:raise RuntimeError(f'Incomplete scale data: {name} {n}')
                values.append(np.array([float(r[key])/n for r in rows]))
            means=np.array([v.mean() for v in values]);ax.errorbar(sizes,means,yerr=np.maximum(0,[[m-v.min() for m,v in zip(means,values)],[v.max()-m for m,v in zip(means,values)]]),marker='o',capsize=3,label='Ours' if name=='IL' else name)
        ax.set(ylabel=label);ax.set_xlabel('Fleet size (EVs)' if key=='unfairness' else '');ax.grid(alpha=.2);ax.legend(frameon=False)
        ax.text(.97,.97,'(a)' if key=='cost' else '(b)',transform=ax.transAxes,ha='right',va='top')
        fig.savefig(dest/filename);files.append(filename);plt.close(fig)
    timing_dir=ROOT/'artifacts/pressure21_resources_20261106';completion=json.loads((timing_dir/'complete.json').read_text());resource_path=timing_dir/'results.json'
    if digest(resource_path)!=completion['results_sha256'] or completion['records']!=42:raise RuntimeError('Resource study incomplete or changed')
    timing_rows=json.loads(resource_path.read_text())['records'];sizes=(500,1000,1500,2000,2500,3000);means=[];maxima=[];memory_means=[];memory_ranges=[]
    for n in sizes:
        subset=[r for r in timing_rows if r['vehicles']==n and r['method']=='linear']
        if len(subset)!=3:raise RuntimeError('Resource scenario count mismatch')
        means.append(float(np.mean([r['seconds_mean'] for r in subset])))
        maxima.append(float(max(r['seconds_max'] for r in subset)))
        values=np.array([r['memory_postloop']['peak_working_set_bytes']/1024**2 for r in subset]);memory_means.append(float(values.mean()));memory_ranges.append((float(values.min()),float(values.max())))
    fig,axes=plt.subplots(2,1,figsize=(4.3,3.5),sharex=True,layout='constrained')
    axes[0].plot(sizes,means,marker='o',label='Mean');axes[0].plot(sizes,maxima,marker='s',label='Largest observed slot');axes[0].set_ylabel('Seconds per slot');axes[0].legend(frameon=False)
    axes[1].errorbar(sizes,memory_means,yerr=[[v-low for v,(low,high) in zip(memory_means,memory_ranges)],[high-v for v,(low,high) in zip(memory_means,memory_ranges)]],marker='o',capsize=3);axes[1].set(xlabel='Fleet size (EVs)',ylabel='Peak working set (MiB)')
    for ax in axes:ax.grid(alpha=.25)
    axes[0].text(.98,.04,'(a)',transform=axes[0].transAxes,ha='right',va='bottom');axes[1].text(.02,.96,'(b)',transform=axes[1].transAxes,ha='left',va='top')
    fig.savefig(dest/'IDEAL_IL_time.pdf');files.append('IDEAL_IL_time.pdf');plt.close(fig)
    sources += [resource_path, timing_dir/'complete.json', li_results.parent/'manifest.json', ROOT/'artifacts/runs/pressure21_v1/il_training_record_provenance.json']
    (dest/'manifest.json').write_text(json.dumps({'sources_sha256':{p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else p.name:digest(p) for p in sources},'figures_sha256':{name:digest(dest/name) for name in files},'note':'Rendered from compact arrays, recorded scenario summaries and timing data; full trajectories were not checked in this render.'},indent=2),encoding='utf8');print('Staged',len(files),'vector PDFs')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dest',type=Path,help='Staging directory for rendered figures')
    args=parser.parse_args()
    main(args.dest)
