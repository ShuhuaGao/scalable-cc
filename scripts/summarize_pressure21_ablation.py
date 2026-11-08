"""Summarize matched formal and large-fleet Pressure group ablations."""
from __future__ import annotations
import csv
import json
from pathlib import Path
from statistics import mean,stdev

from evaluate_pressure21 import CONFIG,ROOT,OUT,digest


def read(path: Path) -> list[dict]:
    """Read scenario-level results.

    :param path: CSV path.
    :return: Rows.
    """
    with path.open(newline='',encoding='utf8') as f:return list(csv.DictReader(f))


def main() -> None:
    """Summarize days inside each run and then across three trained models.

    :return: None.
    """
    cfg=json.loads(CONFIG.read_text());groups=[];run=[];paired=[]
    for dataset in ('formal','scale'):
        full=read(OUT/dataset/'scenarios.csv'); abl=read(OUT/'ablation'/dataset/'scenarios.csv')
        if dataset=='formal':full=[r for r in full if int(r['vehicles'])==200]
        else:full=[r for r in full if int(r['vehicles']) in cfg['scale_additional_training_sizes']]
        for row in full:row['method']='Full'
        data=full+abl;expected=1500 if dataset=='formal' else 90
        if len(data)!=expected:raise RuntimeError(f'{dataset} records {len(data)} != {expected}')
        for n in ([200] if dataset=='formal' else cfg['scale_additional_training_sizes']):
            for limit in (cfg['formal_capacities_kw'] if dataset=='formal' else [4*n]):
                for train in cfg['training_seeds']:
                    selected=[r for r in data if int(r['vehicles'])==n and int(r['capacity_kw'])==limit and int(r['training_seed'])==train]
                    methods={name:{int(r['seed']):r for r in selected if r['method']==name} for name in ('Full','LocalRefined','WithoutA','WithoutB','WithoutC')}
                    reference=methods['Full']; expected_days=50 if dataset=='formal' else 3
                    if len(reference)!=expected_days or any(v.keys()!=reference.keys() for v in methods.values()):raise RuntimeError('Unmatched scenario cohort')
                    for day in reference:
                        if len({v[day]['input_sha256'] for v in methods.values()})!=1:raise RuntimeError('Input hash mismatch across methods')
                    for name,items in methods.items():
                        record={'dataset':dataset,'vehicles':n,'capacity_kw':limit,'method':name,'training_seed':train,'days':len(items)}
                        for key in ('cost','unfairness','weighted_objective','all_met','vehicle_met_percent','gap_kwh','purchased_kwh','failed_vehicles','total_spp_l1'):
                            record[key]=mean(float(r[key]) for r in items.values())
                        record['cost_per_purchased_kwh']=sum(float(r['cost']) for r in items.values())/sum(float(r['purchased_kwh']) for r in items.values())
                        record['max_physical_residual']=max(float(r['physical_max_residual']) for r in items.values());run.append(record)
                        if name!='Full':
                            p={'dataset':dataset,'vehicles':n,'capacity_kw':limit,'training_seed':train,'comparison':'Full_minus_'+name,'days':len(items)}
                            for k in ('cost','unfairness','weighted_objective','all_met','vehicle_met_percent','gap_kwh'):
                                p[k+'_delta']=mean(float(reference[day][k])-float(items[day][k]) for day in reference)
                            p['lower_objective_days']=sum(float(reference[day]['weighted_objective'])<float(items[day]['weighted_objective']) for day in reference)
                            paired.append(p)
                for name in ('Full','LocalRefined','WithoutA','WithoutB','WithoutC'):
                    these=[r for r in run if r['dataset']==dataset and r['vehicles']==n and r['capacity_kw']==limit and r['method']==name]
                    if len(these)!=3:raise RuntimeError('Expected three trained runs')
                    value={'dataset':dataset,'vehicles':n,'capacity_kw':limit,'method':name,'runs':3,'days_per_run':expected_days}
                    for k in ('cost','unfairness','weighted_objective','all_met','vehicle_met_percent','gap_kwh','purchased_kwh','failed_vehicles','total_spp_l1','cost_per_purchased_kwh'):
                        values=[r[k] for r in these];value[k]={'mean':mean(values),'sample_sd':stdev(values)}
                    value['max_physical_residual']=max(r['max_physical_residual'] for r in these);groups.append(value)
    dest=ROOT/'results/pressure21_v1';dest.mkdir(parents=True,exist_ok=True)
    result={'definition':'day means within each training run, then mean and sample SD across three independently trained runs; compare service before F','groups':groups,'run_summaries':run,'paired_run_differences':paired,'sources':{p.relative_to(ROOT).as_posix():digest(p) for p in [OUT/'formal/scenarios.csv',OUT/'scale/scenarios.csv',OUT/'ablation/formal/scenarios.csv',OUT/'ablation/scale/scenarios.csv']}}
    (dest/'ablation_summary.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    with (dest/'ablation_run_summary.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=run[0]);w.writeheader();w.writerows(run)
    print('Formal and scale ablation summaries complete:',len(groups),'groups')


if __name__=='__main__':main()
