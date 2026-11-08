"""Plot recorded fixed-day beta and synthetic input data."""
from __future__ import annotations
import argparse
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from evcharging.core import Scenario, generate, nominal_price
OUT=ROOT/'artifacts/runs/ideal10h_v1/production'; B=[0,.05,.1,.2,.3,.4,.5,.6,.7,.8,.9,.95,1]

def plot_ev_example(scenario: Scenario, destination: Path) -> None:
    """Plot the first ten vehicles on independent time axes.

    :param scenario: Generated example scenario.
    :param destination: Output PDF path.
    :return: None.
    """
    ids = np.arange(1, 11)
    soc_range = np.concatenate((scenario.arrival[:10], scenario.target[:10])) * 100
    norm = plt.Normalize(soc_range.min(), soc_range.max())
    fig, (arrival_ax, departure_ax) = plt.subplots(
        2, 1, sharex=True, figsize=(3.5, 3.05), gridspec_kw={'hspace': 0.13})
    for ax, times, soc, marker, label, limits, ticks in (
        (arrival_ax, scenario.ta, scenario.arrival, 'o', 'Arrival time (h)', (6.8, 11.2), range(7, 12)),
        (departure_ax, scenario.td, scenario.target, 's', 'Departure time (h)', (14.8, 21.2), range(15, 22)),
    ):
        ax.scatter(ids, 7 + times[:10] / 4, c=soc[:10] * 100, cmap='coolwarm',
                   norm=norm, s=25, marker=marker, edgecolor='black', linewidth=.5, zorder=3)
        ax.set_ylabel(label, fontsize=8)
        ax.set_ylim(*limits)
        ax.set_yticks(ticks)
        ax.set_xlim(.5, 10.5)
        ax.grid(True, linestyle='--', linewidth=.4, alpha=.5)
        ax.tick_params(labelsize=7, length=2)
    departure_ax.set_xticks(ids)
    departure_ax.set_xlabel('Vehicle ID', fontsize=8)
    fig.subplots_adjust(left=.18, right=.82, bottom=.13, top=.98)
    colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='coolwarm'),
                            cax=fig.add_axes([.87, .19, .025, .70]))
    colorbar.set_label('SoC (%)', fontsize=8)
    colorbar.ax.tick_params(labelsize=7, length=2)
    fig.savefig(destination)
    plt.close(fig)


def main() -> None:
    """Write the six figures and numerical summary.

    :return: None.
    """
    plt.rcParams.update({'font.family': 'Times New Roman', 'mathtext.fontset': 'stix'})
    d=ROOT/'outputs/beta_figures'; d.mkdir(parents=True,exist_ok=True); s=generate(62000,20,fixed_day=True); ids=np.arange(1,11)
    plot_ev_example(s, d/'EV_1.pdf')
    fig,ax=plt.subplots(figsize=(10,4)); ax.bar(ids-.175,s.capacity[:10],.35,color='#0780cf'); ax.set_xlabel('Vehicle ID',size=18); ax.set_ylabel('Battery Capacity (kWh)',color='#0780cf',size=18); ax.tick_params(axis='y',labelcolor='#0780cf',size=30); ax.set_ylim(20,40); ax.grid(True,linestyle='--',alpha=.6); twin=ax.twinx(); twin.bar(ids+.175,s.power[:10],.35,color='#fa6d1d'); twin.set_ylabel('Rated Charging Power (kW)',color='#fa6d1d',size=18); twin.tick_params(axis='y',labelcolor='#fa6d1d',size=30); twin.set_ylim(4,8); fig.tight_layout(); fig.savefig(d/'EV_2.pdf'); plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4)); x=7+np.arange(56)*.25
    for seed in range(100): ax.step(np.r_[x,21],np.r_[generate(seed,10,fixed_day=True).price,generate(seed,10,fixed_day=True).price[-1]],where='post',color='lightblue',linewidth=.5,label='Simulated Prices' if seed==0 else None)
    ax.step(np.r_[x,21],np.r_[nominal_price(0,56),nominal_price(0,56)[-1]],where='post',label='Nominal Price',linewidth=2); ax.set(xlabel='Time (hours)',ylabel='Price',xlim=(7,21)); ax.xaxis.label.set_size(18); ax.yaxis.label.set_size(18); ax.tick_params(labelsize=16); ax.legend(fontsize=18,loc='upper left',frameon=False); ax.grid(True,linestyle='--',alpha=.7); fig.tight_layout(); fig.savefig(d/'price.pdf'); plt.close(fig)
    arrays=[np.load(OUT/'beta'/f'beta_{b:.2f}.npz') for b in B]; cost=[float(a['cost'].sum()) for a in arrays]; fair=[float(a['unfairness'].sum()) for a in arrays]
    plt.rcParams.update({'font.family': 'sans-serif', 'mathtext.fontset': 'dejavusans'})
    fig,axs=plt.subplots(1,2,figsize=(12,4))
    for ax,vals,color,label,marker in zip(axs,(cost,fair),('#FF6F61','#6B8E23'),('Cost $C$','Unfairness $D$'),('o','s')):
        ax.plot(B[:-1],vals[:-1],color=color,marker=marker,markersize=8,label=label); ax.plot(B[-2:],vals[-2:],color=color,linestyle='--',marker=marker,markersize=8); ax.set_xlabel(r'Coefficient $\beta$',fontsize=16); ax.set_ylabel(label,fontsize=16); ax.legend(loc='upper left' if marker=='o' else 'upper right',frameon=False,fontsize=14); ax.grid(True,alpha=.4,linestyle='--')
    fig.tight_layout(); fig.savefig(d/'beta.pdf'); plt.close(fig); energy=(s.target-s.arrival)*s.capacity/s.efficiency
    plt.rc('font', family='Times New Roman')
    for key,name in [('cost','BETA_COST_CMP'),('unfairness','BETA_FAIR_CMP')]:
        one=arrays[4][key].sum(axis=0)[:10]/energy[:10]; zero=arrays[0][key].sum(axis=0)[:10]/energy[:10]; ang=np.linspace(0,2*np.pi,10,endpoint=False); aa=np.r_[ang,ang[0]]; fig,ax=plt.subplots(figsize=(6,6),subplot_kw={'polar':True})
        for v,c,label in [(one,'#8ECFC9',r'$\beta=0.3$'),(zero,'#FFBE7A',r'$\beta=0.0$')]:
            vv=np.r_[v,v[0]]; ax.fill(aa,vv,color=c,alpha=.4); ax.plot(aa,vv,color=c,linewidth=2,label=label); ax.scatter(ang,v,color=c,s=80,edgecolor='black',zorder=5)
        ax.set_yticks([.2,.4,.6] if key == 'cost' else [.1,.2,.3,.4])
        ax.grid(color='grey',linestyle='--',linewidth=.5); ax.set_xticks(ang); ax.set_xticklabels([f'EV {i}' for i in ids],fontsize=20,fontweight='bold'); ax.tick_params(axis='x',pad=12); ax.tick_params(axis='y',labelsize=20); ax.legend(loc='upper right',bbox_to_anchor=(.2,.15),fontsize=16,frameon=False); fig.savefig(d/(name+'.pdf')); plt.close(fig)
    summary={'beta':B,'cost':cost,'unfairness':fair,'unit_first10':{key:{str(b):float(np.mean(arrays[i][key].sum(axis=0)[:10]/energy[:10])) for i,b in [(0,0),(4,.3)]} for key in ['cost','unfairness']}}
    (d/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ev1-only', type=Path, metavar='PDF',
                        help='Write only the arrival and departure time figure to this PDF.')
    args = parser.parse_args()
    if args.ev1_only is not None:
        plt.rcParams.update({'font.family': 'Times New Roman', 'mathtext.fontset': 'stix'})
        plot_ev_example(generate(62000, 20, fixed_day=True), args.ev1_only)
    else:
        main()
