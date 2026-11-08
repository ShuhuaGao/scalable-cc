"""Plot archived nominal and correlated 200-EV arrival patterns."""
import csv, hashlib, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R=Path(__file__).resolve().parents[1]
O=R/'outputs/r107_figures'

def digest(path: Path) -> str:
    """Return the SHA-256 of a file.

    :param path: Input file.
    :return: Hexadecimal digest.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main() -> None:
    """Render paired day heatmaps and cross-day vehicle correlations.

    :return: None.
    """
    O.mkdir(parents=True, exist_ok=True)
    seeds=range(65000,65050); data={}
    compact=R/'results/figure_data/r107_arrival_departure.npz'
    source_record=json.loads((compact.parent/'r107_sources.json').read_text(encoding='utf8'))
    if digest(compact)!=source_record['compact_sha256']: raise RuntimeError('Compact arrival data changed')
    with np.load(compact) as z:
        for kind in ('Original','Correlated'):
            data[kind]={field:z[kind+'_'+field].copy() for field in ('arrival','departure')}
    heat={k:np.stack([np.bincount(x,minlength=17) for x in d['arrival']])*0.5 for k,d in data.items()}
    assert all(np.allclose(v.sum(axis=1),100) for v in heat.values())
    pairs=np.triu_indices(200,1);means={}
    for kind,ds in data.items():
        for field,x in ds.items():
            values=np.corrcoef(x,rowvar=False)[pairs]
            assert len(values)==19900 and np.isfinite(values).all()
            means[kind,field]=float(values.mean())
    with (O/'figure_heatmap.csv').open('w',encoding='utf8',newline='') as f:
        w=csv.writer(f);w.writerow(['condition','seed','arrival_slot','arrival_percent'])
        for kind,mat in heat.items():
            for row,seed in zip(mat,seeds):
                for slot,value in enumerate(row):w.writerow([kind,seed,slot,value])
    with (O/'figure_correlations.csv').open('w',encoding='utf8',newline='') as f:
        w=csv.writer(f);w.writerow(['condition','variable','mean_pairwise_pearson','days','vehicles','pairs'])
        for (kind,field),value in means.items():w.writerow([kind,field,value,50,200,19900])
    plt.rcParams.update({'font.size':7,'axes.labelsize':7,'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,'pdf.fonttype':42,'font.family':'DejaVu Sans','mathtext.fontset':'dejavusans','axes.linewidth':.8})
    fig=plt.figure(figsize=(6,3.6),layout='constrained')
    gs=fig.add_gridspec(2,3,width_ratios=[1,1,.05],height_ratios=[2.4,1])
    axes=[fig.add_subplot(gs[0,0]),fig.add_subplot(gs[0,1])];cax=fig.add_subplot(gs[0,2]);ax=fig.add_subplot(gs[1,0:2])
    vmax=float(np.ceil(max(x.max() for x in heat.values())/5)*5)
    for i,(axis,kind) in enumerate(zip(axes,('Original','Correlated'))):
        mesh=axis.pcolormesh(np.arange(18)-.5,np.arange(51)+.5,heat[kind],vmin=0,vmax=vmax,cmap='Blues',shading='flat',rasterized=False)
        axis.invert_yaxis();axis.set_xticks([0,4,8,12,16],['07:00','08:00','09:00','10:00','11:00'])
        axis.set_yticks([1,10,20,30,40,50]);axis.set_xlabel('Arrival time')
        if i==0:axis.set_ylabel('Day (fixed seed order)')
        else:axis.tick_params(labelleft=False)
        axis.set_title(f'({chr(97+i)}) {kind} arrivals',fontsize=7)
    cb=fig.colorbar(mesh,cax=cax);cb.solids.set_rasterized(False);cb.set_label('EVs arriving per slot (%)',labelpad=3)
    y=np.array([1.,0.]);colors={'Original':'#666666','Correlated':'#0072b2'}
    ax.axvline(0,color='#555555',lw=.8,zorder=0)
    for kind,offset in (('Original',.13),('Correlated',-.13)):
        vals=[means[kind,'arrival'],means[kind,'departure']]
        ax.scatter(vals,y+offset,s=22,facecolors='none' if kind=='Original' else colors[kind],edgecolors=colors[kind],linewidths=.9,label=kind,zorder=3)
        for value,yy in zip(vals,y+offset):
            ax.annotate(f'{value:.4f}' if abs(value)<.001 else f'{value:.3f}',(value,yy),xytext=(4,0),textcoords='offset points',va='center',fontsize=7)
    ax.set_yticks(y,['Arrival','Departure']);ax.set_ylim(-.48,1.5);ax.set_xlim(-.065,.53)
    ax.set_xticks([0,.2,.4]);ax.set_xlabel('Mean pairwise Pearson correlation');ax.set_title('(c) Across 50 days',fontsize=7);ax.legend(frameon=False,loc='upper right',ncol=2)
    ax.spines[['top','right']].set_visible(False)
    pdf=O/'figure_correlated_arrivals.pdf';png=O/'figure_correlated_arrivals.png'
    fig.savefig(pdf);fig.savefig(png,dpi=250);plt.close(fig)
    report={'sources_sha256':{compact.relative_to(R).as_posix():digest(compact), (compact.parent/'r107_sources.json').relative_to(R).as_posix():digest(compact.parent/'r107_sources.json')},'note':'Rendered from compact arrival and departure arrays; original full inputs were not checked in this render.','script_sha256':digest(Path(__file__)),'heatmap_csv_sha256':digest(O/'figure_heatmap.csv'),'correlation_csv_sha256':digest(O/'figure_correlations.csv'),'output_sha256':{'pdf':digest(pdf),'png':digest(png)},'colorbar_max_percent':vmax,'mean_correlations':{f'{k[0]}_{k[1]}':v for k,v in means.items()},'style':'DejaVu Sans 7 pt; PDF fonttype 42; axes 0.8 pt; Response Figure R1 reference','day_order':'source seed 65000-65049; no sorting by outcome'}
    (O/'figure_manifest.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps({'values':report['mean_correlations'],'color_max':vmax},indent=2))
if __name__=='__main__':main()
