"""Reproduce figures: python make_figures.py RESULTS_CSV SEED1_ZIP OUTPUT_DIR.
Requires numpy and matplotlib. Reads only the restricted uint8 tensor format
used by this project's sample files; does not execute arbitrary pickle globals.
"""
import csv,sys,zipfile,io,pickle,collections
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
conds=['no_real','random','quota','accumulate','real_only']
colors=['#777777','#0072B2','#D55E00','#009E73','#8B58A3']
class TensorReader(pickle.Unpickler):
    def find_class(self,module,name):
        if (module,name)==('torch._utils','_rebuild_tensor_v2'):
            return lambda storage,offset,size,stride,*rest: (storage,offset,size,stride)
        if (module,name)==('torch','ByteStorage'): return 'uint8'
        if (module,name)==('collections','OrderedDict'): return collections.OrderedDict
        raise ValueError('Unsupported pickle global')
    def persistent_load(self,pid):
        assert pid[0]=='storage' and pid[1]=='uint8'
        return pid

def samples(z,name):
    t=zipfile.ZipFile(io.BytesIO(z.read(name))); key=next(n for n in t.namelist() if n.endswith('/data.pkl'))
    st,off,shape,stride=TensorReader(io.BytesIO(t.read(key))).load()
    assert shape==(16400,1,28,28) and stride==(784,784,28,1) and off==0
    raw=t.read(key.removesuffix('data.pkl')+'data/'+st[2])
    assert len(raw)==16400*784
    return np.frombuffer(raw,dtype=np.uint8).reshape(shape)

def main(src,archive,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    rows=list(csv.DictReader(open(src,encoding='utf-8-sig')))
    expected={(str(s),c,str(g)) for s in range(3) for c in conds for g in range(1,6)}|{(str(s),'gen0','0') for s in range(3)}
    assert len(rows)==78 and {(r['seed'],r['cond'],r['gen']) for r in rows}==expected
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    summary=[]
    for metric,factor,title,ylab in [('trouser',100,'Evaluator-predicted Trouser share','Share (%)'),('fd',1,'Overall Frechet distance','FD in evaluator feature space')]:
        fig,ax=plt.subplots(figsize=(7.2,4.7))
        for c,col in zip(conds,colors):
            means=[];sds=[]
            for g in range(6):
                a=[float(r[metric])*factor for r in rows if r['cond']==('gen0' if g==0 else c) and int(r['gen'])==g]
                means.append(np.mean(a));sds.append(np.std(a,ddof=1))
                summary.append([c,g,metric,3,means[-1],sds[-1]])
            means=np.array(means);sds=np.array(sds)
            ax.plot(range(6),means,'o-',color=col,label=c,ms=4)
            ax.fill_between(range(6),means-sds,means+sds,color=col,alpha=.12)
        ax.set(xlabel='Generation',ylabel=ylab,title=title,xticks=range(6));ax.grid(alpha=.2)
        ax.legend(ncol=3,loc='upper left',fontsize=9)
        fig.text(.12,.015,'Mean ± sample SD across seeds 0, 1, 2; generation 0 is shared within each seed.',fontsize=8)
        fig.tight_layout(rect=(0,.04,1,1));fig.savefig(out/(metric+'_by_generation.png'),dpi=240);plt.close(fig)
    with open(out/'summary_mean_sd.csv','w',newline='') as f:
        w=csv.writer(f);w.writerow(['condition','generation','metric','n_seeds','mean','sample_sd']);w.writerows(summary)
    z=zipfile.ZipFile(archive)
    keys={n.replace('\\','/'):n for n in z.namelist()}
    assert len(keys)==26
    idx=np.random.default_rng(453).choice(16400,12,replace=False)
    names=['gen0']+conds
    fig,axes=plt.subplots(6,12,figsize=(10,5.5))
    for row,c in enumerate(names):
        path='seed1/gen0.pt' if c=='gen0' else f'seed1/{c}/gen5.pt'
        a=samples(z,keys[path])
        for j,i in enumerate(idx):
            axes[row,j].imshow(a[i,0],cmap='gray',vmin=0,vmax=255,interpolation='nearest')
            axes[row,j].set_xticks([]);axes[row,j].set_yticks([])
            for sp in axes[row,j].spines.values():sp.set_visible(False)
        axes[row,0].set_ylabel('gen0' if c=='gen0' else c,rotation=0,ha='right',va='center',fontsize=10)
    fig.suptitle('Seed 1: randomly selected samples, generation 0 versus generation 5',fontsize=12)
    fig.subplots_adjust(left=.13,right=.99,bottom=.08,top=.9,wspace=.08,hspace=.18)
    fig.text(.13,.025,'12 indices selected with RNG seed 453; same indices per row, but images are not paired trajectories.',fontsize=8)
    fig.savefig(out/'seed1_sample_grid.png',dpi=240);plt.close(fig)
    print('Validated 78 metric rows; sample grid saved. Indices:',idx.tolist())
if __name__=='__main__': main(*sys.argv[1:])
