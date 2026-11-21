"""Rebuild Figure 2 from released synthetic replicate summaries and band audit."""
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import t,norm
ROOT = Path(__file__).resolve().parents[2]
SIM = ROOT / 'results' / '06_simulation'
s = pd.read_csv(SIM / 'band_condition_audit_900.csv')
orig = pd.read_csv(SIM / 'confirmatory_3x3_100_replicate_metrics_900.csv')
assert len(orig)==900
plt.rcParams.update({'font.size':10,'axes.titlesize':11,'legend.fontsize':8.5,'axes.labelsize':9.5,'savefig.facecolor':'white','font.family':'DejaVu Sans'})
fig,axs=plt.subplots(1,3,figsize=(12.8,3.75),layout='constrained')
alphas=[0.,.5,1.]
for gamma,marker in ((.5,'o'),(1.,'s')):
    h=orig[orig.gamma==gamma]
    mus=[];err=[]
    for a in alphas:
        x=h[h.alpha==a].retrospective_to_deployment_gap.to_numpy(float)*100
        mus.append(x.mean());err.append(t.ppf(.975, len(x)-1)*x.std(ddof=1)/np.sqrt(len(x)))
    axs[0].errorbar(alphas,mus,yerr=err,marker=marker,capsize=3,linewidth=1.6,label=f'γ = {gamma:g}')
axs[0].axhline(0,color='#999999',linewidth=.6)
axs[0].set(title='(a)',xlabel='Prognostic information α',ylabel='Evaluation gap B (pp)',xticks=alphas)
axs[0].legend(loc='upper right',frameon=False)

h=s[s.gamma==1.]
for j,(key,name) in enumerate([('g0_0_1','Margin untested'),('g1_0_1','Margin tested')]):
    mus=[];err=[]
    for a in alphas:
        x=h[h.alpha==a][key].to_numpy(float)*100
        mus.append(x.mean());err.append(t.ppf(.975,len(x)-1)*x.std(ddof=1)/np.sqrt(len(x)))
    width=.34
    pos=np.arange(3)+(-width/2 if j==0 else width/2)
    axs[1].bar(pos,mus,width=.34,alpha=.92,label=name)
    axs[1].errorbar(pos,mus,yerr=err,fmt='none',ecolor='#222222',linewidth=1.0,capsize=2.8)
axs[1].set(title='(b) γ = 1',ylabel='Prediction-weighted signed label change (pp)',xlabel='Prognostic information α',xticks=np.arange(3),xticklabels=['0','0.5','1'])
axs[1].legend(loc='upper right',frameon=False)

for gamma,marker in ((.5,'o'),(1.,'s')):
    h=orig[orig.gamma==gamma]
    yy=[];lo=[];hi=[]
    for a in alphas:
        x=h[h.alpha==a].policy_ranking_strict_reversal.to_numpy(int)
        count=int(x.sum());n=len(x);phat=count/n
        z=norm.ppf(.975)
        den=1+z*z/n
        mid=(phat+z*z/(2*n))/den
        half=z*np.sqrt(phat*(1-phat)/n+z*z/(4*n*n))/den
        a1=max(0,mid-half);b1=min(1,mid+half)
        yy.append(100*phat);lo.append(100*(phat-a1));hi.append(100*(b1-phat))
    axs[2].errorbar(alphas,yy,yerr=[lo,hi],marker=marker,capsize=3,linewidth=1.6,label=f'γ = {gamma:g}')
axs[2].set(title='(c)',xlabel='Prognostic information α',ylabel='Strict ranking reversals (%)',xticks=alphas,ylim=(0,42))
axs[2].legend(loc='upper right',frameon=False)
for ax in axs:
    ax.grid(axis='y',alpha=.18)
    ax.set_axisbelow(True)
out = ROOT / 'figures' / 'main'
out.mkdir(parents=True, exist_ok=True)
fig.savefig(out / 'Fig2_simulation_corrected.png', dpi=300)
fig.savefig(out / 'Fig2_simulation_corrected.svg')
print('Saved corrected Figure 2 to', out)
