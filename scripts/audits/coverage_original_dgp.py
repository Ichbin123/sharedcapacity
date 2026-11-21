#!/usr/bin/env python3
"""Diagnostic coverage simulation with the ORIGINAL uploaded synthetic mechanism.

A separate, independent experiment: freeze trained predictors and an independently
chosen Margin centre, then repeatedly draw calibration/test pools under world.py.
We compare conditional (given calibration) and repeated-calibration estimands.
Mimics the application's fixed-grid and order-statistic selection procedures.
This is NOT the original manuscript's 100-replicate experiment.
"""
import argparse,time,json,hashlib
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from simulation.world import generate,calibrate_intercepts
from simulation.mechanism_grid import _predict,params_for,FREE,PANEL
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

M=16;K=5;PI=5/16;Q=.06;DRAWS=3
GRID=np.arange(.001,1.0,.0025)

def pooled_scores(y,p0,p1,center,rng,random_repeats=3):
    n=len(y);assert n%M==0
    pp0=p0.reshape(-1,M)
    dist=np.abs(pp0-center)
    am=np.zeros((n//M,M),bool)
    picks=np.argsort(dist,axis=1,kind='stable')[:,:K]
    am[np.arange(n//M)[:,None],picks]=True
    am=am.reshape(-1)
    # uniform without replacement, independent draws
    ar=np.zeros((random_repeats,n//M,M),bool)
    for j in range(random_repeats):
        pick=np.argsort(rng.random((n//M,M)),axis=1)[:,:K]
        ar[j,np.arange(n//M)[:,None],pick]=True
    ar=ar.reshape(random_repeats,n)
    return np.where(am,p1,p0),np.where(ar,p1[None,:],p0[None,:]),am

def threshold(scores_positive,rule):
    vals=np.sort(np.asarray(scores_positive,dtype=float).ravel())
    if rule=='grid':
        idx=np.searchsorted(vals,GRID,side='left')
        ok=np.flatnonzero(idx/len(vals)<=Q+1e-12)
        return float(GRID[ok[-1]]) if len(ok) else float(GRID[0])
    # correct order statistic largest t with #{scores < t}/len <= q
    k=int(np.floor(Q*len(vals)+1e-12))
    return float(vals[min(k,len(vals)-1)])

def select(cal,rule,rng):
    sM,sR,_=pooled_scores(*cal, rng=rng)
    y=cal[0].astype(bool)
    return threshold(sM[y],rule), threshold(sR[:,y],rule)

def eval_fpr(test,tm,tr,rng):
    sm,sr,_=pooled_scores(*test,rng=rng)
    neg=~test[0].astype(bool)
    return float((sm[neg]>=tm).mean()-(sr[:,neg]>=tr).mean())

def resample(data,rng):
    # independently bootstrap complete fixed-size pools with replacement
    n=len(data[0]);n_pool=n//M
    idx=(rng.integers(0,n_pool,n_pool)[:,None]*M+np.arange(M)).ravel()
    return tuple(x[idx] if isinstance(x,np.ndarray) else x for x in data)

def ci(boot):
    return np.percentile(boot,[2.5,97.5])

def get_truth_ref(y,p0,p1,center):
    # large, common independent test sample; margin within pools
    n=len(y)
    pp0=p0.reshape(-1,M)
    am=np.zeros((n//M,M),bool)
    pick=np.argsort(abs(pp0-center),axis=1,kind='stable')[:,:K]
    am[np.arange(n//M)[:,None],pick]=True
    neg=(y==0)
    sm=np.where(am.reshape(-1),p1,p0)
    return tuple(np.sort(s[neg]) for s in (sm,p0,p1))

def ref_fpr(sorted_ref,tm,tr):
    sm,s0,s1=sorted_ref
    def tail(s,t):return 1-np.searchsorted(s,t,side='left')/len(s)
    return tail(sm,tm)-((1-PI)*tail(s0,tr)+PI*tail(s1,tr))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--outer',type=int,default=100)
    p.add_argument('--boot',type=int,default=200)
    p.add_argument('--truth-cal',type=int,default=3000)
    p.add_argument('--ref-test',type=int,default=128000)
    p.add_argument('--seed',type=int,default=1234678)
    p.add_argument('--alpha',type=float,default=.5)
    p.add_argument('--endpoint',choices=['D0','O','S'],default='D0')
    p.add_argument('--cal-n',type=int,default=1200)
    p.add_argument('--test-n',type=int,default=2400)
    p.add_argument('--out',default='outputs/bootstrap_coverage_diagnostic.csv')
    args=p.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    rng=np.random.default_rng(args.seed)
    pp=params_for(args.alpha,0.)
    intercepts=calibrate_intercepts(params_for(0.,0.),n=25000,seed=20271118)
    # A single fitted prediction system; freeze scores and centre before outer repetitions.
    training=generate(2500,pp,intercepts,rng)
    tune=generate(1200,pp,intercepts,rng)
    models=[]
    for cols in (FREE,PANEL):
        model=make_pipeline(StandardScaler(),LogisticRegression(max_iter=250,C=1.0,random_state=args.seed))
        model.fit(training[cols].to_numpy(float),getattr(training,args.endpoint).to_numpy(int))
        models.append((cols,model))
    def score(df):
        return tuple(mod.predict_proba(df[cols].to_numpy(float))[:,1] for cols,mod in models)
    p0_tune,_=score(tune)
    tune_pos=np.sort(p0_tune[getattr(tune,args.endpoint).to_numpy(bool)])
    # isolated development-centre selection; fixed for all outer datasets
    k=int(np.floor(Q*len(tune_pos)+1e-12))
    centre=float(tune_pos[min(k,len(tune_pos)-1)])
    print(f'fixed fitted models; independent centre={centre:.6g}; alpha={args.alpha:g}; endpoint={args.endpoint}',flush=True)
    def draw(n):
        df=generate(n,pp,intercepts,rng)
        p0,p1=score(df)
        return getattr(df,args.endpoint).to_numpy(int),p0,p1,centre
    ref=draw(args.ref_test)
    refneg=get_truth_ref(*ref)
    # Monte Carlo approximation to repeated-calibration target for each rule
    truth=[];refc=[]
    tic=time.time()
    for j in range(args.truth_cal):
        cal=draw(args.cal_n)
        tm_grid,tr_grid=select(cal,'grid',rng)
        tm_ord,tr_ord=select(cal,'orderstat',rng)
        refc.append((ref_fpr(refneg,tm_grid,tr_grid),ref_fpr(refneg,tm_ord,tr_ord)))
    means=np.mean(refc,axis=0);ses=np.std(refc,axis=0,ddof=1)/np.sqrt(len(refc))
    truth=dict(grid=float(means[0]),orderstat=float(means[1]))
    print('truth repeated-calibration',truth,'MC standard error(pp)',ses*100,'elapsed',round(time.time()-tic,2),flush=True)
    rows=[]
    for r in range(args.outer):
        c=draw(args.cal_n);t=draw(args.test_n)
        selected={rule:select(c,rule,rng) for rule in ('grid','orderstat')}
        stats={rule:eval_fpr(t,*selected[rule],rng) for rule in selected}
        bootjoint={rule:np.empty(args.boot) for rule in selected}
        bootfixed={rule:np.empty(args.boot) for rule in selected}
        for j in range(args.boot):
            cb=resample(c,rng);tb=resample(t,rng)
            for rule in selected:
                tm,tr=select(cb,rule,rng)
                bootjoint[rule][j]=eval_fpr(tb,tm,tr,rng)
                bootfixed[rule][j]=eval_fpr(tb,*selected[rule],rng)
        for rule in selected:
            jl,ju=ci(bootjoint[rule]);fl,fu=ci(bootfixed[rule])
            theta_cond=ref_fpr(refneg,*selected[rule])
            rows.append(dict(outer=r,rule=rule,estimate=stats[rule],truth_repeated=truth[rule],truth_conditional=theta_cond,
                 joint_low=jl,joint_high=ju,fixed_low=fl,fixed_high=fu,
                 joint_covers_repeated=(jl<=truth[rule]<=ju),fixed_covers_conditional=(fl<=theta_cond<=fu),
                 joint_width=ju-jl,fixed_width=fu-fl,
                 threshold_margin=selected[rule][0],threshold_random=selected[rule][1]))
        if (r+1)%20==0 or r+1==args.outer:
            d=pd.DataFrame(rows)
            d.to_csv(args.out,index=False)
            print(f'outer completed {r+1}/{args.outer}; elapsed {time.time()-tic:.1f}s',flush=True)
    pd.DataFrame(rows).to_csv(args.out,index=False)
    d=pd.DataFrame(rows)
    print(d.groupby('rule')[['joint_covers_repeated','fixed_covers_conditional','joint_width','fixed_width']].mean(),flush=True)
    json_path=args.out.replace('.csv','_manifest.json')
    with open(json_path,'w') as f:
        json.dump({'scope':'independent diagnostic, original synthetic DGP, fixed trained models/centre, original application grid threshold; not manuscript confirmatory simulations',
        'args':vars(args),'margin_centre':centre,'truth_repeated':truth,'truth_se':dict(zip(['grid','orderstat'],ses.tolist())),'first':__file__},f,indent=2)
    print('saved',args.out,flush=True)

if __name__=='__main__':main()
