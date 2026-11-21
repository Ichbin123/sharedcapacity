"""Two-factor synthetic mechanism study, reusing the project's simulation/world.py.

alpha controls CBC loading on *latent need* W; gamma controls direct clinician
response to CBC. At gamma=0, D1==D0 exactly by common latent random uniforms.
Potential D under allocation pi is D(pi) = A_pi D1 + (1-A_pi) D0.

This is NEW synthetic data, NOT a re-analysis of MIMIC/MC-MED and NOT a proof
of a prospective health benefit from withholding actual tests.
"""
from __future__ import annotations

import argparse
import hashlib
from dataclasses import replace
import json
import os
from scipy.stats import t as student_t
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'src'))
from simulation.world import WorldParams, calibrate_intercepts, generate, FREE, CBC
from sharedcap_afa.revision.engine import allocate, priority_scores, threshold_scan, final_prob, error_rates

PANEL = FREE + CBC + ['t1','t2']


def params_for(alpha: float, gamma: float) -> WorldParams:
    """Nonnegative, independently set knobs; all CBC marginal variances remain 1.

    The two parameters control structural equation COEFFICIENTS independently;
    nonlinear indirect effects and conditional distributions need not factorize.
    """
    if not (0 <= alpha <= 1 and 0 <= gamma <= 1):
        raise ValueError('alpha and gamma must be in [0,1]')
    return WorldParams(name=f'factorial_a{alpha:.2f}_g{gamma:.2f}',
        rho_hW=.32*alpha, rho_hV=.60,
        rho_wW=.32*alpha, rho_pW=.28*alpha, rho_pV=.55,
        aM=.45*gamma, g1=1.8*gamma, g2=.9*gamma,
        eps=0, target_D=.45, target_S=.40, target_O=.06)


def _predict(train: pd.DataFrame, cal: pd.DataFrame, test: pd.DataFrame, lab: str, seed: int):
    if train[lab].nunique() != 2:
        raise ValueError(f'{lab}: training label only one class, increase n_train')
    probs = {}
    for typ, cols in [('p0', FREE), ('p1', PANEL)]:
        model = make_pipeline(StandardScaler(), LogisticRegression(
            max_iter=250, C=1.0, random_state=seed))
        model.fit(train[cols].to_numpy(float), train[lab].to_numpy(int))
        probs[('cal',typ)] = model.predict_proba(cal[cols].to_numpy(float))[:,1]
        probs[('test',typ)] = model.predict_proba(test[cols].to_numpy(float))[:,1]
    return probs


def _risk(y, pred):
    return float(np.mean(np.asarray(y,dtype=np.int8) != np.asarray(pred,dtype=np.int8)))


def _allocation_metrics(cal: pd.DataFrame, test: pd.DataFrame, probs, lab: str,
                        k: int, pool: int, target_fnr: float, random_repeats: int, seed: int):
    """Validation-only threshold calibration, frozen before evaluating synthetic test."""
    ncal = len(cal)//pool*pool
    ntest = len(test)//pool*pool
    if ncal < pool or ntest < pool:
        raise ValueError('need at least one full pool')
    yc, yt = cal[lab].to_numpy(int)[:ncal], test[lab].to_numpy(int)[:ntest]
    c0,c1 = (probs['cal',i][:ncal] for i in ('p0','p1'))
    t0,t1 = (probs['test',i][:ntest] for i in ('p0','p1'))
    # One frozen centre selected using free-only calibration risk, not test labels.
    thresholds = np.unique(np.r_[0.0, np.quantile(c0,np.linspace(0,1,201)), 1.0])
    tau_free = threshold_scan(yc.astype(float),1-yc.astype(float),[c0],thresholds,target_fnr,'float64')['threshold']
    mask_mc = allocate(ncal,pool,k,priority_scores(c0,'margin',tau_free))
    mask_mt = allocate(ntest,pool,k,priority_scores(t0,'margin',tau_free))
    pc_m = final_prob(c0,c1,mask_mc)
    pt_m = final_prob(t0,t1,mask_mt)
    # Random is seeded independently of both the model fit and evaluation roles.
    cal_rand=[]; test_rand=[]; masks_r=[]
    for j in range(random_repeats):
        csel=allocate(ncal,pool,k,None,seed=seed+10000+11*j)
        tsel=allocate(ntest,pool,k,None,seed=seed+20000+11*j)
        cal_rand.append(final_prob(c0,c1,csel))
        test_rand.append(final_prob(t0,t1,tsel))
        masks_r.append(tsel)
    thresholds=np.unique(np.r_[0.0,
        np.quantile(np.concatenate([pc_m]+cal_rand),np.linspace(0,1,241)),1.0])
    tm=threshold_scan(yc.astype(float),1-yc.astype(float),[pc_m],thresholds,target_fnr,'float64')
    tr=threshold_scan(yc.astype(float),1-yc.astype(float),cal_rand,thresholds,target_fnr,'float64')
    a_m=(pt_m>=tm['threshold']).astype(np.int8)
    a_rs=[(p>=tr['threshold']).astype(np.int8) for p in test_rand]
    fnr_m,fpr_m=error_rates(yt.astype(float),1-yt.astype(float),pt_m,tm['threshold'],'float64')
    rand_rates=np.asarray([error_rates(yt.astype(float),1-yt.astype(float),p,tr['threshold'],'float64')
                           for p in test_rand])
    rm=_risk(yt,a_m)
    rr=float(np.mean([_risk(yt,p) for p in a_rs]))
    result=dict(label=lab, n=ntest, positive=int(yt.sum()),
         margin_threshold=float(tm['threshold']),random_threshold=float(tr['threshold']),
         margin_validation_feasible=bool(tm['feasible']),random_validation_feasible=bool(tr['feasible']),
         margin_validation_fnr=float(tm['fnr']),random_validation_fnr=float(tr['fnr']),
         margin_fnr=fnr_m,margin_fpr=fpr_m,random_fnr=float(rand_rates[:,0].mean()),
         random_fpr=float(rand_rates[:,1].mean()),
         delta_retrospective_error=rm-rr,delta_fnr=fnr_m-float(rand_rates[:,0].mean()),
         delta_fpr=fpr_m-float(rand_rates[:,1].mean()))
    if lab=='D':
        d1 = test['D'].to_numpy(int)[:ntest]
        d0 = test['D0'].to_numpy(int)[:ntest]
        # Potential policy-specific labels: clinician sees CBC if and only if selected.
        dy_m = np.where(mask_mt,d1,d0)
        dy_rs = [np.where(mask,d1,d0) for mask in masks_r]
        e_m = _risk(dy_m,a_m)
        e_r = float(np.mean([_risk(y, pred) for y,pred in zip(dy_rs,a_rs)]))
        rho_m = float(np.mean(dy_m != d1))
        rho_r = float(np.mean([np.mean(d != d1) for d in dy_rs]))
        bias = (e_m-e_r) - (rm-rr)
        if abs(bias)>rho_m+rho_r+1e-10:
            raise AssertionError('0-1 risk label-instability bound violated')
        result.update(dict(delta_deployment_error=e_m-e_r,
             retrospective_to_deployment_gap=bias,
             rho_margin=rho_m,rho_random=rho_r,
             bound_rho_sum=rho_m+rho_r,
             observed_D1_vs_D0_change=float(np.mean(d1!=d0)),
             label_instability_bound_satisfied=True))
    return result


def one_rep(alpha: float, gamma: float, rep: int, *, seed: int=733,
            n_train: int=2500, n_cal: int=1200, n_test: int=2400,
            pool: int=16, slots: int=5, fnr_target: float=.06,
            random_repeats: int=3, intercepts=None) -> list[dict]:
    if not (0 < slots < pool) or min(n_train,n_cal,n_test)<pool:
        raise ValueError('invalid pool budget or sample sizes')
    p = params_for(alpha,gamma)
    # Calibrate the NO-CBC D0 mechanism once, never on D1 at the current gamma.
    # With fixed shared random numbers this makes D0 invariant across (alpha,gamma).
    ic = intercepts if intercepts is not None else calibrate_intercepts(params_for(0, 0),n=30000,seed=999)
    rseed = seed+101*rep  # deterministic independent random streams per replicate
    rng = np.random.default_rng(rseed)
    tr,ca,te = (generate(n,p,ic,rng) for n in (n_train,n_cal,n_test))
    if gamma == 0 and not ((tr.D==tr.D0).all() and (ca.D==ca.D0).all() and (te.D==te.D0).all()):
        raise AssertionError('negative control gamma=0 must imply D1==D0 exactly')
    # Synthetic-only fingerprints used to verify identical potential baseline labels
    # across all grid cells in the same replicate; not real patient data.
    def hash_binary(a):
        return hashlib.sha256(np.asarray(a, dtype=np.uint8).tobytes()).hexdigest()
    baseline_diag = dict(
        d0_prevalence=float(te.D0.mean()), d1_prevalence=float(te.D.mean()),
        s_prevalence=float(te.S.mean()), o_prevalence=float(te.O.mean()),
        d0_test_sha256=hash_binary(te.D0), s_test_sha256=hash_binary(te.S),
        o_test_sha256=hash_binary(te.O),
    )
    results=[]
    for idx,lab in enumerate(('D','S','O')):
        probs=_predict(tr,ca,te,lab,rseed+idx)
        d=_allocation_metrics(ca,te,probs,lab,slots,pool,fnr_target,random_repeats,rseed+idx*30000)
        d.update(alpha=float(alpha),gamma=float(gamma),rep=int(rep),seed=int(rseed))
        if lab == 'D':
            d.update(baseline_diag)
        if gamma==0 and lab=='D':
            assert d['rho_margin']==d['rho_random']==0
            assert abs(d['retrospective_to_deployment_gap'])<1e-12
        results.append(d)
    return results


def aggregate(rows: pd.DataFrame) -> pd.DataFrame:
    groups=[]
    for (a,g,lab),h in rows.groupby(['alpha','gamma','label'],sort=True):
        item=dict(alpha=a,gamma=g,label=lab,n_reps=len(h))
        for key in ['delta_retrospective_error','delta_fnr','delta_fpr']:
            x=h[key].to_numpy(float)
            item[key+'_mean']=float(x.mean())
            item[key+'_se']=float(x.std(ddof=1)/np.sqrt(len(x))) if len(x)>1 else float('nan')
            item[key+'_ci95_lower']=float(x.mean()-student_t.ppf(.975,len(x)-1)*item[key+'_se']) if len(x)>1 else float('nan')
            item[key+'_ci95_upper']=float(x.mean()+student_t.ppf(.975,len(x)-1)*item[key+'_se']) if len(x)>1 else float('nan')
        if lab=='D':
            for key in ['delta_deployment_error','retrospective_to_deployment_gap',
                        'rho_margin','rho_random','observed_D1_vs_D0_change']:
                item[key+'_mean']=float(h[key].mean())
            item['policy_ranking_reversal_share']=float(np.mean(
                np.sign(h['delta_retrospective_error'].to_numpy(float)) !=
                np.sign(h['delta_deployment_error'].to_numpy(float))))
        groups.append(item)
    return pd.DataFrame(groups)


def _numbers(x: str):
    r=[float(s.strip()) for s in x.split(',') if s.strip()]
    if not r or len(set(r)) != len(r) or any(not(0<=t<=1) for t in r):
        raise ValueError('alphas/gammas must be unique numbers in [0,1]')
    return r


def run(args):
    out=Path(args.out_dir)
    if args.reps<1 or args.n_train<100 or args.n_cal<args.pool or args.n_test<args.pool:
        raise ValueError('invalid experiment sizes')
    alphas,gammas=_numbers(args.alphas),_numbers(args.gammas)
    manifest=dict(version='label-mechanism-factorial-v2-fixed-D0-paired-CRN',alphas=alphas,gammas=gammas,
        replicates=args.reps,n_train=args.n_train,n_cal=args.n_cal,n_test=args.n_test,
        intercept_n=args.intercept_n,pool=args.pool,slots=args.slots,random_repeats=args.random_repeats,
        fnr_target=args.fnr_target,seed=args.seed,
        dgp='existing simulation.world WorldParams; shared intercept calibrated at alpha=0 gamma=0; gamma scales aM/g1/g2; alpha scales W CBC loadings',
        paired_exogenous_by_rep=True, d0_baseline_fixed=True,
        known_counterfactuals='D0 and D1 share latent uD; D(pi)=where(policy_selection,D1,D0)',
        outcome_note='O has eps=0; no allocation-induced treatment/health effect modeled',
        research_status='NEW SYNTHETIC EXPLORATORY; NOT CLINICAL VALIDATION')
    mf=out/'manifest.json'
    if out.exists() and any(out.iterdir()):
        if not args.resume:
            raise FileExistsError(f'Refusing to overwrite nonempty {out}; use --resume for same manifest')
        if not mf.is_file() or json.loads(mf.read_text()) != manifest:
            raise ValueError('Existing run manifest differs from requested settings; use a NEW output directory')
    else:
        out.mkdir(parents=True,exist_ok=True)
        mf.write_text(json.dumps(manifest,indent=2)+'\n')
    cells=out/'cells'
    cells.mkdir(exist_ok=True)
    rows=[]
    # The marginal law of (X0, W, U, D0, S, O) is invariant to alpha/gamma.
    # Keep a single intercept triple fixed for every cell, including the
    # counterfactual no-test decision. Do not recenter observed D1 separately.
    common_ic=calibrate_intercepts(params_for(0.0,0.0), n=args.intercept_n,
                                   seed=args.seed+10000)
    for ai,a in enumerate(alphas):
        for gi,g in enumerate(gammas):
            missing=[rep for rep in range(args.reps) if not (cells/f'a{ai}_g{gi}_rep{rep}.json').exists()]
            if missing:
                for rep in missing:
                    # The same 'rep' uses the same random draws across the entire
                    # factorial grid. Between different reps seeds remain distinct.
                    cell=one_rep(a,g,rep,seed=args.seed,
                        n_train=args.n_train,n_cal=args.n_cal,n_test=args.n_test,
                        pool=args.pool,slots=args.slots,fnr_target=args.fnr_target,
                        random_repeats=args.random_repeats,intercepts=common_ic)
                    f=cells/f'a{ai}_g{gi}_rep{rep}.json'
                    tmp=f.with_suffix('.tmp')
                    tmp.write_text(json.dumps(cell,indent=2,allow_nan=False)+'\n')
                    os.replace(tmp,f)
            for rep in range(args.reps):
                f=cells/f'a{ai}_g{gi}_rep{rep}.json'
                vals=json.loads(f.read_text())
                if len(vals)!=3 or set(x['label'] for x in vals)!={'D','S','O'}:
                    raise ValueError(f'incomplete synthetic replicate {f}')
                rows.extend(vals)
            print(f'completed alpha={a:.2f}, gamma={g:.2f}, reps={args.reps} (resumable)',flush=True)
    df=pd.DataFrame(rows)
    # Cross-cell invariance checks: this is the main fix relative to pilot_02.
    for rep,h in df[df.label=='D'].groupby('rep'):
        for name in ('d0_test_sha256','s_test_sha256','o_test_sha256'):
            if h[name].nunique()!=1:
                raise AssertionError(f'Paired CRN / fixed-baseline violation rep={rep}: {name}')
        h0=h[h.gamma==0]
        if not h0.empty and not np.allclose(h0.d0_prevalence, h0.d1_prevalence, atol=0, rtol=0):
            raise AssertionError(f'gamma=0 D1!=D0 in rep={rep}')
    df.to_csv(out/'replicate_metrics.csv',index=False)
    summary=aggregate(df)
    summary.to_csv(out/'summary.csv',index=False)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        d=summary[summary.label=='D']
        grid=d.pivot(index='gamma',columns='alpha',values='retrospective_to_deployment_gap_mean')
        fig,ax=plt.subplots(figsize=(6,4.5))
        im=ax.imshow(grid.to_numpy(),origin='lower',aspect='auto')
        ax.set_xticks(range(len(grid.columns)),[f'{v:g}' for v in grid.columns]);ax.set_yticks(range(len(grid.index)),[f'{v:g}' for v in grid.index])
        ax.set_xlabel('Prognostic strength alpha');ax.set_ylabel('Decision response gamma')
        ax.set_title('Simulated retrospective–deployment contrast gap')
        fig.colorbar(im,ax=ax,label='0–1 error contrast gap')
        fig.tight_layout();fig.savefig(out/'mechanism_gap.png',dpi=160);plt.close(fig)
    except ImportError:
        print('Matplotlib unavailable; numeric results were saved.')
    print(f'Saved synthetic-only aggregate outputs to {out}')


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--alphas',default='0,0.5,1')
    ap.add_argument('--gammas',default='0,0.5,1')
    ap.add_argument('--reps',type=int,default=4)
    ap.add_argument('--n-train',type=int,default=2500)
    ap.add_argument('--n-cal',type=int,default=1200)
    ap.add_argument('--n-test',type=int,default=2400)
    ap.add_argument('--intercept-n',type=int,default=25000)
    ap.add_argument('--pool',type=int,default=16)
    ap.add_argument('--slots',type=int,default=5)
    ap.add_argument('--fnr-target',type=float,default=.06)
    ap.add_argument('--random-repeats',type=int,default=3)
    ap.add_argument('--seed',type=int,default=733)
    ap.add_argument('--resume',action='store_true',help='Resume same config from per-rep JSON files')
    ap.add_argument('--out-dir',default='outputs/research_label_mechanism/simulation_pilot')
    a=ap.parse_args(argv)
    run(a)

if __name__=='__main__':
    main()
