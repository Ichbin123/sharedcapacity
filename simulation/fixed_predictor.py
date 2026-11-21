"""Paired synthetic FIXED-PREDICTOR PURE-LABEL mechanism experiment.

Within (alpha, replicate), fit on D(gamma=0)==D0 ONCE; freeze predictors,
margin centre, decision thresholds, simulated competition pools, allocations
and classification decisions. Subsequently vary only D1(gamma), evaluate
retrospective labels D1(gamma) against policy-specific labels D(pi,gamma).

All patient observations are synthetic; only aggregate metrics leave memory.
This is a sensitivity experiment, NOT identification of clinical effects.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.stats import t as student_t

from simulation.world import calibrate_intercepts, generate, FREE, CBC
from simulation.assignment_contributions import decompose_by_assignment, GROUPS
from simulation.mechanism_grid import PANEL, _predict, _risk, params_for, _numbers
from sharedcap_afa.revision.engine import (allocate, priority_scores,
                                            threshold_scan, final_prob)

VERSION = "synthetic-fixed-predictor-assignment-contributions-v2"
FEATURES = FREE + CBC + ["t1", "t2"]


def array_hash(data: Any) -> str:
    arr = np.asarray(data)
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def prepare_frozen_policy(train: pd.DataFrame, cal: pd.DataFrame, test: pd.DataFrame,
                          *, pool: int, slots: int, fnr_target: float,
                          random_repeats: int, seed: int) -> dict:
    """Predict and calibrate once using *only* gamma=0 labels and calibration data.

    Mirrors mechanism_grid._allocation_metrics at gamma=0 for comparability,
    but additionally preserves fixed per-person decisions/masks in memory.
    """
    if not 0 < slots < pool or random_repeats < 1:
        raise ValueError("Require 0 < slots < pool and random_repeats >= 1")
    ncal = len(cal) // pool * pool
    ntest = len(test) // pool * pool
    if ncal == 0 or ntest == 0:
        raise ValueError("Need at least one complete calibration and test pool")
    if not np.array_equal(train.D.to_numpy(), train.D0.to_numpy()):
        raise ValueError("Frozen training must be gamma=0, so D==D0")
    if not np.array_equal(cal.D.to_numpy(), cal.D0.to_numpy()):
        raise ValueError("Frozen calibration must be gamma=0, so D==D0")
    probs = _predict(train, cal, test, "D", seed)
    yc = cal.D.to_numpy(np.int8)[:ncal]
    c0, c1 = (np.asarray(probs["cal", i][:ncal], dtype=float) for i in ("p0", "p1"))
    t0, t1 = (np.asarray(probs["test", i][:ntest], dtype=float) for i in ("p0", "p1"))
    thresholds = np.unique(np.r_[0.0, np.quantile(c0, np.linspace(0, 1, 201)), 1.0])
    tau_free = float(threshold_scan(yc.astype(float), 1-yc.astype(float),
                                    [c0], thresholds, fnr_target, "float64")["threshold"])
    m_cal = allocate(ncal, pool, slots, priority_scores(c0, "margin", tau_free))
    m_test = allocate(ntest, pool, slots, priority_scores(t0, "margin", tau_free))
    p_m_cal = final_prob(c0, c1, m_cal)
    p_m_test = final_prob(t0, t1, m_test)
    r_cal_probs, r_test_probs, r_masks = [], [], []
    for j in range(random_repeats):
        r_cal_mask = allocate(ncal, pool, slots, None, seed=seed+10000+11*j)
        r_test_mask = allocate(ntest, pool, slots, None, seed=seed+20000+11*j)
        r_cal_probs.append(final_prob(c0, c1, r_cal_mask))
        r_test_probs.append(final_prob(t0, t1, r_test_mask))
        r_masks.append(r_test_mask)
    th_grid = np.unique(np.r_[0.0, np.quantile(np.concatenate([p_m_cal]+r_cal_probs),
                                                  np.linspace(0, 1, 241)), 1.0])
    tm = threshold_scan(yc.astype(float), 1-yc.astype(float), [p_m_cal],
                        th_grid, fnr_target, "float64")
    tr = threshold_scan(yc.astype(float), 1-yc.astype(float), r_cal_probs,
                        th_grid, fnr_target, "float64")
    pred_m = (p_m_test >= tm["threshold"]).astype(np.int8)
    pred_r = [(p >= tr["threshold"]).astype(np.int8) for p in r_test_probs]
    for mask in [m_test]+r_masks:
        if not np.all(np.asarray(mask).reshape(-1, pool).sum(axis=1) == slots):
            raise AssertionError("Capacity check failed")
    return {
        "n_test": ntest, "margin_mask": m_test, "random_masks": r_masks,
        "margin_pred": pred_m, "random_pred": pred_r,
        "tau_free": tau_free, "margin_threshold": float(tm["threshold"]),
        "random_threshold": float(tr["threshold"]),
        "margin_validation_fnr": float(tm["fnr"]),
        "random_validation_fnr": float(tr["fnr"]),
        "margin_validation_feasible": bool(tm["feasible"]),
        "random_validation_feasible": bool(tr["feasible"]),
        "predictor_sha256": array_hash(np.column_stack([t0, t1])),
        "decision_sha256": array_hash(np.column_stack([pred_m]+pred_r)),
        "assignment_sha256": array_hash(np.column_stack([m_test]+r_masks)),
    }


def evaluate_fixed(test: pd.DataFrame, frozen: dict, *, alpha: float, gamma: float,
                   rep: int, seed: int) -> dict:
    """Only the evaluation labels D1(gamma) change; all predictions stay fixed."""
    n = frozen["n_test"]
    d1 = test.D.to_numpy(np.int8)[:n]
    d0 = test.D0.to_numpy(np.int8)[:n]
    m_mask = frozen["margin_mask"]
    rmasks = frozen["random_masks"]
    a_m = frozen["margin_pred"]
    a_rs = frozen["random_pred"]
    if any(len(arr)!=n for arr in [d1,d0,m_mask,a_m]+rmasks+a_rs):
        raise AssertionError("Prediction/label/assignment size mismatch")
    y_m = np.where(m_mask, d1, d0)
    y_rs = [np.where(mask, d1, d0) for mask in rmasks]
    retro_m = _risk(d1, a_m)
    retro_rs = np.array([_risk(d1, pr) for pr in a_rs])
    deploy_m = _risk(y_m, a_m)
    deploy_rs = np.array([_risk(y, pr) for y,pr in zip(y_rs,a_rs)])
    retro_gap = retro_m-float(retro_rs.mean())
    deploy_gap = deploy_m-float(deploy_rs.mean())
    bias = deploy_gap-retro_gap
    rho_m=float(np.mean(y_m!=d1))
    rho_r=float(np.mean([np.mean(y!=d1) for y in y_rs]))
    if abs(bias) > rho_m+rho_r+1e-10:
        raise AssertionError("Label-instability bound violated")
    # Exact individual error-change identity; useful for auditing cancellation.
    corr_m = np.mean((~m_mask)*(d1-d0)*(2*a_m-1))
    corr_r = np.mean([np.mean((~mask)*(d1-d0)*(2*pr-1))
                      for mask,pr in zip(rmasks,a_rs)])
    if not (np.isclose(corr_m, deploy_m-retro_m, atol=1e-12)
            and np.isclose(corr_r, np.mean(deploy_rs-retro_rs), atol=1e-12)
            and np.isclose(bias, corr_m-corr_r, atol=1e-12)):
        raise AssertionError("Per-patient label-change identity failed")
    changed = (d1!=d0)
    disagree = np.array([np.mean(a_m != ar) for ar in a_rs])
    changed_and_disagree = np.array([np.mean(changed & (a_m != ar)) for ar in a_rs])
    selection_disagree = np.array([np.mean(m_mask != mask) for mask in rmasks])
    attribution = decompose_by_assignment(d0, d1, a_m, a_rs, m_mask, rmasks)
    if not np.isclose(attribution["gap"], bias, rtol=0, atol=1e-12):
        raise AssertionError("Assignment contribution identity failed")
    strict_reversal = bool(retro_gap*deploy_gap < -1e-15)
    result = {
        "alpha":float(alpha), "gamma":float(gamma), "rep":int(rep),"seed":int(seed),
        "model_training_gamma":0.0,"predictions_and_allocations_frozen":True,
        "n":int(n), "positive_D1":int(d1.sum()),
        "d0_prevalence":float(d0.mean()),"d1_prevalence":float(d1.mean()),
        "observed_D1_vs_D0_change":float(changed.mean()),
        "margin_retrospective_error":float(retro_m),
        "random_retrospective_error":float(retro_rs.mean()),
        "margin_deployment_error":float(deploy_m),
        "random_deployment_error":float(deploy_rs.mean()),
        "delta_retrospective_error":float(retro_gap),
        "delta_deployment_error":float(deploy_gap),
        "retrospective_to_deployment_gap":float(bias),
        "margin_label_correction":float(corr_m),
        "random_label_correction":float(corr_r),
        "rho_margin":rho_m,"rho_random":rho_r,
        "bound_rho_sum":rho_m+rho_r,
        "label_instability_bound_satisfied":True,
        "policy_ranking_strict_reversal":strict_reversal,
        "policy_ranking_tie_change":bool((abs(retro_gap)<1e-12)!=(abs(deploy_gap)<1e-12)),
        "mean_prediction_disagreement_rate":float(disagree.mean()),
        "mean_changed_on_prediction_disagreement_rate":float(changed_and_disagree.mean()),
        "mean_selection_disagreement_rate":float(selection_disagree.mean()),
        "margin_center":float(frozen["tau_free"]),
        "margin_threshold":float(frozen["margin_threshold"]),
        "random_threshold":float(frozen["random_threshold"]),
        "margin_validation_fnr":float(frozen["margin_validation_fnr"]),
        "random_validation_fnr":float(frozen["random_validation_fnr"]),
        "margin_validation_feasible":bool(frozen["margin_validation_feasible"]),
        "random_validation_feasible":bool(frozen["random_validation_feasible"]),
        "predictor_sha256":frozen["predictor_sha256"],
        "decision_sha256":frozen["decision_sha256"],
        "assignment_sha256":frozen["assignment_sha256"],
        "d0_test_sha256":array_hash(d0.astype(np.uint8)),
        "s_test_sha256":array_hash(test.S.to_numpy(np.uint8)[:n]),
        "o_test_sha256":array_hash(test.O.to_numpy(np.uint8)[:n]),
    }
    for g in GROUPS:
        for metric, value in attribution["groups"][g].items():
            result[f"group_{g}_{metric}"] = float(value)
    if gamma==0:
        if not np.array_equal(d0,d1) or bias != 0 or strict_reversal:
            raise AssertionError("Zero-mediation negative control failed")
    return result


def one_paired_replicate(alpha: float, gammas: list[float], rep: int, *,
                         intercepts, seed: int=733, n_train: int=2500,
                         n_cal: int=1200, n_test: int=2400, pool: int=16,
                         slots: int=5, fnr_target: float=.06,
                         random_repeats: int=3) -> list[dict]:
    """One model fit per (alpha, rep), shared for all gammas in that block."""
    rseed=seed+101*rep
    rng=np.random.default_rng(rseed)
    param0=params_for(alpha,0.)
    train0,cal0,test0=(generate(n,param0,intercepts,rng)
                       for n in (n_train,n_cal,n_test))
    frozen=prepare_frozen_policy(train0,cal0,test0,pool=pool,slots=slots,
                                 fnr_target=fnr_target,
                                 random_repeats=random_repeats,seed=rseed)
    rows=[]
    for gamma in gammas:
        rng_gamma=np.random.default_rng(rseed)
        param=params_for(alpha,float(gamma))
        tr,ca,te=(generate(n,param,intercepts,rng_gamma)
                  for n in (n_train,n_cal,n_test))
        # Prevent silent model/policy adaptation or changed patients across gamma.
        for name,observed,reference in [
            ("train features",tr[FEATURES],train0[FEATURES]),
            ("calibration features",ca[FEATURES],cal0[FEATURES]),
            ("test features",te[FEATURES],test0[FEATURES]),
            ("train D0",tr.D0,train0.D0),
            ("cal D0",ca.D0,cal0.D0),
            ("test D0",te.D0,test0.D0),
            ("test S",te.S,test0.S),("test O",te.O,test0.O),
        ]:
            if not np.array_equal(np.asarray(observed),np.asarray(reference)):
                raise AssertionError(f"Shared exogenous samples violation: {name}")
        rows.append(evaluate_fixed(te,frozen,alpha=alpha,gamma=gamma,rep=rep,seed=rseed))
    return rows


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Across-replicate t intervals, NOT patient-level/seed-level confidence bounds."""
    items=[]
    for (alpha,gamma),h in df.groupby(["alpha","gamma"],sort=True):
        q={"alpha":float(alpha),"gamma":float(gamma),"n_reps":len(h),
           "ranking_strict_reversal_share":float(h.policy_ranking_strict_reversal.mean())}
        for key in ("delta_retrospective_error","delta_deployment_error",
                    "retrospective_to_deployment_gap","observed_D1_vs_D0_change",
                    "rho_margin","rho_random","bound_rho_sum",
                    "margin_label_correction","random_label_correction",
                    "mean_changed_on_prediction_disagreement_rate"):
            x=h[key].to_numpy(float)
            q[key+"_mean"]=float(np.mean(x))
            se=float(np.std(x,ddof=1)/np.sqrt(len(x))) if len(x)>1 else np.nan
            q[key+"_ci95_lower"]=float(np.mean(x)-student_t.ppf(.975,len(x)-1)*se) if len(x)>1 else np.nan
            q[key+"_ci95_upper"]=float(np.mean(x)+student_t.ppf(.975,len(x)-1)*se) if len(x)>1 else np.nan
        items.append(q)
    return pd.DataFrame(items)


def paired_gamma_shifts(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Within-(alpha, rep) paired contrasts relative to the gamma=0 world.

    Descriptive t intervals are across independent replicate worlds, with all
    gamma cells of one replicate sharing the same test patients and predictors.
    """
    h0=df[df.gamma==0.0][['alpha','rep','delta_retrospective_error',
                           'delta_deployment_error','retrospective_to_deployment_gap']].copy()
    if len(h0)!=df.groupby(['alpha','rep']).ngroups:
        raise ValueError('Paired gamma shifts require a gamma=0 reference per alpha and replicate')
    j=df.merge(h0,on=['alpha','rep'],validate='many_to_one',suffixes=('','_at_gamma0'))
    for field in ('delta_retrospective_error','delta_deployment_error',
                  'retrospective_to_deployment_gap'):
        j[field+'_change_from_gamma0']=j[field]-j[field+'_at_gamma0']
    by=[]
    for (alpha,gamma),h in j.groupby(['alpha','gamma'],sort=True):
        q={'alpha':float(alpha),'gamma':float(gamma),'n_reps':int(len(h))}
        for field in ('delta_retrospective_error_change_from_gamma0',
                      'delta_deployment_error_change_from_gamma0',
                      'retrospective_to_deployment_gap_change_from_gamma0'):
            x=h[field].to_numpy(float)
            se=float(x.std(ddof=1)/np.sqrt(len(x))) if len(x)>1 else np.nan
            ci=student_t.ppf(.975,len(x)-1)*se if len(x)>1 else np.nan
            q[field+'_mean']=float(x.mean())
            q[field+'_ci95_lower']=float(x.mean()-ci)
            q[field+'_ci95_upper']=float(x.mean()+ci)
        by.append(q)
    keep=['alpha','gamma','rep']+[f+'_change_from_gamma0' for f in (
           'delta_retrospective_error','delta_deployment_error','retrospective_to_deployment_gap')]
    return j[keep],pd.DataFrame(by)


def verify_dataframe(df: pd.DataFrame, manifest: dict) -> None:
    expected={(a,g,r) for a in manifest['alphas'] for g in manifest['gammas']
              for r in range(manifest['replicates'])}
    keys=[(float(x.alpha),float(x.gamma),int(x.rep)) for x in df.itertuples()]
    if len(keys)!=len(expected) or set(keys)!=expected:
        raise AssertionError("Missing or duplicated alpha × gamma × replicate cells")
    for (a,r),h in df.groupby(['alpha','rep']):
        for field in ('predictor_sha256','decision_sha256','assignment_sha256',
                      'margin_center','margin_threshold','random_threshold',
                      'd0_test_sha256','s_test_sha256','o_test_sha256'):
            if h[field].nunique()!=1:
                raise AssertionError(f"Fixed-predictor violation at alpha={a}, rep={r}: {field}")
    for rep,h in df.groupby('rep'):
        for field in ('d0_test_sha256','s_test_sha256','o_test_sha256'):
            if h[field].nunique()!=1:
                raise AssertionError(f"Paired D0/S/O violation rep={rep}: {field}")
    if not df.label_instability_bound_satisfied.all():
        raise AssertionError("A label instability bound was violated")
    if 0.0 in manifest['gammas']:
        h0=df.loc[df.gamma==0.0]
        if not (h0.observed_D1_vs_D0_change.eq(0).all()
                and h0.retrospective_to_deployment_gap.abs().lt(1e-12).all()):
            raise AssertionError("Gamma=0 negative control did not hold")


def compare_refit(df: pd.DataFrame, path: Path, manifest: dict) -> tuple[pd.DataFrame,pd.DataFrame]:
    """Only compare with EXACT paired, v2 fixed-D0 refit+retune pilot."""
    old_manifest_path=path/'manifest.json'
    old_csv=path/'replicate_metrics.csv'
    if not old_manifest_path.is_file() or not old_csv.is_file():
        raise FileNotFoundError("Adaptive pilot must contain manifest.json and replicate_metrics.csv")
    m=json.loads(old_manifest_path.read_text())
    if m.get('version')!='label-mechanism-factorial-v2-fixed-D0-paired-CRN':
        raise ValueError("Refit comparison requires v2 fixed-D0/paired-CRN pilot")
    fields=(('alphas','alphas'),('gammas','gammas'),('replicates','replicates'),
            ('n_train','n_train'),('n_cal','n_cal'),('n_test','n_test'),
            ('intercept_n','intercept_n'),('pool','pool'),('slots','slots'),
            ('random_repeats','random_repeats'),('fnr_target','fnr_target'),('seed','seed'))
    for a,b in fields:
        if m.get(a)!=manifest.get(b):
            raise ValueError(f"Refit pairing mismatch in {a}: {m.get(a)} vs {manifest.get(b)}")
    old=pd.read_csv(old_csv)
    old=old.loc[old.label=='D'].copy()
    if old.duplicated(['alpha','gamma','rep']).any():
        raise ValueError("Duplicated D cells in refit input")
    cols=['alpha','gamma','rep','d0_test_sha256','retrospective_to_deployment_gap',
          'delta_retrospective_error','delta_deployment_error']
    if any(c not in old for c in cols):
        raise ValueError("Incomplete refit input columns")
    paired=df.merge(old[cols],on=['alpha','gamma','rep'],how='outer',indicator=True,
                    suffixes=('_fixed','_refit'),validate='one_to_one')
    if not paired._merge.eq('both').all() or len(paired)!=len(df):
        raise ValueError("Refit and fixed data have different cell sets")
    if not paired.d0_test_sha256_fixed.eq(paired.d0_test_sha256_refit).all():
        raise ValueError("Refit D0 hashes differ: not the same simulated patients")
    paired['gap_refit_minus_fixed']=(paired.retrospective_to_deployment_gap_refit-
                                      paired.retrospective_to_deployment_gap_fixed)
    by=[]
    for (alpha,gamma),h in paired.groupby(['alpha','gamma'],sort=True):
        x=h.gap_refit_minus_fixed.to_numpy(float)
        se=float(np.std(x,ddof=1)/np.sqrt(len(x))) if len(x)>1 else np.nan
        ci=student_t.ppf(.975,len(x)-1)*se if len(x)>1 else np.nan
        by.append({'alpha':alpha,'gamma':gamma,'n_reps':len(x),
                   'refit_minus_fixed_gap_mean':float(np.mean(x)),
                   'ci95_lower':float(np.mean(x)-ci),
                   'ci95_upper':float(np.mean(x)+ci)})
    result=paired[['alpha','gamma','rep','retrospective_to_deployment_gap_fixed',
                   'retrospective_to_deployment_gap_refit','gap_refit_minus_fixed']].copy()
    return result,pd.DataFrame(by)


def _numbers_exact(s: str) -> list[float]:
    vals=_numbers(s)
    return [float(v) for v in vals]


def run(args) -> None:
    mechanism_source=Path(__file__).with_name('mechanism_grid.py').read_text()
    if 'label-mechanism-factorial-v2-fixed-D0-paired-CRN' not in mechanism_source:
        raise RuntimeError('Install Mechanism v2 fixed-D0 / paired-CRN patch first')
    if (args.reps<1 or args.n_train<100 or args.n_cal<args.pool
        or args.n_test<args.pool or not 0<args.slots<args.pool
        or args.random_repeats<1 or args.intercept_n<100):
        raise ValueError("Invalid sizes or capacity")
    alphas,gammas=_numbers_exact(args.alphas),_numbers_exact(args.gammas)
    code_files=[Path(__file__),Path(__file__).with_name('mechanism_grid.py'),
                Path(__file__).with_name('world.py'),
                Path(__file__).with_name('assignment_contributions.py'),
                Path(__file__).resolve().parent.parent/'src/sharedcap_afa/revision/engine.py']
    source_hashes={str(f.relative_to(Path(__file__).resolve().parent.parent)):hashlib.sha256(f.read_bytes()).hexdigest()
                   for f in code_files}
    manifest={'version':VERSION,'alphas':alphas,'gammas':gammas,
              'source_sha256':source_hashes,
              'environment_versions':{'python':sys.version.split()[0],
                  'numpy':np.__version__,'pandas':pd.__version__,
                  'scipy':scipy.__version__,'sklearn':sklearn.__version__},
              'replicates':int(args.reps),'n_train':args.n_train,
              'n_cal':args.n_cal,'n_test':args.n_test,'intercept_n':args.intercept_n,
              'pool':args.pool,'slots':args.slots,'fnr_target':float(args.fnr_target),
              'random_repeats':args.random_repeats,'seed':args.seed,
              'predictor_fit_gamma':0.,'fit_label':'D0',
              'selection_calibration_label':'D0',
              'frozen_across_gamma':['p0','p1','margin_center','thresholds',
                                     'pool_membership','margin_and_random_selections',
                                     'binary_predictions'],
              'labels':'retro=D1(gamma), deployment=where(selected,D1(gamma),D0)',
              'analysis_status':'SYNTHETIC EXPLORATORY; not clinical causal identification',
              'safety_gate_status':'NO CLAIM OF FNR SAFETY CERTIFICATE'}
    out=Path(args.out_dir)
    mf=out/'manifest.json'
    if out.exists() and any(out.iterdir()):
        if not args.resume:
            raise FileExistsError(f"Refusing to overwrite {out}; use a NEW directory or --resume")
        if not mf.exists() or json.loads(mf.read_text())!=manifest:
            raise ValueError("Resume manifest does not match new experiment")
    else:
        out.mkdir(parents=True,exist_ok=True)
        mf.write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
    cells=out/'cells';cells.mkdir(exist_ok=True)
    ic=calibrate_intercepts(params_for(0.,0.),n=args.intercept_n,
                            seed=args.seed+10000)
    rows=[]
    for ai,alpha in enumerate(alphas):
        for rep in range(args.reps):
            f=cells/f'a{ai}_rep{rep}.json'
            if not f.exists():
                r=one_paired_replicate(alpha,gammas,rep,intercepts=ic,
                                       seed=args.seed,n_train=args.n_train,
                                       n_cal=args.n_cal,n_test=args.n_test,
                                       pool=args.pool,slots=args.slots,
                                       fnr_target=args.fnr_target,
                                       random_repeats=args.random_repeats)
                tmp=f.with_suffix('.tmp')
                tmp.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
                os.replace(tmp,f)
            v=json.loads(f.read_text())
            if len(v)!=len(gammas) or [float(x['gamma']) for x in v]!=gammas:
                raise ValueError(f"Incomplete/reordered gamma cells in {f}")
            rows.extend(v)
        print(f"alpha={alpha:g}: {args.reps} paired replications complete",flush=True)
    df=pd.DataFrame(rows)
    verify_dataframe(df,manifest)
    # Export only synthetic aggregates; rep-level values have no real patient IDs.
    df.to_csv(out/'replicate_metrics.csv',index=False)
    aggregate(df).to_csv(out/'summary.csv',index=False)
    if 0.0 in gammas:
        shifts, shifts_summary = paired_gamma_shifts(df)
        shifts.to_csv(out/'paired_gamma_shifts.csv',index=False)
        shifts_summary.to_csv(out/'paired_gamma_shifts_summary.csv',index=False)
    if args.adaptive_run:
        comp,comp_sum=compare_refit(df,Path(args.adaptive_run),manifest)
        comp.to_csv(out/'refit_comparison.csv',index=False)
        comp_sum.to_csv(out/'refit_comparison_summary.csv',index=False)
    print(f"Saved fixed-predictor synthetic metrics to {out}")
    print("Gamma changes D1 only; predictions, policy selections, and thresholds remain frozen.")


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--alphas',default='0,0.5,1')
    parser.add_argument('--gammas',default='0,0.5,1')
    parser.add_argument('--reps',type=int,default=10)
    parser.add_argument('--n-train',type=int,default=2500)
    parser.add_argument('--n-cal',type=int,default=1200)
    parser.add_argument('--n-test',type=int,default=2400)
    parser.add_argument('--intercept-n',type=int,default=25000)
    parser.add_argument('--pool',type=int,default=16)
    parser.add_argument('--slots',type=int,default=5)
    parser.add_argument('--fnr-target',type=float,default=.06)
    parser.add_argument('--random-repeats',type=int,default=3)
    parser.add_argument('--seed',type=int,default=733)
    parser.add_argument('--out-dir',default='outputs/research_label_mechanism/fixed_predictor_01')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--adaptive-run',default=None,
                        help='Optional v2 fixed-D0 refit+retune run directory, with matching manifest')
    a=parser.parse_args(argv)
    run(a)

if __name__=='__main__':
    main()
