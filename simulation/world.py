"""Synthetic emergency department with known ground truth.

The point of the simulation is to construct two "worlds" that look alike under
the original evaluation (disposition label, CBC masked from the model) but
differ in why the CBC appears valuable:

* world P ("prognostic"): CBC abnormalities mostly reflect hidden acute
  severity W, which also drives the patient's true need S and the hard outcome O;
* world M ("decision-mediated"): CBC abnormalities mostly reflect a chronic,
  non-prognostic component V, but clinicians admit strongly on CBC findings.

In both worlds the disposition D is a clinician decision made *after* seeing
the CBC.  The CBC-blind decision D0 (same patient, same random draw, no CBC
terms) is available only in simulation and defines the CBC-induced admissions.

Variables (one row per ED visit)
    x0..x5      triage features (observed before testing)
    hgb,wbc,plt standardised CBC components (low hgb/plt and high wbc are abnormal)
    t1,t2       CBC finding tiers (critical / markedly abnormal), set by quantiles
    W,V,U       hidden acute severity, chronic CBC component, clinician bedside impression
    S           true need for hospital-level care (oracle label)
    D, D0       disposition with and without the CBC (D observed, D0 counterfactual)
    O           hard outcome (critical event), optionally reduced by CBC-triggered treatment
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm

BETA = np.array([0.9, 0.6, 0.5, 0.4, 0.3, 0.2])


@dataclass(frozen=True)
class WorldParams:
    name: str
    rho_hW: float      # haemoglobin loading on acute severity
    rho_hV: float      # haemoglobin loading on chronic component
    rho_wW: float      # WBC loading on acute severity
    rho_pW: float      # platelet loading on acute severity
    rho_pV: float      # platelet loading on chronic component
    kS: float = 1.3    # effect of W on need
    aU: float = 0.9    # clinician use of bedside impression
    aM: float = 0.5    # clinician use of the CBC abnormality index
    g1: float = 1.0    # admission push of a critical CBC finding
    g2: float = 0.4    # admission push of a markedly abnormal CBC finding
    oR: float = 1.0    # hard-outcome dependence on triage risk
    oW: float = 1.4    # hard-outcome dependence on acute severity
    eps: float = 0.0   # relative risk reduction of CBC-triggered treatment on O
    target_D: float = 0.45
    target_S: float = 0.40
    target_O: float = 0.06
    q_t1: tuple = (0.008, 0.006, 0.008)   # tail probabilities (hgb low, wbc high, plt low)
    q_t2: tuple = (0.08, 0.05, 0.03)


# Parameters chosen so that both worlds show a similar, realistic benefit of universal
# CBC under the disposition label (about 5-6 FPR points at a 6% FNR target, close to
# the 4.3-5.6 points observed in MIMIC and MC-MED), while differing in why.
WORLD_P = WorldParams("prognostic", rho_hW=0.32, rho_hV=0.20, rho_wW=0.32, rho_pW=0.27, rho_pV=0.15,
                      aM=0.30, g1=0.8, g2=0.3)
WORLD_M = WorldParams("decision-mediated", rho_hW=0.10, rho_hV=0.75, rho_wW=0.15, rho_pW=0.10, rho_pV=0.70,
                      aM=0.30, g1=3.0, g2=1.7)


def mix(a: WorldParams, b: WorldParams, lam: float) -> WorldParams:
    """Linear interpolation between two worlds (lam=0 -> a, lam=1 -> b)."""
    f = lambda x, y: float((1 - lam) * x + lam * y)
    return replace(a, name=f"mix{lam:.2f}", rho_hW=f(a.rho_hW, b.rho_hW), rho_hV=f(a.rho_hV, b.rho_hV),
                   rho_wW=f(a.rho_wW, b.rho_wW), rho_pW=f(a.rho_pW, b.rho_pW), rho_pV=f(a.rho_pV, b.rho_pV),
                   aM=f(a.aM, b.aM), g1=f(a.g1, b.g1), g2=f(a.g2, b.g2))


def _latent(n: int, P: WorldParams, rng: np.random.Generator) -> dict:
    X0 = rng.normal(size=(n, 6))
    r = X0 @ BETA
    W, V = rng.normal(size=n), rng.normal(size=n)
    e = rng.normal(size=(n, 3))
    s_h = np.sqrt(max(1e-6, 1 - P.rho_hW ** 2 - P.rho_hV ** 2))
    s_w = np.sqrt(max(1e-6, 1 - P.rho_wW ** 2))
    s_p = np.sqrt(max(1e-6, 1 - P.rho_pW ** 2 - P.rho_pV ** 2))
    hgb = -(P.rho_hW * W + P.rho_hV * V) + s_h * e[:, 0]
    wbc = P.rho_wW * W + s_w * e[:, 1]
    plt = -(P.rho_pW * W + P.rho_pV * V) + s_p * e[:, 2]
    zl = lambda q: norm.ppf(q)
    t1 = (hgb < zl(P.q_t1[0])) | (wbc > -zl(P.q_t1[1])) | (plt < zl(P.q_t1[2]))
    t2 = (hgb < zl(P.q_t2[0])) | (wbc > -zl(P.q_t2[1])) | (plt < zl(P.q_t2[2]))
    U = W + 0.8 * rng.normal(size=n)
    m = (-hgb + wbc - plt) / np.sqrt(3.0)
    return dict(X0=X0, r=r, W=W, V=V, hgb=hgb, wbc=wbc, plt=plt, t1=t1, t2=t2, U=U, m=m,
                uD=rng.random(n), uS=rng.random(n), uO=rng.random(n))


def _solve_intercept(f, target: float) -> float:
    lo, hi = -15.0, 15.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if f(mid) > target:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


@dataclass(frozen=True)
class Intercepts:
    d0: float
    s0: float
    o0: float


def calibrate_intercepts(P: WorldParams, n: int = 400_000, seed: int = 999) -> Intercepts:
    z = _latent(n, P, np.random.default_rng(seed))
    lin_D = z["r"] + P.aU * z["U"] + P.aM * z["m"] + P.g1 * z["t1"] + P.g2 * z["t2"]
    d0 = _solve_intercept(lambda c: expit(c + lin_D).mean(), P.target_D)
    s0 = _solve_intercept(lambda c: expit(c + z["r"] + P.kS * z["W"]).mean(), P.target_S)
    o0 = _solve_intercept(lambda c: expit(c + P.oR * z["r"] + P.oW * z["W"]).mean(), P.target_O)
    return Intercepts(d0, s0, o0)


def generate(n: int, P: WorldParams, ic: Intercepts, rng: np.random.Generator) -> pd.DataFrame:
    z = _latent(n, P, rng)
    lin_D = ic.d0 + z["r"] + P.aU * z["U"] + P.aM * z["m"] + P.g1 * z["t1"] + P.g2 * z["t2"]
    lin_D0 = ic.d0 + z["r"] + P.aU * z["U"]
    D = z["uD"] < expit(lin_D)
    D0 = z["uD"] < expit(lin_D0)
    S = z["uS"] < expit(ic.s0 + z["r"] + P.kS * z["W"])
    pO = expit(ic.o0 + P.oR * z["r"] + P.oW * z["W"]) * (1 - P.eps * (D & z["t2"]))
    O = z["uO"] < pO
    df = pd.DataFrame(z["X0"], columns=[f"x{i}" for i in range(6)])
    for k in ("hgb", "wbc", "plt", "W", "V", "U"):
        df[k] = z[k]
    df["t1"], df["t2"] = z["t1"], z["t2"]
    df["D"], df["D0"], df["S"], df["O"] = D.astype(int), D0.astype(int), S.astype(int), O.astype(int)
    return df


FREE = [f"x{i}" for i in range(6)]
CBC = ["hgb", "wbc", "plt"]
