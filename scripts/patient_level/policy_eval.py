"""Patient-level estimators for comparing capacity-constrained allocation policies.

Designed to be run by the data custodians (MIMIC-IV/MIMIC-IV-ED, MC-MED credentialed users).
Nothing here needs network access or the aggregate results package.

Implements (manuscript section in brackets):
  * complete random pools, margin and random allocation                       [2.1]
  * order-statistic and grid threshold selection                              [3.2]
  * empirical operating characteristic, lower convex hull, frontier
    decomposition A + O + residual (Proposition 1) and matched-sensitivity
    averages / value-of-information share                                     [2.3, 3.1]
  * nested pool bootstrap that re-selects thresholds on resampled calibration
    pools and re-draws Random allocations                                     [3.2]
  * evaluation-gap identity (Theorem 1), closed form (Corollary 1) and
    bound (Corollary 2) for cost-weighted loss                                [3.3]
  * closed-form unstructured label-flip bounds (Proposition 3)                [3.4]
"""
from __future__ import annotations
import math
import numpy as np

# ------------------------------------------------------------------ pools & allocation
def make_pools(n: int, m: int, rng: np.random.Generator) -> np.ndarray:
    """Random partition of range(n) into complete pools of size m (remainder discarded); shape (J, m)."""
    idx = rng.permutation(n)
    J = n // m
    return idx[: J * m].reshape(J, m)


def allocate_margin(p0: np.ndarray, pools: np.ndarray, K: int, center: float) -> np.ndarray:
    """Allocate, within each pool, the K encounters whose no-panel score is closest to `center`."""
    A = np.zeros(p0.shape[0], bool)
    dist = np.abs(p0[pools] - center)
    pick = np.argsort(dist, axis=1, kind="stable")[:, :K]
    A[np.take_along_axis(pools, pick, axis=1).ravel()] = True
    return A


def allocate_random(pools: np.ndarray, K: int, rng: np.random.Generator, n: int, draws: int = 10) -> np.ndarray:
    """`draws` independent uniform allocations of K per pool; boolean array of shape (draws, n)."""
    J, m = pools.shape
    out = np.zeros((draws, n), bool)
    for r in range(draws):
        pick = np.argsort(rng.random((J, m)), axis=1)[:, :K]
        out[r, np.take_along_axis(pools, pick, axis=1).ravel()] = True
    return out


def policy_scores(p0, p1, A):
    return np.where(A, p1, p0)

# ------------------------------------------------------------------ thresholds
def threshold_orderstat(pos_scores: np.ndarray, q: float) -> float:
    """Largest threshold t with calibration FNR = #{s < t}/n1 <= q, for predictions 1{s >= t}.

    With sorted positive scores s_(1) <= ... <= s_(n1) and k = floor(q n1) the answer is s_(k+1).
    For a randomised policy pass the positive scores pooled over allocation draws."""
    s = np.sort(np.asarray(pos_scores, float).ravel())
    k = int(math.floor(q * s.size + 1e-12))
    return float(s[min(k, s.size - 1)]) if k < s.size else float(np.inf)


def threshold_grid(pos_scores: np.ndarray, q: float, start: float = 0.001, step: float = 0.0025) -> float:
    """Original rule: largest grid value whose calibration FNR does not exceed q."""
    s = np.sort(np.asarray(pos_scores, float).ravel())
    grid = np.arange(start, 1.0, step)
    fnr = np.searchsorted(s, grid, side="left") / s.size
    ok = np.flatnonzero(fnr <= q + 1e-12)
    return float(grid[ok.max()]) if ok.size else float(grid[0])

# ------------------------------------------------------------------ operating characteristic
def rates(y: np.ndarray, s: np.ndarray, t: float):
    """(FNR, FPR) of 1{s >= t}; y and s may be 2-D (draws x n) for randomised policies."""
    y = np.asarray(y); s = np.asarray(s)
    pos, neg = y == 1, y == 0
    return float(np.mean(s[pos] < t)), float(np.mean(s[neg] >= t))


def empirical_frontier(y: np.ndarray, s: np.ndarray):
    """All attainable deterministic operating points (FNR, FPR) over thresholds at the observed scores."""
    y = np.asarray(y).ravel(); s = np.asarray(s, float).ravel()
    sp, sn = np.sort(s[y == 1]), np.sort(s[y == 0])
    t = np.unique(np.concatenate([s, [np.inf]]))
    fnr = np.searchsorted(sp, t, side="left") / sp.size
    fpr = 1.0 - np.searchsorted(sn, t, side="left") / sn.size
    return fnr, fpr


def lower_hull(x, y) -> np.ndarray:
    order = np.lexsort((y, x)); x, y = np.asarray(x)[order], np.asarray(y)[order]
    keep_x, keep_y = [], []
    for xi, yi in zip(x, y):          # min y per x
        if keep_x and xi == keep_x[-1]:
            continue
        keep_x.append(xi); keep_y.append(yi)
    hull: list[tuple[float, float]] = []
    for p in zip(keep_x, keep_y):
        while len(hull) >= 2:
            (x1, y1), (x2, y2) = hull[-2], hull[-1]
            if (x2 - x1) * (p[1] - y1) - (y2 - y1) * (p[0] - x1) <= 0:
                hull.pop()
            else:
                break
        hull.append(p)
    return np.array(hull)


def F(hull: np.ndarray, f: float) -> float:
    x, y = hull[:, 0], hull[:, 1]
    return float(np.interp(f, x, y)) if x.min() - 1e-12 <= f <= x.max() + 1e-12 else float("nan")


def frontier_decomposition(hull_M, hull_R, fM, yM, fR, yR) -> dict:
    """Proposition 1: Delta_proc = A + O + (r_M - r_R)."""
    a, b, c, d = F(hull_M, fM), F(hull_M, fR), F(hull_R, fM), F(hull_R, fR)
    A = 0.5 * ((a - c) + (b - d)); O = 0.5 * ((a - b) + (c - d)); res = (yM - a) - (yR - d)
    sbar = -O / (fM - fR) if abs(fM - fR) > 1e-12 else float("nan")
    return dict(proc=yM - yR, A=A, O=O, residual=res, slope=sbar)


def matched_average(hull_M, hull_R, lo=0.02, hi=0.05, n=301) -> float:
    lo = max(lo, hull_M[:, 0].min(), hull_R[:, 0].min()); hi = min(hi, hull_M[:, 0].max(), hull_R[:, 0].max())
    g = np.linspace(lo, hi, n)
    return float(np.mean([F(hull_M, f) - F(hull_R, f) for f in g]))


def voi_share(hull_P, hull_0, hull_1, lo=0.02, hi=0.05, n=301, min_headroom=0.02) -> float:
    lo = max(lo, *(h[:, 0].min() for h in (hull_P, hull_0, hull_1))); hi = min(hi, *(h[:, 0].max() for h in (hull_P, hull_0, hull_1)))
    g = np.linspace(lo, hi, n)
    num = np.mean([F(hull_0, f) - F(hull_P, f) for f in g]); den = np.mean([F(hull_0, f) - F(hull_1, f) for f in g])
    return float(num / den) if den > min_headroom else float("nan")

# ------------------------------------------------------------------ full comparison and nested bootstrap
def compare_policies(cal: dict, test: dict, q: float, K: int, center: float, rng, draws=10, rule="orderstat") -> dict:
    """cal/test: dicts with arrays y, p0, p1 and pools (J x m).  Returns procedure contrast and decomposition."""
    sel = threshold_orderstat if rule == "orderstat" else threshold_grid
    out = {}
    for name, dat in (("cal", cal), ("test", test)):
        n = dat["y"].size
        AM = allocate_margin(dat["p0"], dat["pools"], K, center)
        AR = allocate_random(dat["pools"], K, rng, n, draws)
        out[name] = dict(sM=policy_scores(dat["p0"], dat["p1"], AM), sR=policy_scores(dat["p0"][None], dat["p1"][None], AR),
                         y=dat["y"], yR=np.broadcast_to(dat["y"], (draws, n)))
    c, t = out["cal"], out["test"]
    tM = sel(c["sM"][c["y"] == 1], q); tR = sel(c["sR"][c["yR"] == 1], q)
    fM, yM = rates(t["y"], t["sM"], tM); fR, yR = rates(t["yR"], t["sR"], tR)
    hM = lower_hull(*empirical_frontier(t["y"], t["sM"])); hR = lower_hull(*empirical_frontier(t["yR"], t["sR"]))
    dec = frontier_decomposition(hM, hR, fM, yM, fR, yR)
    dec.update(tM=tM, tR=tR, fM=fM, fR=fR, iso_avg=matched_average(hM, hR))
    return dec


def _resample_pools(dat: dict, rng) -> dict:
    J = dat["pools"].shape[0]
    pick = dat["pools"][rng.integers(0, J, J)]
    idx = pick.ravel()
    new = {k: np.asarray(v)[idx] for k, v in dat.items() if k != "pools"}
    new["pools"] = np.arange(idx.size).reshape(pick.shape)
    return new


def nested_pool_bootstrap(cal: dict, test: dict, q: float, K: int, center: float, B: int = 1000, seed: int = 1,
                          draws: int = 10, rule: str = "orderstat") -> dict:
    """Each replicate resamples calibration pools (thresholds re-selected), test pools and Random draws."""
    rng = np.random.default_rng(seed)
    keys = ("proc", "A", "O", "residual", "iso_avg")
    reps = {k: np.empty(B) for k in keys}
    for b in range(B):
        r = compare_policies(_resample_pools(cal, rng), _resample_pools(test, rng), q, K, center, rng, draws, rule)
        for k in keys:
            reps[k][b] = r[k]
    return {k: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))) for k, v in reps.items()} | {"replicates": reps}

# ------------------------------------------------------------------ decision-dependent labels
def w_c(h, c=0.5):
    h = np.asarray(h, float)
    return c * h - (1 - c) * (1 - h)


def loss_c(h, y, c=0.5):
    h = np.asarray(h, float); y = np.asarray(y, float)
    return c * h * (1 - y) + (1 - c) * (1 - h) * y


def gap_bruteforce(D0, D1, AM, hM, AR_draws, hR_draws, c=0.5) -> float:
    """B = (R_M^dep - R_R^dep) - (R_M^ret - R_R^ret) by direct evaluation."""
    ret = lambda h: np.mean(loss_c(h, D1, c))
    dep = lambda h, a: np.mean(loss_c(h, np.where(a, D1, D0), c))
    RMr, RMd = ret(hM), dep(hM, AM)
    RRr = np.mean([ret(h) for h in hR_draws]); RRd = np.mean([dep(h, a) for a, h in zip(AR_draws, hR_draws)])
    return float((RMd - RRd) - (RMr - RRr))


def gap_identity(D0, D1, AM, h0M, AR_draws, h0R, c=0.5) -> float:
    """Theorem 1 (finite set of Random draws): B = mean(delta [(1-A_M) w(h0_M) - mean_r(1-A_R) w(h0_R)])."""
    delta = np.asarray(D1, float) - np.asarray(D0, float)
    return float(np.mean(delta * ((1 - AM) * w_c(h0M, c) - (1 - np.asarray(AR_draws, float)).mean(0) * w_c(h0R, c))))


def gap_closed_form(D0, D1, AM, h0R, pi, c=0.5) -> float:
    """Corollary 1: B = pi (1 - pi) (gbar_0 - gbar_1), g = delta * w_c(h0_R) (expectation over Random)."""
    g = (np.asarray(D1, float) - np.asarray(D0, float)) * w_c(h0R, c)
    return float(pi * (1 - pi) * (g[~AM].mean() - g[AM].mean()))


def gap_bound(D0, D1, AM, AR_draws, h0M, h0R, c=0.5) -> float:
    """Corollary 2 (finite draws): max(c,1-c) P(delta!=0, A_M!=A_R) + P(delta!=0, both unallocated, h0 differ)."""
    ch = np.asarray(D1) != np.asarray(D0)
    AR = np.asarray(AR_draws, bool)
    single = np.mean([(ch & (AM != a)).mean() for a in AR])
    both0 = np.mean([(ch & ~AM & ~a & (h0M != h0R)).mean() for a in AR])
    return float(max(c, 1 - c) * single + both0)

# ------------------------------------------------------------------ unstructured label flips
def label_flip_bounds(hM, hR, y, k: int, common: bool = True):
    """Proposition 3 (0-1 loss): attainable interval for the error difference after <= k label changes."""
    hM, hR, y = (np.asarray(v).astype(int) for v in (hM, hR, y))
    n = y.size; eM, eR = np.mean(hM != y), np.mean(hR != y)
    Dw = int(np.sum((hM != y) & (hR == y))); Db = int(np.sum((hM == y) & (hR != y)))
    d = (Dw - Db) / n
    if common:
        return d - 2 * min(k, Dw) / n, d + 2 * min(k, Db) / n
    return d - min(k / n, eM) - min(k / n, 1 - eR), d + min(k / n, 1 - eM) + min(k / n, eR)


def min_flips_to_tie(hM, hR, y) -> int:
    hM, hR, y = (np.asarray(v).astype(int) for v in (hM, hR, y))
    Dw = int(np.sum((hM != y) & (hR == y))); Db = int(np.sum((hM == y) & (hR != y)))
    return int(math.ceil(abs(Db - Dw) / 2))
