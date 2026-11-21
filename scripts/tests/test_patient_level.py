"""Self-tests of the patient-level toolkit on synthetic data. Run: python scripts/tests/test_patient_level.py"""
import itertools, sys, math
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from patient_level import policy_eval as pe


def synth(n=4800, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n); z = rng.normal(size=n)
    p0 = 1 / (1 + np.exp(-(-1.0 + 1.2 * x))); p1 = 1 / (1 + np.exp(-(-1.0 + 1.2 * x + 0.9 * z)))
    D0 = (rng.random(n) < p0).astype(int)
    flip_up = (rng.random(n) < 0.06 * (z > 0.5)); flip_dn = (rng.random(n) < 0.04 * (z < -0.5))
    D1 = np.where(flip_up, 1, np.where(flip_dn, 0, D0))
    return rng, p0, p1, D0, D1


def test_identity_and_closed_form():
    rng, p0, p1, D0, D1 = synth()
    pools = pe.make_pools(p0.size, 16, rng); K = 5; pi = K / 16
    keep = pools.ravel(); p0, p1, D0, D1 = p0[keep], p1[keep], D0[keep], D1[keep]; pools = np.arange(keep.size).reshape(pools.shape)
    t = float(np.quantile(p0, 0.6)); AM = pe.allocate_margin(p0, pools, K, t)
    AR = pe.allocate_random(pools, K, rng, p0.size, draws=400)
    hM = (pe.policy_scores(p0, p1, AM) >= t).astype(int)
    hR = (pe.policy_scores(p0[None], p1[None], AR) >= t).astype(int)
    h0 = (p0 >= t).astype(int)
    for c in (0.5, 0.2, 0.8):
        B = pe.gap_bruteforce(D0, D1, AM, hM, AR, hR, c)
        assert abs(B - pe.gap_identity(D0, D1, AM, h0, AR, h0, c)) < 1e-12, "Theorem 1"
        # exact expectation over Random: P(A_R=1)=pi for every encounter
        Bexp = pe.gap_identity(D0, D1, AM, h0, np.full((1, p0.size), pi), h0, c)
        assert abs(Bexp - pe.gap_closed_form(D0, D1, AM, h0, pi, c)) < 1e-12, "Corollary 1"
        assert abs(B) <= pe.gap_bound(D0, D1, AM, AR, h0, h0, c) + 1e-12, "Corollary 2"


def test_label_flip_closed_form_bruteforce():
    rng = np.random.default_rng(3)
    for _ in range(60):
        n = 9; y = rng.integers(0, 2, n); hM = rng.integers(0, 2, n); hR = rng.integers(0, 2, n)
        err = lambda h, yy: np.mean(h != yy)
        for k in range(0, 4):
            vals = []
            for S in itertools.chain.from_iterable(itertools.combinations(range(n), r) for r in range(k + 1)):
                yy = y.copy(); yy[list(S)] = 1 - yy[list(S)]
                vals.append(err(hM, yy) - err(hR, yy))
            lo, hi = pe.label_flip_bounds(hM, hR, y, k, common=True)
            assert abs(min(vals) - lo) < 1e-12 and abs(max(vals) - hi) < 1e-12, "Proposition 3(a)"


def test_orderstat_threshold():
    rng = np.random.default_rng(5)
    s = rng.random(304)
    for q in (0.02, 0.06, 0.12):
        t = pe.threshold_orderstat(s, q)
        assert np.mean(s < t) <= q + 1e-12
        assert np.mean(s < np.nextafter(t, 2)) > q, "maximality"
        assert pe.threshold_grid(s, q) <= t + 1e-12, "grid threshold is never larger"


def test_frontier_decomposition():
    rng, p0, p1, D0, D1 = synth(seed=7)
    pools = pe.make_pools(p0.size, 16, rng)
    n = pools.size; keep = pools.ravel()
    cal = dict(y=D1[keep][: n // 2], p0=p0[keep][: n // 2], p1=p1[keep][: n // 2]); cal["pools"] = np.arange(cal["y"].size).reshape(-1, 16)
    test = dict(y=D1[keep][n // 2:], p0=p0[keep][n // 2:], p1=p1[keep][n // 2:]); test["pools"] = np.arange(test["y"].size).reshape(-1, 16)
    r = pe.compare_policies(cal, test, 0.06, 5, float(np.quantile(p0, 0.5)), rng, draws=5)
    assert abs(r["proc"] - (r["A"] + r["O"] + r["residual"])) < 1e-12, "Proposition 1"
    bs = pe.nested_pool_bootstrap(cal, test, 0.06, 5, float(np.quantile(p0, 0.5)), B=20, draws=3)
    assert bs["proc"][0] <= bs["proc"][1]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
