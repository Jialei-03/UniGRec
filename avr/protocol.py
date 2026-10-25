"""Anchor-Verified Re-ranking (AVR): scoring, weight fitting and feature-set selection.

    s(i) = gamma_1 lp_1 + gamma_2 lp_2 + lp_3 + lp_4 + (w_T <q, a_i> + w_S <z, u_i>) / tau_a

Two feature sets are compared: H = {w_T, w_S} and H+L = H + {gamma_1, gamma_2}. The set is
chosen by two-fold cross-validation over validation users (mean held-out NDCG@10 gain over
the beam order); its weights are then fitted on the whole validation split by coordinate
ascent on NDCG@10 and applied once to test.
"""
import numpy as np

HEAD_KEYS = ("tgt", "seq")
TAU_A = 0.07  # anchor temperature; set from the checkpoint by run_avr.py
GRID = dict(a1=[0.6, 0.8, 1.0, 1.2, 1.5], a2=[0.6, 0.8, 1.0, 1.2, 1.5],
            **{f"w_{h}": [0, .05, .1, .2, .3, .5, .8, 1.2] for h in HEAD_KEYS})


def score(d, p, tau_a=None):
    tau_a = TAU_A if tau_a is None else tau_a
    identity = (p.get("a1", 1.0) == 1.0 and p.get("a2", 1.0) == 1.0
                and all(p.get(f"w_{h}", 0.0) == 0 for h in HEAD_KEYS))
    if identity:
        return d["gen_order"]
    lp = d["lp"]
    s = p.get("a1", 1.0) * lp[..., 0] + p.get("a2", 1.0) * lp[..., 1] + lp[..., 2:].sum(-1)
    for h in HEAD_KEYS:
        s = s + p.get(f"w_{h}", 0.0) * d[h] / tau_a
    return s


def ranks(s, t):
    order = np.argsort(-s, axis=1, kind="stable")
    ts = np.take_along_axis(t, order, 1)
    return np.where(ts.any(1), ts.argmax(1) + 1, 0)


def metrics(s, t, topk=(5, 10, 20)):
    r = ranks(s, t)
    out = {}
    for k in topk:
        hit = (r > 0) & (r <= k)
        out[f"recall@{k}"] = float(hit.mean())
        out[f"ndcg@{k}"] = float(np.where(hit, 1 / np.log2(np.maximum(r, 1) + 1), 0).mean())
    return out


def subset(d, idx):
    n = d["gen_order"].shape[0]
    return {k: (v[idx] if isinstance(v, np.ndarray) and v.shape[:1] == (n,) else v) for k, v in d.items()}


def fit(d, keys, rounds=3):
    """Coordinate ascent on NDCG@10, starting from the beam order."""
    p = {k: (1.0 if k in ("a1", "a2") else 0.0) for k in keys}
    best = metrics(score(d, p), d["is_target"])["ndcg@10"]
    for _ in range(rounds):
        for k in keys:
            for v in GRID[k]:
                q = dict(p, **{k: v})
                m = metrics(score(d, q), d["is_target"])["ndcg@10"]
                if m > best + 1e-12:
                    best, p = m, q
    return p


def feature_sets():
    h = [f"w_{x}" for x in HEAD_KEYS]
    return {"H": h, "H+L": h + ["a1", "a2"]}


def select_on_validation(dv, seed=0):
    n = dv["gen_order"].shape[0]
    perm = np.random.default_rng(seed).permutation(n)
    folds = [perm[: n // 2], perm[n // 2:]]
    cv = {}
    for name, keys in feature_sets().items():
        gains = []
        for i in range(2):
            tr, te = subset(dv, folds[i]), subset(dv, folds[1 - i])
            p = fit(tr, keys)
            base = metrics(te["gen_order"], te["is_target"])["ndcg@10"]
            new = metrics(score(te, p), te["is_target"])["ndcg@10"]
            gains.append(new / base - 1 if base > 0 else new - base)
        cv[name] = float(np.mean(gains))
    return max(cv, key=cv.get), cv
