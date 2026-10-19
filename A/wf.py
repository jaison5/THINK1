from __future__ import annotations

import math
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "B"))

import numpy as np
import pandas as pd

START, END = "2003-12-01", "2024-12-31"
ASSETS = {"EURUSD": "ohlc_EURUSD_X.csv", "GBPUSD": "ohlc_GBPUSD_X.csv", "USDJPY": "ohlc_JPY_X.csv"}
TEST_YEARS = list(range(2015, 2025))
SEQ = 5


def load(asset: str, variant: str):
    df = pd.read_csv(HERE / "data" / ASSETS[asset], parse_dates=["date"]).set_index("date")
    df = df[["Open", "High", "Low", "Close"]].astype(float)
    df = df[(df > 0).all(axis=1)].dropna()
    df = df[~df.index.duplicated(keep="last")].sort_index().loc[START:END]
    if variant == "aligned":
        df = df.assign(High=df["High"].shift(1), Low=df["Low"].shift(1)).iloc[1:]
    o = df.values
    X, y, lo, hi, last, tgt, tdate, hl_mid = [], [], [], [], [], [], [], []
    for i in range(SEQ - 1, len(o) - 1):
        w = o[i - SEQ + 1:i + 1]
        mn, mx = w.min(), w.max()
        if mx - mn < 1e-12:
            continue
        X.append((w - mn) / (mx - mn)); y.append((o[i + 1, 3] - mn) / (mx - mn))
        lo.append(mn); hi.append(mx); last.append(o[i, 3]); tgt.append(o[i + 1, 3])
        tdate.append(df.index[i + 1]); hl_mid.append((o[i, 1] + o[i, 2]) / 2)
    return {"X": np.array(X), "y": np.array(y), "lo": np.array(lo), "hi": np.array(hi), "last": np.array(last),
            "tgt": np.array(tgt), "tyear": np.array([d.year for d in tdate]), "tdate": np.array(tdate),
            "hl_mid": np.array(hl_mid), "rows": len(df)}


def fold_idx(d, test_year: int):
    ty = d["tyear"]
    return np.where(ty <= test_year - 2)[0], np.where(ty == test_year - 1)[0], np.where(ty == test_year)[0]


def to_price(d, idx, p):
    return p * (d["hi"][idx] - d["lo"][idx]) + d["lo"][idx]


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def da_paper(yhat, y):
    dy, dp = np.diff(y), np.diff(yhat)
    m = dy != 0
    return float(100 * np.mean(np.sign(dp[m]) == np.sign(dy[m])))


def da_known(yhat, y, last):
    d = y - last
    m = d != 0
    return float(100 * np.mean(np.sign(yhat[m] - last[m]) == np.sign(d[m])))


def dm_test(e_model, e_base):
    from scipy import stats
    d = e_model ** 2 - e_base ** 2
    T = len(d)
    s = d.var(ddof=0)
    if s == 0:
        return 0.0, 1.0
    dm = d.mean() / math.sqrt(s / T)
    dm *= math.sqrt((T - 1) / T)
    return float(dm), float(2 * stats.t.sf(abs(dm), df=T - 1))


def pt_test(pred_dir, true_dir):
    from scipy import stats
    m = (pred_dir != 0) & (true_dir != 0)
    x, y = pred_dir[m] > 0, true_dir[m] > 0
    n = len(x)
    p_hat = np.mean(x == y)
    py, px = y.mean(), x.mean()
    p_star = py * px + (1 - py) * (1 - px)
    v_hat = p_star * (1 - p_star) / n
    v_star = ((2 * py - 1) ** 2 * px * (1 - px) / n + (2 * px - 1) ** 2 * py * (1 - py) / n
              + 4 * py * px * (1 - py) * (1 - px) / n ** 2)
    if v_hat - v_star <= 0:
        return 0.0, 0.5
    stat = (p_hat - p_star) / math.sqrt(v_hat - v_star)
    return float(stat), float(stats.norm.sf(stat))


def make_model(name: str):
    import fastsim as F
    return {"QLSTM": lambda: F.QLSTM(), "QFWP": lambda: F.QFWP(), "FWPC": lambda: F.FWPClassical(),
            "LSTM1": lambda: F.LSTM(H=1), "LSTM3": lambda: F.LSTM(H=3)}[name]()


def train_eval(d, tr, va, te, name, lr, seed, batch=16, max_epochs=10, patience=3):
    import torch
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    model = make_model(name)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xtr, ytr = torch.tensor(d["X"][tr]), torch.tensor(d["y"][tr])
    Xva, Xte = torch.tensor(d["X"][va]), torch.tensor(d["X"][te])
    rng = np.random.default_rng(seed + 1000)

    def predict(Xs, idx):
        with torch.no_grad():
            p = torch.cat([model(Xs[s:s + 512])[:, 0] for s in range(0, len(Xs), 512)]).numpy()
        return to_price(d, idx, p)

    best, best_state, best_ep, wait, hist = np.inf, None, 0, 0, []
    for ep in range(1, max_epochs + 1):
        perm = rng.permutation(len(tr))
        for s in range(0, len(tr), batch):
            j = torch.as_tensor(perm[s:s + batch])
            opt.zero_grad()
            loss = ((model(Xtr[j])[:, 0] - ytr[j]) ** 2).mean()
            if not torch.isfinite(loss):
                break
            loss.backward()
            opt.step()
        v = rmse(predict(Xva, va), d["tgt"][va])
        hist.append(v)
        if v < best - 1e-12:
            best, best_ep, wait = v, ep, 0
            best_state = {k: t.clone() for k, t in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return {"val_rmse": float(best), "best_epoch": best_ep, "epochs_run": len(hist), "val_hist": hist,
            "pred_test": predict(Xte, te).tolist()}


def linear_fit(d, tr, va, te):
    A = np.c_[d["X"][tr].reshape(len(tr), -1), np.ones(len(tr))]
    w, *_ = np.linalg.lstsq(A, d["y"][tr], rcond=None)
    f = lambda idx: to_price(d, idx, np.c_[d["X"][idx].reshape(len(idx), -1), np.ones(len(idx))] @ w)
    return {"val_rmse": rmse(f(va), d["tgt"][va]), "pred_test": f(te).tolist()}
