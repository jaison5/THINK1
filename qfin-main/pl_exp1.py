from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time

import numpy as np

from common import RES, da_known, da_paper, dump
from exp1a import prep

CFG = {}


def _init(cfg):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import torch
    torch.set_num_threads(1)
    CFG.update(cfg)


def run(job):
    import torch
    import pl_models as P
    P.configure(CFG["device"], CFG["diff"])
    name, batch, seed = job
    out = RES / f"exp1a_pl_{CFG['opt']}{CFG['lr']:g}" / f"{name}_b{batch}_s{seed}.json"
    if out.exists():
        return
    df, X, y, lo, hi, last, tgt, end, n_tr = prep("paper")
    torch.manual_seed(seed)
    model = P.MODELS[name]()
    params = list(model.parameters())
    opt = torch.optim.Adam(params, lr=CFG["lr"]) if CFG["opt"] == "adam" else torch.optim.RMSprop(params, lr=CFG["lr"])
    Xtr, ytr = torch.tensor(X[:n_tr]), torch.tensor(y[:n_tr])
    rng = np.random.default_rng(seed + 1000)
    tf = tb = 0.0
    t0 = time.perf_counter()
    for _ in range(2):
        idx = rng.permutation(n_tr)
        for s in range(0, n_tr, batch):
            j = torch.as_tensor(idx[s:s + batch])
            opt.zero_grad()
            a = time.perf_counter()
            loss = ((model(Xtr[j])[:, 0] - ytr[j]) ** 2).mean()
            b = time.perf_counter()
            loss.backward()
            c = time.perf_counter()
            opt.step()
            tf += b - a
            tb += c - b
    full = time.perf_counter() - t0
    t1 = time.perf_counter()
    with torch.no_grad():
        Xte = torch.tensor(X[n_tr:])
        p = torch.cat([model(Xte[s:s + 256])[:, 0] for s in range(0, len(Xte), 256)]).numpy()
    t_inf = time.perf_counter() - t1
    yhat = p * (hi[n_tr:] - lo[n_tr:]) + lo[n_tr:]
    yt = tgt[n_tr:]
    res = {"model": name, "batch": batch, "seed": seed, "variant": "paper", "backend": f"{CFG['device']}/{CFG['diff']}",
           "n_params": int(sum(q.numel() for q in params)),
           "rmse": float(np.sqrt(np.mean((yhat - yt) ** 2))),
           "da_paper": da_paper(yhat, yt), "da_known": da_known(yhat, yt, last[n_tr:]),
           "train_forward_s": tf, "backward_s": tb, "full_train_s": full, "inference_s": t_inf,
           "steps_per_epoch": int(np.ceil(n_tr / batch))}
    dump(res, out)
    print(f"{name:5s} b={batch:2d} s={seed} rmse={res['rmse']:.5f} DA={res['da_paper']:.2f} train={full:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--models", default="QLSTM,QFWP,LSTM")
    ap.add_argument("--batches", default="1,4,8,16,32,64")
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--opt", default="adam", choices=["adam", "rmsprop"])
    ap.add_argument("--device", default="lightning.qubit")
    ap.add_argument("--diff", default="adjoint")
    a = ap.parse_args()
    cfg = {"lr": a.lr, "opt": a.opt, "device": a.device, "diff": a.diff}
    d = RES / f"exp1a_pl_{a.opt}{a.lr:g}"
    d.mkdir(parents=True, exist_ok=True)
    df, X, y, lo, hi, last, tgt, end, n_tr = prep("paper")
    dump({"rows": len(df), "windows": len(X), "train": n_tr, "test": len(X) - n_tr,
          "train_end": str(df.index[end[n_tr - 1]].date()), "test_start": str(df.index[end[n_tr]].date()),
          "rw_rmse": float(np.sqrt(np.mean((last[n_tr:] - tgt[n_tr:]) ** 2))),
          "rw_da_paper": da_paper(last[n_tr:], tgt[n_tr:]), **cfg}, d / "_data.json")
    print(f"訓練樣本 {n_tr}（論文 4374）、測試 {len(X) - n_tr}；結果存到 {d}", flush=True)
    jobs = [(m, int(b), s) for m in a.models.split(",") for s in range(a.seed0, a.seed0 + a.seeds)
            for b in a.batches.split(",")]
    ctx = mp.get_context("fork" if os.name != "nt" else "spawn")
    with ctx.Pool(a.jobs, initializer=_init, initargs=(cfg,)) as pool:
        pool.map(run, jobs, chunksize=1)
