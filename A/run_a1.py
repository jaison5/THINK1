from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / "results" / "a1"
MODELS = ["QLSTM", "QFWP", "FWPC", "LSTM3", "LSTM1"]
LRS = [0.005, 0.02, 0.05]
VARIANTS = ["aligned", "leaky"]
_CACHE = {}


def path(asset, variant, year, model, lr, seed):
    return OUT / f"{asset}_{variant}" / f"y{year}" / f"{model}_lr{lr:g}_s{seed}.json"


def run(job):
    import wf
    asset, variant, year, model, lr, seed = job
    out = path(*job)
    if out.exists():
        return
    key = (asset, variant)
    if key not in _CACHE:
        _CACHE[key] = wf.load(asset, variant)
    d = _CACHE[key]
    tr, va, te = wf.fold_idx(d, year)
    t0 = time.time()
    r = wf.train_eval(d, tr, va, te, model, lr, seed)
    r.update({"asset": asset, "variant": variant, "test_year": year, "model": model, "lr": lr, "seed": seed,
              "n_train": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(te)), "seconds": time.time() - t0})
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(r))
    print(f"{asset} {variant:7s} {year} {model:5s} lr={lr:<5g} s={seed:2d} val={r['val_rmse']:.6f} "
          f"ep={r['best_epoch']}/{r['epochs_run']} {r['seconds']:.0f}s", flush=True)


def choose_lr(assets, years):
    import numpy as np
    choice = {}
    for asset in assets:
        for variant in VARIANTS:
            for year in years:
                for model in MODELS:
                    scores = {}
                    for lr in LRS:
                        v = [json.loads(path(asset, variant, year, model, lr, s).read_text())["val_rmse"] for s in range(5)]
                        scores[lr] = float(np.mean(v))
                    choice[f"{asset}|{variant}|{year}|{model}"] = {"lr": min(scores, key=scores.get), "val_rmse": scores}
    (OUT / "lr_choice.json").write_text(json.dumps(choice, indent=1))
    return choice


if __name__ == "__main__":
    import wf
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--assets", default="EURUSD,GBPUSD,USDJPY")
    a = ap.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    assets, years = a.assets.split(","), wf.TEST_YEARS
    ctx = mp.get_context("spawn")
    stage1 = [(asset, v, y, m, lr, s) for m in MODELS for asset in assets for v in VARIANTS for y in years
              for lr in LRS for s in range(5)]
    print(f"階段 1：{len(stage1)} 次訓練", flush=True)
    with ctx.Pool(a.jobs) as pool:
        pool.map(run, stage1, chunksize=1)
    choice = choose_lr(assets, years)
    stage2 = [(asset, v, y, m, choice[f"{asset}|{v}|{y}|{m}"]["lr"], s) for m in MODELS for asset in assets
              for v in VARIANTS for y in years for s in range(5, 20)]
    print(f"階段 2：{len(stage2)} 次訓練", flush=True)
    with ctx.Pool(a.jobs) as pool:
        pool.map(run, stage2, chunksize=1)
    print("A1 完成", flush=True)
