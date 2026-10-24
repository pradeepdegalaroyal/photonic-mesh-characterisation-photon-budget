"""Reproduce all results: ``python experiments/run_all.py`` (add ``--quick`` for a smoke run)."""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
DATA_TYPES = ("single", "two", "both")


def conditions(quick: bool) -> list[tuple[float, int]]:
    """(sigma [rad], total preparation events per phase setting)."""
    if quick:
        return [(0.06, 2000)]
    shots = [(0.06, b) for b in (300, 3000, 30000)]
    sig = [(s, 3000) for s in (0.03, 0.10)]
    return shots + sig


def run_job(args: tuple[int, int, bool]) -> list[dict]:
    """One (n_modes, seed): compile a Haar target once, then run every condition."""
    n, seed, quick = args
    torch.set_num_threads(2)
    from photomesh import (build_unitary, compile_ideal, fidelity, fit_errors, haar_unitary,
                           n_mzi, recompile, simulate_counts, single_photon_probs, two_photon_probs)
    rng = np.random.default_rng(1000 * n + seed)
    m, k_set = n_mzi(n), 8
    target = haar_unitary(n, rng)
    phases = compile_ideal(target, restarts=4 if quick else 6, steps=250 if quick else 400, seed=seed)[:3]
    rows = []
    for sigma, budget in conditions(quick):
        d1 = torch.as_tensor(rng.normal(0, sigma, m))
        d2 = torch.as_tensor(rng.normal(0, sigma, m))
        f_of = lambda p: float(fidelity(target, build_unitary(*p, d1, d2, n))[0])  # noqa: E731
        base = dict(n=n, seed=seed, sigma=sigma, budget=budget)
        rows.append(dict(base, method="naive", fidelity=f_of(phases), d_rmse=float(torch.sqrt(torch.mean(torch.cat([d1, d2]) ** 2)))))
        orc = recompile(target, phases, d1, d2, steps=100)
        rows.append(dict(base, method="oracle", fidelity=f_of(orc), d_rmse=0.0))
        settings = tuple(torch.rand(k_set, w, dtype=torch.float64) * 2 * np.pi for w in (m, m, n))
        v = build_unitary(*settings, d1, d2, n)
        p1, p2 = single_photon_probs(v), two_photon_probs(v)
        for dt in DATA_TYPES:
            n_cfg = {"single": n, "two": p2.shape[1], "both": n + p2.shape[1]}[dt]
            per = max(1, budget // n_cfg)
            c1 = simulate_counts(p1, per, rng) if dt != "two" else None
            c2 = simulate_counts(p2, per, rng) if dt != "single" else None
            f1, f2 = fit_errors(settings, c1, c2, n, steps=40 if quick else 150, lr=0.03)
            new = recompile(target, phases, f1, f2, steps=40 if quick else 100)
            err = float(torch.sqrt(torch.mean(torch.cat([f1 - d1, f2 - d2]) ** 2)))
            rows.append(dict(base, method=dt, fidelity=f_of(new), d_rmse=err))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    quick = ap.parse_args().quick
    t0 = time.time()
    modes, seeds = ([4], [0]) if quick else ([4, 6], [0, 1, 2])
    jobs = [(n, s, quick) for n in modes for s in seeds]
    with ProcessPoolExecutor(max_workers=1 if quick else 3) as ex:
        rows = [r for part in ex.map(run_job, jobs) for r in part]
    df = pd.DataFrame(rows)
    df["infidelity"] = 1 - df["fidelity"]
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    name = "raw_quick.csv" if quick else "raw.csv"
    df.to_csv(out / name, index=False)
    if not quick:
        summarise(df)
    (out / ("runtime_quick.json" if quick else "runtime.json")).write_text(
        json.dumps({"seconds": round(time.time() - t0, 1)}))
    print(f"done in {time.time() - t0:.0f}s")


def summarise(df: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    g = df.groupby(["n", "sigma", "budget", "method"])
    s = g.agg(inf_mean=("infidelity", "mean"), inf_std=("infidelity", "std"),
              fid_mean=("fidelity", "mean"), fid_std=("fidelity", "std"),
              rmse_mean=("d_rmse", "mean"), rmse_std=("d_rmse", "std"),
              n_seeds=("seed", "nunique")).reset_index()
    s.to_csv(ROOT / "results" / "summary.csv", index=False)
    fig_dir = ROOT / "figures"
    fig_dir.mkdir(exist_ok=True)
    styles = {"naive": ("k", "--"), "single": ("tab:blue", "-"), "two": ("tab:orange", "-"),
              "both": ("tab:green", "-"), "oracle": ("tab:gray", ":")}

    def panel(ax, n, xcol, sub, xlabel):
        for meth, (c, ls) in styles.items():
            d = sub[(sub.n == n) & (sub.method == meth)].sort_values(xcol)
            ax.errorbar(d[xcol], d.inf_mean, yerr=d.inf_std, color=c, ls=ls, marker="o", capsize=2, label=meth)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(xlabel); ax.set_ylabel("infidelity 1 - F"); ax.set_title(f"{n} modes")

    for tag, xcol, sub, xlabel in (
            ("shots", "budget", s[s.sigma == 0.06], "photon-preparation events per phase setting"),
            ("sigma", "sigma", s[s.budget == 3000], "beamsplitter angle error sigma (rad)")):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
        for ax, n in zip(axes, sorted(s.n.unique())):
            panel(ax, n, xcol, sub, xlabel)
        axes[0].legend(fontsize=8)
        fig.tight_layout(); fig.savefig(fig_dir / f"infidelity_vs_{tag}.png", dpi=200); plt.close(fig)
    fig, ax = plt.subplots(figsize=(5, 4))
    d = s[(s.sigma == 0.06) & (s.n == 6) & s.method.isin(["single", "two", "both"])]
    for meth in ("single", "two", "both"):
        e = d[d.method == meth].sort_values("budget")
        ax.errorbar(e.budget, e.rmse_mean, yerr=e.rmse_std, color=styles[meth][0], marker="o", capsize=2, label=meth)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("photon-preparation events per phase setting"); ax.set_ylabel("RMSE of fitted angle errors (rad)")
    ax.set_title("6 modes, sigma = 0.06 rad"); ax.legend()
    fig.tight_layout(); fig.savefig(fig_dir / "angle_error_rmse.png", dpi=200); plt.close(fig)


if __name__ == "__main__":
    main()
