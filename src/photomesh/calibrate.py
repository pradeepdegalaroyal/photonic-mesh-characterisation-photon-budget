"""Characterise beamsplitter errors from photon-count data, then recompile the mesh phases."""
from __future__ import annotations

import numpy as np
import torch

from .mesh import build_unitary, fidelity, n_mzi
from .stats import single_photon_probs, two_photon_probs

TAU = 6.283185307179586


def compile_ideal(target: torch.Tensor, restarts: int = 8, steps: int = 600, seed: int = 0):
    """Phases realising ``target`` on an ideal mesh (best of several Adam restarts)."""
    n, m = target.shape[-1], n_mzi(target.shape[-1])
    g = torch.Generator().manual_seed(seed)
    th = (torch.rand(restarts, m, generator=g, dtype=torch.float64) * TAU).requires_grad_()
    ph = (torch.rand(restarts, m, generator=g, dtype=torch.float64) * TAU).requires_grad_()
    out = (torch.rand(restarts, n, generator=g, dtype=torch.float64) * TAU).requires_grad_()
    z = torch.zeros(1, m, dtype=torch.float64)
    opt = torch.optim.Adam([th, ph, out], lr=0.05)
    for _ in range(steps):
        opt.zero_grad()
        (1 - fidelity(target, build_unitary(th, ph, out, z, z, n))).sum().backward()
        opt.step()
    with torch.no_grad():
        f = fidelity(target, build_unitary(th, ph, out, z, z, n))
        k = int(torch.argmax(f))
    return th[k].detach(), ph[k].detach(), out[k].detach(), float(f[k])


def simulate_counts(probs: torch.Tensor, shots_per_config: int, rng: np.random.Generator) -> np.ndarray:
    """Multinomial counts; probs (..., n_outcomes) -> same shape."""
    p = probs.detach().numpy()
    p = p / p.sum(-1, keepdims=True)
    flat = p.reshape(-1, p.shape[-1])
    c = np.stack([rng.multinomial(shots_per_config, row) for row in flat])
    return c.reshape(p.shape)


def fit_errors(settings, counts_1, counts_2, n: int, steps: int = 300, lr: float = 0.02):
    """Maximum-likelihood fit of per-beamsplitter angle errors.

    settings: (theta, phi, out) tensors with batch dim K. counts_1/counts_2 may be None.
    Returns fitted (d1, d2) of shape (M,).
    """
    th, ph, out = settings
    m = n_mzi(n)
    d1 = torch.zeros(m, dtype=torch.float64, requires_grad=True)
    d2 = torch.zeros(m, dtype=torch.float64, requires_grad=True)
    c1 = None if counts_1 is None else torch.as_tensor(counts_1, dtype=torch.float64)
    c2 = None if counts_2 is None else torch.as_tensor(counts_2, dtype=torch.float64)
    opt = torch.optim.Adam([d1, d2], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        v = build_unitary(th, ph, out, d1, d2, n)
        nll = 0.0
        if c1 is not None:
            nll = nll - (c1 * torch.log(single_photon_probs(v) + 1e-12)).sum()
        if c2 is not None:
            nll = nll - (c2 * torch.log(two_photon_probs(v) + 1e-12)).sum()
        nll.backward()
        opt.step()
    return d1.detach(), d2.detach()


def recompile(target, phases0, d1, d2, steps: int = 250, lr: float = 0.02):
    """Re-optimise phases on a model mesh with given errors, warm-started from ``phases0``."""
    n = target.shape[-1]
    th, ph, out = (p.clone().requires_grad_() for p in phases0)
    opt = torch.optim.Adam([th, ph, out], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        (1 - fidelity(target, build_unitary(th, ph, out, d1, d2, n))).sum().backward()
        opt.step()
    return th.detach(), ph.detach(), out.detach()
