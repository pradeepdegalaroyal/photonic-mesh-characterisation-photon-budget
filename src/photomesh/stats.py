"""Single- and two-photon output statistics of a linear-optical unitary."""
from __future__ import annotations

import itertools

import torch


def single_photon_probs(v: torch.Tensor) -> torch.Tensor:
    """P(out j | in i) as (B, n_in, n_out)."""
    return (v.abs() ** 2).transpose(-1, -2)


def n_two_outcomes(n: int) -> int:
    """Number of unordered output pairs (bunched included)."""
    return n * (n + 1) // 2


def two_photon_probs(v: torch.Tensor) -> torch.Tensor:
    """Photon-number-resolved output probabilities for inputs |1_a 1_b>, a<b.

    Returns (B, n(n-1)/2, n(n+1)/2); outcomes are unordered output pairs i<=j.
    """
    n = v.shape[-1]
    ins = list(itertools.combinations(range(n), 2))
    outs = list(itertools.combinations_with_replacement(range(n), 2))
    res = []
    for a, b in ins:
        row = []
        for i, j in outs:
            amp = v[:, i, a] * v[:, j, b] + v[:, i, b] * v[:, j, a]
            row.append(amp.abs() ** 2 / (2.0 if i == j else 1.0))
        res.append(torch.stack(row, dim=-1))
    return torch.stack(res, dim=1)
