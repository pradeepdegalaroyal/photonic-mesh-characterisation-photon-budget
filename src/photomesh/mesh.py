"""Rectangular Mach-Zehnder interferometer (MZI) mesh with beamsplitter-angle errors."""
from __future__ import annotations

import math

import numpy as np
import torch


def n_mzi(n: int) -> int:
    """Number of MZIs in a universal rectangular mesh on ``n`` modes."""
    return n * (n - 1) // 2


def mesh_layout(n: int) -> list[int]:
    """Upper mode index ``m`` of every MZI (acting on modes m, m+1), in application order."""
    out: list[int] = []
    for layer in range(n):
        out.extend(range(layer % 2, n - 1, 2))
    return out


def _bs(a: torch.Tensor) -> torch.Tensor:
    """Beamsplitter matrices (...,2,2) with splitting angle ``a`` (pi/4 is 50:50)."""
    c, s = torch.cos(a), torch.sin(a)
    row0 = torch.stack([c + 0j, 1j * s], dim=-1)
    row1 = torch.stack([1j * s, c + 0j], dim=-1)
    return torch.stack([row0, row1], dim=-2)


def _ph(x: torch.Tensor) -> torch.Tensor:
    """diag(exp(ix), 1) matrices (...,2,2)."""
    z = torch.zeros_like(x) + 0j
    row0 = torch.stack([torch.exp(1j * x), z], dim=-1)
    row1 = torch.stack([z, torch.ones_like(x) + 0j], dim=-1)
    return torch.stack([row0, row1], dim=-2)


def build_unitary(theta, phi, out, d1, d2, n: int) -> torch.Tensor:
    """Unitary of a mesh. theta/phi/d1/d2: (B, M) or (M,), out: (B, n) or (n,).

    d1, d2 are splitting-angle errors (rad) of the two beamsplitters of each MZI.
    """
    theta, phi, d1, d2 = (torch.atleast_2d(torch.as_tensor(t)) for t in (theta, phi, d1, d2))
    out = torch.atleast_2d(torch.as_tensor(out))
    b = max(theta.shape[0], phi.shape[0], d1.shape[0], d2.shape[0], out.shape[0])
    rows = [torch.eye(n, dtype=torch.complex128).expand(b, n, n)[:, r, :] for r in range(n)]
    for k, m in enumerate(mesh_layout(n)):
        t = _bs(math.pi / 4 + d2[:, k]) @ _ph(theta[:, k]) @ _bs(math.pi / 4 + d1[:, k]) @ _ph(phi[:, k])
        r0, r1 = rows[m], rows[m + 1]
        rows[m] = t[:, 0, 0:1] * r0 + t[:, 0, 1:2] * r1
        rows[m + 1] = t[:, 1, 0:1] * r0 + t[:, 1, 1:2] * r1
    u = torch.stack(rows, dim=1)
    return torch.exp(1j * out)[:, :, None] * u


def fidelity(target: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Process fidelity |Tr(U^dag V)|^2 / n^2, batched over V."""
    n = target.shape[-1]
    tr = torch.einsum("ij,bij->b", target.conj(), v)
    return (tr.abs() ** 2) / n**2


def haar_unitary(n: int, rng: np.random.Generator) -> torch.Tensor:
    """Haar-random n x n unitary."""
    z = (rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))) / math.sqrt(2)
    q, r = np.linalg.qr(z)
    q = q * (np.diag(r) / np.abs(np.diag(r)))
    return torch.as_tensor(q, dtype=torch.complex128)
