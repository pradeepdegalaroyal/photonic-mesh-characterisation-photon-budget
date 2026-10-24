import numpy as np
import torch

from photomesh import (build_unitary, fidelity, haar_unitary, n_mzi, single_photon_probs,
                       two_photon_probs)


def _rand(n, rng, sigma=0.0):
    m = n_mzi(n)
    t = lambda w: torch.as_tensor(rng.uniform(0, 6.28, (1, w)))  # noqa: E731
    d = lambda: torch.as_tensor(rng.normal(0, sigma, (1, m)))  # noqa: E731
    return build_unitary(t(m), t(m), t(n), d(), d(), n)


def test_mesh_is_unitary_even_with_errors():
    v = _rand(5, np.random.default_rng(0), sigma=0.1)[0]
    assert torch.allclose(v.conj().T @ v, torch.eye(5, dtype=torch.complex128), atol=1e-10)


def test_probabilities_normalised():
    v = _rand(4, np.random.default_rng(1), sigma=0.05)
    assert torch.allclose(single_photon_probs(v).sum(-1), torch.ones(1, 4, dtype=torch.float64))
    assert torch.allclose(two_photon_probs(v).sum(-1), torch.ones(1, 6, dtype=torch.float64))


def test_hom_dip_on_balanced_beamsplitter():
    # one ideal MZI with theta=pi/2 acts as a 50:50 splitter; two photons must bunch
    z = torch.zeros(1, 1, dtype=torch.float64)
    v = build_unitary(torch.full((1, 1), 1.5707963267948966, dtype=torch.float64), z, torch.zeros(1, 2, dtype=torch.float64), z, z, 2)
    p = two_photon_probs(v)[0, 0]  # outcomes (0,0),(0,1),(1,1)
    assert float(p[1]) < 1e-9


def test_fidelity_of_identical_unitaries_is_one():
    u = haar_unitary(4, np.random.default_rng(2))
    assert float(fidelity(u, u[None])[0]) > 1 - 1e-12
