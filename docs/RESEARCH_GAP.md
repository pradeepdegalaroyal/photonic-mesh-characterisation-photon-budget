# Research gap

## State of the art (from the supplied literature)
- Photonic quantum computing is moving to manufacturable, modular integrated platforms: Zhou et al., 2025, doi:10.1038/s41586-025-08820-7; Aghaee Rad et al., 2025, doi:10.1038/s41586-024-08406-9. The latter notes that complete integrated systems with all basic functionalities are still needed.
- Fusion-based quantum computation tolerates a finite per-fusion photon-loss probability (Bartolucci et al., 2023, doi:10.1038/s41467-023-36493-1), so every residual circuit error eats into a fixed loss/error budget.
- Large-scale integrated graph photonics and photonic chips for communication (Bao et al., 2023, doi:10.1038/s41566-023-01187-z; Luo et al., 2023, doi:10.1038/s41377-023-01173-8) rely on programmable interferometer meshes whose beamsplitters are never exactly 50:50.
- Single-photon platforms (Maring et al., 2024, doi:10.1038/s41566-024-01403-4; Couteau et al., 2023, doi:10.1038/s42254-023-00583-2) offer single- and two-photon (Hong-Ou-Mandel) data as the natural probes of a chip.

## Gap
Papers above show that scaling needs programmable meshes, but they do not tell a practitioner **which photon data to spend a finite measurement budget on** when characterising beamsplitter errors before recompiling the mesh. Classical (single-photon intensity) characterisation is standard; two-photon interference carries phase information that intensities lack, yet costs shots and, at equal preparation budget, is spread over fewer input configurations.

## Hypothesis (falsifiable)
At an equal photon-preparation budget per phase setting, maximum-likelihood characterisation from two-photon coincidence statistics yields a lower post-recompilation process infidelity than characterisation from single-photon intensities alone, for 4- and 6-mode rectangular meshes with Gaussian beamsplitter-angle errors (sigma 0.03-0.10 rad). It is falsified if the paired median infidelity of the two-photon strategy is not lower than single-photon in most paired cases.

**Metric:** process infidelity 1 - |Tr(U†V)|²/n² of the recompiled mesh on the true (erroneous) device, mean ± std over seeds; secondary metric: root-mean-square error (RMSE) of fitted angle errors.

**Scale:** exact simulation (statevector-free: permanent formulas for 2 photons), 4 and 6 modes, 3 seeds. No hardware claims.
