# Method

**Device model.** Rectangular mesh of n(n-1)/2 MZIs; each MZI = BS(pi/4+d2) · P(theta) · BS(pi/4+d1) · P(phi), plus output phases. d1, d2 ~ N(0, sigma²) are unknown splitting-angle errors; phases are assumed exactly set. No photon loss is modelled.

**Pipeline.**
1. Compile a Haar-random target U on the *ideal* mesh (Adam, best of several restarts) → **naive** baseline (what a standard decomposition-based compiler would apply).
2. Probe the chip at K = 8 random phase settings. Each setting spends a total of B photon-preparation events, split evenly over the input configurations of the chosen data type:
   - **single**: n single-photon inputs, intensity outputs (the standard classical characterisation);
   - **two**: n(n-1)/2 two-photon inputs, photon-number-resolved coincidences (HOM-type);
   - **both**: all of the above, same total B (ablation: mixing).
   Counts are multinomial (shot noise).
3. Fit all d1, d2 by maximum likelihood (Adam on the multinomial NLL).
4. Recompile phases on the fitted model (warm start from the naive phases, 100 Adam steps) and evaluate on the true device.

**Baselines.** Classical/standard: naive ideal compilation and single-photon characterisation. Upper bound: **oracle** recompilation with the true errors (same optimiser budget, so it is also limited by 100 steps). 

**Ablation:** data type (single / two / both) at fixed B. **Sweeps:** budget B ∈ {300, 3000, 30000} at sigma = 0.06; sigma ∈ {0.03, 0.06, 0.10} at B = 3000. **Seeds:** 3 per mode count (target, errors, settings, shot noise differ per seed). n ∈ {4, 6}.
