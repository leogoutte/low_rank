# Low-Rank Approximation

This repository contains the source code and notebooks for the low-rank approximation to the Lindlad master equation. This code was used to produce the results in the paper Leo Goutte and Vincenzo Savona, "Low-rank optimal control of quantum devices", http://arxiv.org/abs/2508.18114.

The Low-Rank Approximation (LRA) consists of truncating the full $N \times N$ density matrix $\hat{\rho} = {\mathbf{m}} {\mathbf{m}}^{\dagger}$ to a fixed rank $M \ll N$. The Lindblad equation
```math
\dot{\hat{\rho}} = -i \left[\hat{H}, \hat{\rho} \right] + \sum_k \hat{L_k} \hat{\rho} \hat{L_k}^\dagger - \frac{\{\hat{L_k}^\dagger \hat{L_k}, \hat{\rho}\}}{2}
```
can be re-written as an equation of motion for $\mathbf{m}$,
```math
\dot{\mathbf{m}} = - i \hat{H} \mathbf{m} + \frac{1}{2}\sum_k \hat{L_k} \mathbf{m} (\mathbf{m}^{-1} \hat{L_k} \mathbf{m})^\dagger - \hat{L_k}^\dagger \hat{L_k} \mathbf{m}.
```

The notebook `low_rank_readout.ipynb` contains a minimal working example comparing the full, rotating-wave approximation (RWA), and LRA evolutions in the context of transmon readout. The relevant source code for the LRA is found in the `src/low_rank.jl` file. 