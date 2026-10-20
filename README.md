# SmartSynch — Physics-Informed Dual Physical-Reservoir Synchronization

This repository is the paper-aligned implementation of:

**Learning When to Listen: Physics-Informed Synchronization in Molecular Channels**

The code follows the final manuscript architecture rather than the earlier generic/random-ESN prototype.

## Paper-to-code mapping

The default implementation mirrors the final manuscript:

- Free-diffusion molecular channel and Poisson observations: Eqs. (1)–(4)
- Candidate timing set: Eq. (5)
- Physical receptor delay-line state: Eq. (7)
- Linearized physical-reservoir / ESN representation: Eqs. (8)–(11)
- RC-PHY and RC-AGENT integration: Eqs. (12)–(13)
- Ridge-regression readout training: Eq. (14)
- Final paper parameters:
  - `T = 4 s`
  - `t_s = 0.1 s`
  - `N = 40`
  - `delta_step = 0.1 s`
  - `J = 40`
  - `M = 8000`
  - `D = 3.3 µm²/s`
  - `r_Rx = 4 µm`
  - `d = 11 µm`
  - `d_RC = 3 µm`
  - `k_on = 6.022e8 M^-1 s^-1`
  - `k_off = 1 s^-1`
  - `rho(W) = 0.89`
  - `beta = 3e-4`

## Important architecture change from the earlier Colab

The old notebook used two conventional random tanh ESNs with 80–160 units and trained RC-AGENT using an exact finite-action expected-reward gradient.

The final paper instead specifies:

1. **Two identical physical reservoirs** whose fixed dynamics follow the diffusion/receptor model.
2. **N = 40** virtual nodes for both reservoirs.
3. A linearized fixed recurrent matrix with the delay-line structure in Eq. (10).
4. **RC-PHY** learns the likelihood surrogate with a ridge-trained readout.
5. **RC-AGENT** receives the RC-PHY likelihood/reward trajectory and learns the timing estimate with a second ridge-trained readout, as stated by Eqs. (13)–(14).


## Repository layout

```text
smartsynch_dual_rc/
├── README.md
├── pyproject.toml
├── requirements.txt
├── ARCHITECTURE.md
├── smartsynch/
│   ├── __init__.py
│   ├── config.py
│   ├── physics.py
│   ├── reservoir.py
│   ├── models.py
│   ├── data.py
│   ├── evaluation.py
│   └── plotting.py
├── scripts/
│   └── run_paper_pipeline.py
└── notebooks/
    └── SmartSynch_Paper_Aligned_Colab.ipynb
```

## Quick start

```bash
pip install -e .
python scripts/run_paper_pipeline.py
```

The Colab notebook is self-contained and follows the same sequence as the package implementation.
