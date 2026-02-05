# State-Constrained HJ via PINNs

This project trains a PINN to approximate the viscosity solution of the static Hamilton–Jacobi equation
with state-constrained boundary conditions by solving the penalized problem:

$$u(x) + H(x, \nabla u(x)) = d(x, \Omega) / \epsilon \quad \text{in}\quad \mathbb{R}^n$$

and then decreasing $\epsilon -> 0$.

First experiment:
- Domain $\Omega = [-k, k]$
- Hamiltonian $H(x, p) = |p - 1| - 1$ inside $[0,2]$, with a quadratic tail outside (C^1, convex, coercive)
- The viscosity solution inside $\Omega$: $u(x) = \exp(x - k)$

## Setup

Install uv (choose one):
```
brew install uv
```
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```
uv sync
```

## Run (Unified Experiment Script)

All experiments are run via a single script: `scripts/train.py`.

Example configs live in `configs/`:

- `configs/state_constrained_1d.yaml`
- `configs/state_constrained_cylinder.yaml`

Run with a YAML config:

```
uv run python scripts/train.py --config configs/state_constrained_1d.yaml
```

Override any value from the config on the CLI:

```
uv run python scripts/train.py --config configs/state_constrained_1d.yaml --steps 8000 --lr 5e-4
```

You can also run directly without a config:

```
uv run python scripts/train.py --n 1 --domain box --k 2 --eps 0.5 0.2 0.1 0.05 --steps 4000 --width 128 --layers 3 --batch 2048
```

and for the cylindrical domain:

```
uv run python scripts/train.py --n 3 --domain cylinder --k 2 --eps 0.5 0.2 0.1 0.05 --steps 4000 --width 128 --layers 3 --batch 2048
```

Arguments (selected):
- --n: dimension
- --domain: box | cylinder | auto
- --eval: exact | residual | auto
- --k: domain scale
- --eps: list of epsilons for outer loop
- --steps: training steps per epsilon (inner loop)
- --batch: collocation batch size
- --margin: sampling margin (default 0.5*k for box, 0 for cylinder)
- --lr: learning rate (default: 1e-3)
- --device: cpu or cuda (default: auto)
- Plots and checkpoints are always saved during training.

For 1D, the script prints the MSE vs the exact solution u(x) = exp(x - k). For higher dimensions, it reports residual MSE on an evaluation set and can also plot slice visualizations.

## Config Structure

Configs are grouped into sections. All keys are flattened internally, and top-level keys (if provided)
override grouped values.

- `experiment`: `seed`, `device`
- `setting`: `n`, `domain`, `k`, `margin`, `alpha` (domain controls the *training* sampling domain)
- `training`: `eps`, `steps`, `batch`, `lr`
- `training`: `eps`, `steps`, `batch`, `lr`, `sample_every`, `max_grad_norm`
- `model`: `width`, `layers`, `act`
- `evaluation`: `eval`, `eval_points`, `log_every`

## Outputs

Training always saves outputs under:
- `results/{YYYYMMDD_HHMMSS}_n{n}_k{k}/plots` (plots for each epsilon; 2D/3D use heatmaps/slices)
- `results/{YYYYMMDD_HHMMSS}_n{n}_k{k}/checkpoints` (model checkpoints)
- `results/{YYYYMMDD_HHMMSS}_n{n}_k{k}/config` (config snapshots used for the run)
- `results/{YYYYMMDD_HHMMSS}_n{n}_k{k}/logs` (training_log.jsonl, epsilon_metrics.jsonl)

Notes:
- For n>=2, residuals use the cylindrical distance function and evaluation sampling is on the target cylinder domain.
- For n=1, residuals and evaluation sampling use the box domain.

## Notes

- The code is structured to scale to higher dimensions:
  - `distance_to_box` handles general boxes in R^n
  - PINN MLP supports arbitrary input dim
  - The Hamiltonian interface is modular
- For n > 1 you can define your own H(x, p) in `pinn_hj/hamiltonians.py`, and keep the training code unchanged.

## Mathematical Notes

- Du denotes the spatial gradient of u and d(x, Omega) is the distance from x to the domain Omega.
- The state-constrained boundary condition is enforced in the epsilon -> 0 limit via the penalization term d(x, Omega) / epsilon.
