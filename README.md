# State-Constrained HJ via PINNs

This project trains a PINN to approximate the viscosity solution of the static Hamilton–Jacobi equation
with state-constrained boundary conditions by solving the penalized problem:
$$
u(x) + H(x, \nabla u(x)) = d(x, \Omega) / \epsilon \quad \text{in} \mathbb{R}^n
$$
and then decreasing $\epsilon -> 0$.

First experiment:
- Domain $\Omega = [-k, k]$
- Hamiltonian $H(x, p) = |p - 1| - 1$ for $p \in [0, 2]$, extended smoothly as a convex quadratic outside.
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

## Run (1D experiment)

```
uv run python experiments/state_constrained_1d.py --k 2 --eps 0.5 0.2 0.1 0.05 --steps 4000 --width 128 --layers 3 --batch 2048
```

Arguments:
- --k: half-width of domain [-k, k]
- --eps: list of epsilons for outer loop
- --steps: training steps per epsilon (inner loop)
- --batch: collocation batch size
- --margin: sample outside Omega up to this margin (default: 0.5*k)
- --lr: learning rate (default: 1e-3)
- --device: cpu or cuda (default: auto)
- --plot: set to plot predictions after each epsilon

The script prints the MSE vs the exact solution u(x) = exp(x - k) on a fine grid after each epsilon stage.

## Notes

- The code is structured to scale to higher dimensions:
  - `distance_to_box` handles general boxes in R^n
  - PINN MLP supports arbitrary input dim
  - The Hamiltonian interface is modular
- For n > 1 you can define your own H(x, p) in `pinn_hj/hamiltonians.py`, and keep the training code unchanged.

## Mathematical Notes

- Du denotes the spatial gradient of u and d(x, Omega) is the distance from x to the domain Omega.
- The state-constrained boundary condition is enforced in the epsilon -> 0 limit via the penalization term d(x, Omega) / epsilon.
