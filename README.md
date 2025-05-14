# QNSTR-Optimizer

An implementation code of [A Quasi-Newton Subspace Trust Region Algorithm for nonmonotone variational inequalities in adversarial learning over box constraints](https://arxiv.org/abs/2302.05935)

---

## Install

```bash
uv sync
```

## Usage

```python
from qnstr_optimizer.optimizer import QnstrOptimizer


def loss_fn(x, y):
    return x**2 - 5 * x * y - y**2

x = torch.tensor([1.0], requires_grad=False)
y = torch.tensor([1.0], requires_grad=False)
params = [x, y]
optimizer = QnstrOptimizer(
    params,
    loss_fn,
    **dict(
        zeta1=0.1,
        zeta2=0.4,
        beta1=0.5,
        beta2=5,
        eta=0.5,
        nu=200,
        tau=0.9,
        epsilon=1e-6,
        epsilon_criteria=1e-4,
        memory_size=10,
        bfgs_dir_count=3,
        max_step=100,
        mu_s=1e-2,
    ),
)
optimizer.step()
assert abs(x.item()) < 0.1 and abs(y.item()) < 0.1
```

