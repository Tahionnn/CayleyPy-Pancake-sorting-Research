# Base solution by Egor K. Experiment №1

## Public score
`157038`

## Source code
[cayley-pancake-base-variant](cayley-pancake-base-variant)

## Methology
Methology based on [Beam Search with CaleyPy](https://www.kaggle.com/code/fedimser/beam-search-with-cayleypy) notebook by [Dmytro Fedoriaka](https://www.kaggle.com/fedimser) and [Chervov, A. et al. "A Machine Learning Approach That Beats Large Rubik’s Cubes: The CayleyPy Project". arXiv:2502.13266 [cs.LG]. 2025.](https://arxiv.org/abs/2502.13266)

### 2. Neural Network Predictor
Designed a neural network model `StateScorer` that takes a permutation and outputs a score.
The model architecture:
  - Input: `n_pancakes`
  - The input is normalized to the range $[-1, 1]$ to help training.

```python
import torch
from torch import nn

class StateScorer(nn.Module):
    def __init__(self, n_pancakes: int, hidden_dim: int):
        super().__init__()
        self.n_pancakes = n_pancakes
        self.net = nn.Sequential(nn.Linear(n_pancakes, hidden_dim), nn.LeakyReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, x):
        return self.net(2*x.float() / (self.n_pancakes - 1) - 1).squeeze(-1)
```

### 3. Training the Predictor
- Constructed 13 separate Cayley graphs corresponding to each unique pancake size in the dataset: $n \in {5, 12, 15, 16, 20, 25, 30, 35, 40, 45, 50, 75, 100}$

Model Training Specifications:
- Seed: `42`
- Individual models: Trained `13 separate neural networks` - one for each graph size
- Training duration: `4 epochs` per model with batch size of 32
- Walk parameters: Width = $n \times 100$, Length = $\lfloor 1.6n \rfloor$
- The model is trained using `MSE` loss and the `AdamW` optimizer.

Computational Performance:

- Total training time: `~15 minutes` for all 13 models
- Data generation rate: `~650 sample's/second` consistently across different graph sizes

### 4. Beam Search with Heuristic
The beam search uses the trained neural network as a heuristic to prioritize promising states.

```python
heurestic_paths = []
for _, row in tqdm(test.iterrows(), total=len(test)):
    perms = np.array(row["permutation"].split(",")).astype(int)
    moves = pancake_sort_path(perms)
    heurestic_paths.append(".".join(moves))

heurestic_length = heurestic_paths[i].count(".") + 1

result = graphs[n].beam_search(
    start_state=perms, 
    beam_width=1000, 
    max_steps=heurestic_length, 
    predictor=models[n],
    return_path=True
)
```

Search Strategy:
- Adaptive beam width: 1000 states explored per step
- Early termination: Search limited to heuristic path length
- Fallback mechanism: Revert to classical pancake sort if neural search fails or underperforms
- Size-based optimization: Neural search applied only for $n < 20$, classical heuristic for larger instances

## System Specifications
The experiment was conducted on Kaggle using a CPU.

```bash
Architecture:          x86_64
CPU op-mode(s):        32-bit, 64-bit
Byte Order:            Little Endian
CPU(s):                4
On-line CPU(s) list:   0-3
Thread(s) per core:    2
Core(s) per socket:    2
Socket(s):             1
NUMA node(s):          1
Vendor ID:             GenuineIntel
CPU family:            6
Model:                 79
Model name:            Intel(R) Xeon(R) CPU @ 2.20GHz
Stepping:              0
CPU MHz:               2199.998
BogoMIPS:              4399.99
Hypervisor vendor:     KVM
Virtualization type:   full
L1d cache:             32K
L1i cache:             32K
L2 cache:              256K
L3 cache:              56320K
NUMA node0 CPU(s):     0-3
```