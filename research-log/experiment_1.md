# Base solution by Egor K. 

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

# Base solution by Kirill L. 

## Public score
`154406` 

## Source code
[cayley-pancake-base](cayley-pancake-base)

## Methology
Methology based on [Beam Search with CaleyPy](https://www.kaggle.com/code/fedimser/beam-search-with-cayleypy) notebook by [Dmytro Fedoriaka](https://www.kaggle.com/fedimser) and [Chervov, A. et al. "A Machine Learning Approach That Beats Large Rubik’s Cubes: The CayleyPy Project". arXiv:2502.13266 [cs.LG]. 2025.](https://arxiv.org/abs/2502.13266)

### 2. Neural Network Predictor
Take neural network from [Beam Search with CaleyPy](https://www.kaggle.com/code/fedimser/beam-search-with-cayleypy) 
```python
class Net(torch.nn.Module):
    def __init__(self, input_size, num_classes, hidden_dims):
        super().__init__()
        self.num_classes = num_classes
        input_size = input_size * self.num_classes
        layers = []
        for hidden_dim in hidden_dims:
            layers.append(torch.nn.Linear(input_size, hidden_dim))
            layers.append(torch.nn.GELU())
            input_size = hidden_dim
        layers.append(torch.nn.Linear(input_size, 1))
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        x = torch.nn.functional.one_hot(x.long(), num_classes=self.num_classes).float().flatten(start_dim=-2)
        return self.layers(x.float()).squeeze(-1)
```

### 3. Training the Predictor
- Constructed 13 separate Cayley graphs corresponding to each unique pancake size in the dataset: $n \in {5, 12, 15, 16, 20, 25, 30, 35, 40, 45, 50, 75, 100}$

Model Training Specifications:
- Seed: `42`
- Individual models: Trained `13 separate neural networks` - one for each graph size
- Training duration: `20 epochs` per model with batch size of 512
- Walk parameters: Width = $n \times 2000$, Length = $\min(2*n, 100)$
- The model is trained using `MSE` loss and the `AdamW` optimizer with `learning rate 0.001`

Computational Performance:

- Total training time: `3 minutes 51 seconds` for all 13 models

### 4. Beam Search with Heuristic
[The beam search uses the trained neural network as a heuristic to prioritize promising states](#4-beam-search-with-heuristic)
Fallback mechanism: Revert to classical pancake sort if neural search fails or underperforms

## System Specifications
The experiment was conducted on a local Linux machine with GPU acceleration.

```bash
Architecture:          x86_64
CPU(s):                12 (6 cores, 12 threads)
Vendor:                AMD
Model:                 AMD Ryzen 5 5600 6-Core Processor
Max Frequency:         4.47 GHz
L1 Cache:              384 KiB (192+192)
L2 Cache:              3 MiB
L3 Cache:              32 MiB

GPU:                   NVIDIA GeForce RTX 3060 Ti
VRAM:                  8 GB GDDR6
Driver Version:        550.163.01
CUDA Version:          12.4

OS:                    Linux
Virtualization:        AMD-V enabled
Memory:                8 GB GPU RAM + System RAM
```