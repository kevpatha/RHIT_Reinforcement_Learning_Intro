# Rosie's RL Journey: An Introduction to Reinforcement Learning

A hands-on lecture series that walks through the core ideas of reinforcement learning — from the simplest tabular methods to deep neural network agents — using a custom grid-world environment starring Rosie the Elephant.

---

## Overview

This project is structured as a progressive lecture, where each stage introduces a new concept that motivates the next. The environments grow in complexity alongside the algorithms, making each limitation concrete and the solution to it intuitive.

| Stage | Algorithm | Environment | Key Concept |
|---|---|---|---|
| 1 | Tabular Q-Learning | `RosieGridEnv` | Bellman updates, Q-tables |
| 2 | Discretized Q-Learning | `RosieContinuousEnv` | Continuous states, curse of dimensionality |
| 3 | Linear FA — SARSA(λ) | `RosieContinuousEnv` | Tile coding, Fourier, polynomial bases |
| 4 | Tabular Q-Learning | `RosieMomentumRandomHolesEnv` | Generalization failure |
| 5 | Linear SARSA(λ) | `RosieMomentumRandomHolesEnv` | Generalization with features |
| 6 | MLP-DQN | `RosieMomentumRandomHolesEnv` | Neural function approximation |

---

## Environments

### `RosieGridEnv` (`rosie_bonfire_env.py`)
A discrete 10×10 grid world. Rosie starts at a random cell and must reach the **Bonfire** (bottom-right corner) while avoiding five fixed **Holes**. This is the cleanest possible MDP — ideal for building intuition about Q-tables and the Bellman equation.

- **State:** `(x, y)` grid coordinates (integer tuple)
- **Actions:** UP, DOWN, LEFT, RIGHT
- **Rewards:** `+1` (goal), `-10` (hole), `-1` (step)
- **Visualization:** Jupyter inline animation or a Tkinter GUI

### `RosieContinuousEnv` (`rosie_continuous_env.py`)
Rosie now moves in continuous 2D space with velocity. The state is no longer a discrete index, which forces the question: *how do you store a Q-table when the state space is infinite?*

- **State:** `[x, y, vx, vy]`
- **Actions:** Directional thrust impulses
- **Key lesson:** Discretization works, but requires exponentially more bins as dimensions grow

### `RosieMomentumRandomHolesEnv` (`rosie_momentum_random_holes_env.py`)
The hardest environment. Rosie has momentum and a water level that depletes over time — she must reach the goal before it runs out. Hole positions are **randomized each episode**, and the state is augmented with the relative vector to the nearest hole. A tabular agent can only memorize specific hole layouts; it cannot generalize. This is the environment where tabular methods, linear function approximation, and DQN are tested head-to-head.

- **State:** `[x, y, vx, vy, water, dx_hole, dy_hole]` (7-dimensional)
- **Key lesson:** When the environment changes between episodes, you need a method that *generalizes*

---

## Algorithms

### 1. Tabular Q-Learning
The classic. A lookup table stores `Q(s, a)` for every visited state-action pair, updated via the Bellman equation:

```
Q(s,a) ← Q(s,a) + α [r + γ max_a' Q(s',a') - Q(s,a)]
```

Simple and guaranteed to converge on discrete MDPs, but completely unable to generalize across states it hasn't seen.

### 2. Discretized Q-Learning
Continuous states are binned into discrete buckets and handled with a standard Q-table. Demonstrates the **curse of dimensionality**: doubling resolution or adding a dimension multiplies table size exponentially. Three grid resolutions are compared to make this concrete.

### 3. Linear Function Approximation — SARSA(λ)
Instead of a table, Q-values are approximated as a linear combination of hand-crafted features:

```
Q(s,a) ≈ w · φ(s,a)
```

Three feature representations are compared:
- **Tile Coding** — sparse, overlapping grids that generalize locally
- **Polynomial Basis** — products of normalized state variables up to a fixed order
- **Fourier Basis** — cosine terms that capture global structure

Eligibility traces (λ) allow credit to propagate backwards through recent transitions, accelerating learning.

### 4. Tabular Q-Learning on Random Holes
The same tabular algorithm is now applied to the hard environment, where holes move every episode. This concretely demonstrates the generalization failure: the agent learns nothing transferable between episodes, since the state it memorized is never encountered again.

### 5. Linear SARSA(λ) on Random Holes
Linear function approximation is applied to the same hard environment. Because the agent works with a compact weight vector over features (including the hole distance vector), it *can* generalize — the same weights that describe "hole is close ahead" apply regardless of where on the map the hole happens to be.

### 6. Deep Q-Network (MLP-DQN)
A neural network replaces the weight vector. The network takes the raw 7-dimensional state as input and outputs Q-values for all actions. Key components:

- **Replay Buffer** — stores past transitions and samples random mini-batches to break temporal correlations
- **Target Network** — a periodically-frozen copy of the policy network used to compute stable Bellman targets
- **ε-greedy decay** — exploration probability anneals from 1.0 to 0.05 over training

The network architecture is a 3-layer MLP with 128 hidden units and ReLU activations.

---

## Project Structure

```
.
├── rosie_bonfire_env.py                  # Discrete grid world (Stage 1)
├── rosie_continuous_env.py               # Continuous 2D navigation (Stages 2–3)
├── rosie_momentum_random_holes_env.py    # Random holes + momentum + water (Stages 4–6)
├── dqn_model.py                          # DQN and ReplayBuffer definitions
├── main_lecture.py                       # All training functions + main() runner
├── MAIN_Lecture.ipynb                    # Notebook with visualizations and commentary
└── saved_results/                        # Auto-created; stores pickled Q-tables and model weights
```

> **Note:** `rosie_momentum_env.py` is included in the repository but is not used in the current lecture. It represents an intermediate environment (momentum + water level, but with fixed holes) that was cut from the progression.

---

## Getting Started

Training all stages from scratch is time-consuming. Pre-trained results are available to download so you can jump straight into the notebook:

**[Download saved_results folder](https://rosehulman-my.sharepoint.com/:f:/g/personal/haughn_rose-hulman_edu/IgAL9YOA_oWCT5TFSUVYkKj3AZ7u5lP6ovr5shDOys3e_dU?e=65VLmP)**

Download the folder and place it in the project root so the structure looks like this:

```
.
├── saved_results/
│   ├── q_table_grid.pkl
│   ├── continuous_q_tables.pkl
│   └── ...
├── main_lecture.py
├── MAIN_Lecture.ipynb
└── ...
```

The notebook and `main_lecture.py` will automatically detect and load these files, skipping any training steps that have already been completed.

---

## Dependencies

```
numpy
matplotlib
torch
IPython
```

Install with:
```bash
pip install numpy matplotlib torch ipython
```

Tkinter is used for the `run_gui()` visualizer in Stage 1 and is included with most standard Python distributions.

---

## Running the Project

**As a script** (trains all stages sequentially, caches results to `saved_results/`):
```bash
python main_lecture.py
```

Results are pickled and reloaded on subsequent runs, so individual stages can be re-run without retraining everything. On Windows, the script also prevents the system from sleeping during long training runs.

**As a notebook** — open `MAIN_Lecture.ipynb` in Jupyter. Each section corresponds to a stage in the lecture and includes inline visualizations of learned policies and training curves.

---

## Key Takeaways

The course builds a single argument across six stages:

1. **Tabular Q-Learning works perfectly** when the state space is small and discrete.
2. **Discretization extends it** to continuous spaces, but the table grows exponentially with dimension.
3. **Linear function approximation** breaks the curse by parameterizing Q with a compact weight vector — but requires hand-engineered features.
4. **Tabular methods break entirely** when the environment changes between episodes, because they memorize rather than generalize.
5. **Linear FA generalizes** because the same weights apply to any configuration, as long as the features capture what matters.
6. **Deep Q-Networks** learn those features automatically from data, removing the need for domain-specific feature engineering.

Each transition is motivated by a concrete failure of the previous approach, so by the time DQN appears, its architecture choices — replay buffer, target network — should each feel like obvious solutions to problems already encountered.
