# Neuroevolution Snake — Adaptive Genetic Algorithm for Neural Network Weight Optimization

A complete implementation of a **Neural Network** trained via an **Adaptive Genetic Algorithm** to autonomously play the Snake game. No backpropagation is used — the network weights are evolved entirely through evolutionary operators (selection, crossover, mutation). The project includes a togglable comparison between **Adaptive GA** and **Standard GA** to demonstrate how dynamic parameter control improves convergence.

---

## Table of Contents

1. [Soft Computing Concepts](#soft-computing-concepts)
2. [System Architecture](#system-architecture)
3. [File-by-File Breakdown](#file-by-file-breakdown)
4. [Getting Started](#getting-started)
5. [Interactive Controls](#interactive-controls)
6. [Experiment Runner](#experiment-runner)

---

## Soft Computing Concepts

This project is built entirely on **Soft Computing** principles — techniques that tolerate imprecision, uncertainty, and approximation to achieve tractable, robust solutions for problems that are too complex for exact algorithms.

### 1. Neural Networks (Connectionist Model)

A **Feed-Forward Neural Network (FFNN)** acts as the "brain" of each snake agent. It is a function approximator that maps a 24-dimensional sensory input to a 4-dimensional action output.

- **Architecture:** `24 → 16 → 8 → 4` (input → hidden1 → hidden2 → output)
- **Hidden Activation — Hyperbolic Tangent (`tanh`):** Maps values to the range `[-1, +1]`. This is critical for neuroevolution because it provides both positive and negative signal flow, allowing evolved weights to express inhibitory connections — something `ReLU` cannot do without bias tuning.
- **Output Activation — Softmax:** Converts the 4 raw output logits into a probability distribution over the four movement directions (Up, Right, Down, Left). The agent picks the direction with the highest probability (`argmax`).
- **No Backpropagation:** The weights are never updated via gradient descent. Instead, the entire weight vector is treated as a "chromosome" and optimized by the Genetic Algorithm.

### 2. Evolutionary Computation (Genetic Algorithm)

A **Genetic Algorithm (GA)** is a metaheuristic search algorithm inspired by Darwinian natural selection. It maintains a population of candidate solutions (neural networks) and iteratively evolves them using biologically inspired operators.

#### 2a. Representation (Chromosome Encoding)

Each neural network's weights and biases are **flattened** into a single 1D vector of real numbers. This flat vector is the "chromosome" that the GA operates on. For the `[24, 16, 8, 4]` architecture, each chromosome contains:
- Weights: `(24×16) + (16×8) + (8×4) = 384 + 128 + 32 = 544` values
- Biases: `16 + 8 + 4 = 28` values
- **Total: 572 genes per chromosome**

#### 2b. Tournament Selection

Instead of roulette-wheel selection (which is biased toward high-fitness outliers), **Tournament Selection** is used:
1. Randomly sample `K=5` individuals from the population.
2. The individual with the highest fitness in that sample becomes a parent.
3. Repeat to select the second parent.

This provides selection pressure while maintaining diversity — weaker individuals still have a chance if they land in a weak tournament.

#### 2c. Layer-Wise Crossover

Standard uniform crossover (randomly swapping individual weight values between parents) breaks **co-adapted features** — groups of weights within a layer that work together to detect a specific pattern. Instead, **layer-wise crossover** is used:
- For each layer in the network, the child inherits the entire weight matrix and bias vector from either Parent A or Parent B (50/50 coin flip per layer).
- This preserves the internal structure of each layer while still recombining information from two parents.

#### 2d. Gaussian Mutation

After crossover, each gene (weight value) in the child has a probability `mutation_rate` of being perturbed by Gaussian noise scaled by `mutation_strength`:
```
weight += N(0, 1) × mutation_strength
```
This is analogous to small random perturbations in biological DNA replication.

#### 2e. Elitism

The top `ELITE_K = 8` individuals are copied directly into the next generation without crossover or mutation. This guarantees that the best solution found so far is never lost, ensuring **monotonic progress** in the best-case fitness.

### 3. Adaptive Mutation (Self-Regulating Parameters)

The core research contribution. Traditional GAs use fixed mutation rates, which leads to a fundamental tradeoff:
- **High mutation** = good exploration, but destabilises good solutions.
- **Low mutation** = good exploitation, but gets stuck in local optima.

This project implements a **Dual-Adaptive Mutation** strategy that dynamically tunes both **mutation rate** (how often) and **mutation strength** (how much) using two mutually exclusive strategies:

#### Strategy A — Fitness-Based (Exploitation)

When best fitness improves from the previous generation:
```
mutation_rate     *= 0.9  (decrease, clamped to MUT_RATE_MIN)
mutation_strength *= 0.9  (decrease, clamped to MUT_STR_MIN)
```
Rationale: Progress is being made → reduce perturbation to fine-tune the current solution.

#### Strategy B — Stagnation & Diversity-Based (Exploration)

When fitness has **not improved** for `STAGNATION_N = 5` consecutive generations, OR when population diversity (measured by standard deviation of all chromosomes) drops below `DIVERSITY_THRESHOLD = 0.5`:
```
mutation_rate     *= 1.5  (increase, clamped to MUT_RATE_MAX)
mutation_strength *= 1.5  (increase, clamped to MUT_STR_MAX)
```
Rationale: The search is stuck → inject randomness to escape the local optimum.

#### Why Mutually Exclusive?

The two strategies are chained as `if / elif / else` to prevent a conflict where Strategy A decreases the rate and Strategy B immediately increases it back in the same generation.

### 4. Fitness Sharing (Speciation)

To prevent a single dominant genotype from taking over the population (premature convergence), **fitness sharing** is applied before selection:

1. For each individual, count how many other individuals are within a Euclidean distance of `SHARING_RADIUS = 5.0` in chromosome space.
2. Divide that individual's fitness by the count.

This penalises individuals in crowded niches and rewards unique solutions, maintaining population diversity.

### 5. Population Diversity Monitoring

Diversity is quantified as the **standard deviation** of all flattened weight vectors across the population. This is a scalar measure of how "spread out" the population is in the search space. It is:
- Logged every generation for plotting.
- Used as a trigger for Strategy B of the adaptive mutation system.

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    neuroevolution_snake.py                    │
│              (Main loop, event handling, plotting)            │
├──────────────┬────────────────┬──────────────┬───────────────┤
│  snake_env   │ neural_network │ genetic_algo │  ui_renderer  │
│   (Game)     │   (Brain)      │  (Engine)    │   (Display)   │
├──────────────┼────────────────┼──────────────┼───────────────┤
│              │                │              │               │
│ Snake class  │ NeuralNetwork  │ GeneticAlgo  │  Renderer     │
│ • get_state  │ • forward      │ • select     │  • start menu │
│ • step       │ • get_flat     │ • crossover  │  • game grid  │
│ • fitness    │ • set_flat     │ • mutate     │  • info panel  │
│              │ • clone        │ • sharing    │  • chart       │
│              │                │ • adaptive   │               │
├──────────────┴────────────────┴──────────────┴───────────────┤
│                         config.py                            │
│           (All hyperparameters and constants)                 │
├──────────────────────────────────────────────────────────────┤
│                     run_experiment.py                         │
│       (Headless batch runner for statistical comparison)      │
└──────────────────────────────────────────────────────────────┘
```

---

## File-by-File Breakdown

### `config.py` — Central Configuration

All hyperparameters and constants are defined here as module-level variables, imported by every other file. This makes tuning experiments trivial — change one value and it propagates everywhere.

| Parameter | Value | Purpose |
|---|---|---|
| `GRID_W`, `GRID_H` | 20, 20 | Snake game grid dimensions (in cells) |
| `CELL` | 24 | Pixel size of each grid cell |
| `PANEL_W` | 340 | Width of the right-side info panel |
| `WIN_W`, `WIN_H` | Computed | Total window size (grid + panel) |
| `POP_SIZE` | 150 | Number of snake agents per generation |
| `MAX_STEPS` | 200 | Base starvation limit (steps without eating) |
| `MUTATION_RATE` | 0.1 | Initial probability of mutating each gene |
| `MUT_RATE_MIN/MAX` | 0.01 / 0.5 | Adaptive mutation rate bounds |
| `MUT_STEP_UP/DOWN` | 1.5 / 0.9 | Multiplicative factors for adaptive adjustment |
| `STAGNATION_N` | 5 | Generations of no progress before forcing exploration |
| `DIVERSITY_THRESHOLD` | 0.5 | Minimum population diversity before forcing exploration |
| `MUTATION_STR` | 0.4 | Initial Gaussian noise scale for mutation |
| `MUT_STR_MIN/MAX` | 0.05 / 1.5 | Adaptive mutation strength bounds |
| `ELITE_K` | 8 | Number of top individuals preserved via elitism |
| `TOURNAMENT_K` | 5 | Tournament selection pool size |
| `SHARING_RADIUS` | 5.0 | Euclidean distance threshold for fitness sharing |
| `USE_ADAPTIVE_GA` | True | Toggle between Adaptive and Standard GA (set from UI) |
| `LAYER_SIZES` | [24,16,8,4] | Neural network layer dimensions |
| `FPS_DEFAULT` | 30 | Default rendering frame rate |
| `RENDER_TOP_N` | 3 | Number of top-scoring snakes to render simultaneously |
| `C_*` | RGB tuples | UI color palette |

---

### `neural_network.py` — The Brain

Defines the `NeuralNetwork` class: a fully-connected feed-forward network with no training loop.

#### `__init__(self, layer_sizes)`
Initializes weight matrices and bias vectors between consecutive layers. Weights are sampled from `N(0, 0.5)` (Gaussian with std=0.5) — this is the Xavier-like initialization that gives a reasonable starting signal magnitude.

#### `forward(self, x) → np.array`
The **forward pass**. Takes a 24-dimensional input vector and propagates it through the network:
1. For each hidden layer: compute `x = tanh(x @ W + b)`
2. For the output layer: compute `softmax(x @ W + b)`

The `tanh` activation produces values in `[-1, 1]`, critical for allowing the evolved weights to express both excitatory and inhibitory connections. The `softmax` output gives a probability distribution over 4 directions.

#### `get_flat(self) → np.array`
Serializes all weights and biases into a single 1D array (the "chromosome"). Used by the GA for crossover, mutation, and diversity calculation.

#### `set_flat(self, flat)`
Deserializes a 1D array back into the structured weight matrices and bias vectors. The inverse of `get_flat`.

#### `clone(self) → NeuralNetwork`
Creates a deep copy of the network (used for elitism — preserving top individuals).

---

### `snake_env.py` — The Environment

Defines the `Snake` class: a single snake agent that lives on the grid, perceives its surroundings, and acts based on neural network output.

#### Constants
- `DIRS`: Direction vectors for Up, Right, Down, Left.
- `DIR_UP/RIGHT/DOWN/LEFT`: Integer indices (0–3).

#### `__init__(self, nn)`
Stores a reference to the neural network and calls `reset()`.

#### `reset(self)`
Places the snake at the center of the grid with a body of length 3, facing right. Spawns food randomly.

#### `_spawn_food(self) → tuple`
Randomly places food on a cell not occupied by the snake's body.

#### `get_state(self) → np.array` — Sensory Perception
Constructs the **24-dimensional input vector** the neural network uses to decide its next move:

1. **8 Danger Rays (indices 0–7):** Cast rays in 8 directions (N, NE, E, SE, S, SW, W, NW) from the snake's head. Each ray returns a normalised inverse distance to the nearest wall or body segment. Value of `1.0` means danger is adjacent; closer to `0.0` means far away.
2. **8 Food Rays (indices 8–15):** Cast the same 8 rays, but return a binary `1.0` if food is visible along that ray (not blocked by the body), `0.0` otherwise.
3. **4 Direction One-Hot (indices 16–19):** The snake's current direction encoded as `[1,0,0,0]` for Up, `[0,1,0,0]` for Right, etc.
4. **4 Wall Distances (indices 20–23):** Normalised distances to the four walls.

**Performance Optimization:** The body is converted to a `set` at the start of `get_state` for O(1) membership checks instead of O(n) list scans per ray per cell.

#### `step(self)`
One simulation tick:
1. Compute the state vector via `get_state()`.
2. Forward-pass through the neural network.
3. Pick the direction with the highest output (`argmax`).
4. Prevent 180° reversal (going directly backward kills you in real Snake — here we just ignore the move).
5. Move the head, check for wall/self collision.
6. If food is eaten: grow and respawn food. Otherwise: remove tail.
7. If the snake hasn't eaten in `MAX_STEPS + score × 50` steps: it starves (dies).

#### `compute_fitness(self) → float`
The **fitness function** — the most important design decision in any GA:
```python
if score == 0:
    fitness = steps × 0.1           # tiny reward for surviving
else:
    fitness = score × 1000 + (score / steps) × 500
```
- **Score is the primary signal** — each food eaten is worth 1000 points.
- **Efficiency bonus** — `score / steps` rewards snakes that find food quickly.
- **Survival-only snakes** get almost nothing (0.1 per step), preventing the GA from optimizing for "circle in place" behavior.

---

### `genetic_algorithm.py` — The Evolutionary Engine

Defines the `GeneticAlgorithm` class: manages the population, selection, reproduction, and adaptive parameter control.

#### `__init__(self)`
Creates a population of `POP_SIZE = 150` randomly initialized neural networks. Initializes mutation rate, strength, stagnation counter, and history tracking dictionaries.

#### `calculate_diversity(self) → float`
Flattens all population chromosomes into a matrix and returns the **standard deviation** across all weights. This is the scalar diversity metric used to trigger exploration.

#### `evaluate(self, snakes)`
Calls `compute_fitness()` on every snake after a generation finishes.

#### `tournament_select(self, snakes) → NeuralNetwork`
Randomly samples `TOURNAMENT_K = 5` snakes and returns the neural network of the one with the highest fitness. This is one "parent selection" event.

#### `crossover(self, p1, p2) → NeuralNetwork`
**Layer-wise crossover.** For each of the 3 layer pairs (weights + biases), flip a coin:
- Heads: child gets that layer from Parent 1.
- Tails: child gets that layer from Parent 2.

This preserves co-adapted weights within each layer.

#### `mutate(self, nn) → NeuralNetwork`
Gaussian mutation. Each gene has a `self.mutation_rate` probability of being perturbed by `N(0, 1) × self.mutation_strength`. In Adaptive mode, both `mutation_rate` and `mutation_strength` are dynamically adjusted.

#### `apply_fitness_sharing(self, snakes)`
**Fitness sharing / speciation.** For each snake:
1. Compute Euclidean distances from its chromosome to all other chromosomes.
2. Count how many are within `SHARING_RADIUS = 5.0`.
3. Divide its fitness by that count.

Effect: snakes in crowded regions of the search space get penalized, encouraging the population to spread out and explore diverse strategies.

#### `next_generation(self, snakes)`
The main evolutionary cycle:
1. **Apply fitness sharing** to penalize crowded niches.
2. **Sort** population by shared fitness (descending).
3. **Record** best score, best fitness, average fitness.
4. **Calculate diversity** (std of all chromosomes).
5. **Adaptive mutation adjustment** (if `USE_ADAPTIVE_GA` is True):
   - If fitness improved → decrease rate/strength (exploit).
   - Elif stagnated or low diversity → increase rate/strength (explore).
   - Else → increment stagnation counter.
6. **Log** all metrics to history for plotting.
7. **Elitism:** Copy top `ELITE_K = 8` individuals unchanged.
8. **Fill remaining slots** with tournament-selected, crossed-over, mutated children.
9. Increment generation counter.

---

### `ui_renderer.py` — The Display

Defines the `Renderer` class: handles all Pygame drawing for both the start menu and the main simulation.

#### `__init__(self, screen, font_big, font_med, font_sm)`
Stores screen and font references. Creates additional larger fonts for the start menu. Initializes empty rectangles for hit-testing the toggle and start button.

#### `draw_start_menu(self, mouse_pos)`
Renders the **pre-training start screen**:
- Decorative background grid.
- Centered card with title "NEUROEVOLUTION" and subtitle.
- **Adaptive Mutation toggle:** A visual ON/OFF switch that flips `config.USE_ADAPTIVE_GA`.
- **START button:** Hover-responsive button that transitions to the training state.
- Description text showing the current mode's behavior.

#### `hit_toggle(self, pos) → bool`
Returns True if the given mouse position falls within the adaptive toggle's bounding rectangle.

#### `hit_start(self, pos) → bool`
Returns True if the given mouse position falls within the START button's bounding rectangle.

#### `draw_grid(self)`
Draws the vertical and horizontal grid lines on the game area.

#### `draw_snake(self, snake)`
Renders a snake's body segments (green, with a brighter head) and its food pellet (red circle).

#### `draw_panel(self, ga, fps, paused, snakes)`
Renders the right-side information panel during training:
- Title and subtitle.
- **GA mode indicator** (ADAPTIVE GA in green, STANDARD GA in red).
- Generation number, population size, best score, best fitness.
- Alive count and FPS.
- Pause indicator.
- **Mini fitness chart** (best and average fitness over generations).
- Controls reference.
- Architecture summary (layer sizes, current mutation rate, diversity, mutation strength, elite count, tournament size).

#### `_draw_chart(self, x, y, w, h, ga)`
Draws the inline fitness-over-generations chart. Plots two lines:
- **Cyan (best fitness):** The best individual's fitness per generation.
- **Dim gray (average fitness):** The population's average fitness per generation.

Both are normalized to the maximum value for auto-scaling.

---

### `neuroevolution_snake.py` — Main Entry Point

Orchestrates the entire application: window setup, state machine, event loop, and result export.

#### `plot_results(ga)`
Called on exit (Q/ESC/window close). Generates a 2-panel matplotlib figure:
- **Left panel:** Best fitness vs generations.
- **Right panel:** Mutation rate vs generations.

Saves to `ga_results_adaptive.png` or `ga_results_standard.png` depending on the mode. Prints a summary to the console.

#### `run()`
The main function. Sets up Pygame, creates the renderer, and enters a **state machine** with two states:

**State: `menu`**
- Renders the start screen via `renderer.draw_start_menu()`.
- Listens for mouse clicks on the toggle (flips adaptive mode) and start button (transitions to `running`).
- No GA or snakes are created yet.

**State: `running`**
- Creates a `GeneticAlgorithm` and initial population of `Snake` objects.
- Each frame: steps all alive snakes, checks if the generation is done (all dead), and triggers the GA's `evaluate → next_generation` cycle.
- Renders the grid, top snakes, and info panel.
- Keyboard controls: SPACE (pause), R (restart), +/- (speed), Q/ESC (quit and save).

---

### `run_experiment.py` — Headless Batch Runner

A separate script for **statistically rigorous comparison** between Adaptive and Standard GA. No rendering, no Pygame — pure headless simulation.

#### `run_trial(adaptive, trial_id) → list`
Runs a single trial:
1. Sets `config.USE_ADAPTIVE_GA` to the specified mode.
2. Creates a fresh `GeneticAlgorithm`.
3. For each of `NUM_GENERATIONS = 100` generations:
   - Creates all snakes, runs them to death, evaluates fitness, evolves.
   - Records best fitness.
4. Returns the per-generation fitness curve.

#### `main()`
1. Runs `NUM_TRIALS = 10` trials in Adaptive mode.
2. Runs `NUM_TRIALS = 10` trials in Standard mode.
3. Saves raw data to `experiment_results.json`.
4. Plots mean fitness curves with **95% confidence intervals** (±1.96 × std / √n) for both modes.
5. Saves to `experiment_comparison.png`.
6. Prints final-generation summary statistics.

---

## Getting Started

### Requirements
- Python 3.12+
- Libraries: `pygame`, `numpy`, `matplotlib`

### Setup & Execution
```powershell
# Create virtual environment
python -m venv venv
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch interactive simulation
python neuroevolution_snake.py

# Run headless batch experiment (takes several minutes)
python run_experiment.py
```

---

## Interactive Controls

| Key | Action |
|:---|:---|
| **Start Screen** | Click toggle to switch Adaptive/Standard, click START to begin |
| `SPACE` | Pause / Resume simulation |
| `R` | Restart from Generation 1 |
| `+` / `-` | Speed up / Slow down (adjust FPS) |
| `Q` / `ESC` | Quit and save performance plots |

---

## Experiment Runner

Run `python run_experiment.py` to execute 10 trials of each mode (Adaptive vs Standard) for 100 generations each. Outputs:

| File | Contents |
|:---|:---|
| `experiment_results.json` | Raw per-generation fitness data for all 20 trials |
| `experiment_comparison.png` | Mean fitness curves with 95% confidence intervals |

---

*Developed for Soft Computing Research — exploring Adaptive Evolutionary Computation, Neuroevolution, and Neural Network Optimization without Gradient Descent.*
