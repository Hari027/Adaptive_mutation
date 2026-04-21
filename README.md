# Adaptive Genetic Algorithm for Neural Network Weight Optimization

An advanced implementation of a Neural Network trained via a **Dual-Adaptive Genetic Algorithm (DAGA)** to solve the Snake navigation problem. This project demonstrates high-level **Soft Computing** principles, specifically focused on solving the common problem of **premature convergence** through autonomous parameter tuning.

---

## 🧬 Research Focus: Dual-Adaptive Mutation
Traditional Genetic Algorithms use fixed parameters, which often lead to populations getting stuck in local optima. This project implements a **Multi-Strategy Adaptive Logic** to balance **Exploration** and **Exploitation**:

### 1. Adaptive Mutation Rate (Search Frequency)
The algorithm monitors the best fitness score and population diversity. 
- **Stagnation**: If the score does not improve for `STAGNATION_N` generations, the mutation rate **increases** (up to 50%) to force the population to try radically different strategies.
- **Progress**: When improvement is detected, the rate **decreases** to allow the population to "exploit" and stabilize.

### 2. Adaptive Mutation Strength (Search Radius)
Not only how *often* we mutate, but how *much* we change.
- **High Variance**: During stagnation, the "Mutation Strength" (magnitude of weight change) increases. This allows the GA to "jump" across the search space to find entirely new behaviors.
- **Precision Tuning**: In stable generations, the strength decreases for fine-grained local optimization.

### 3. Diversity Monitoring (Standard Deviation)
The engine calculates the Standard Deviation of the entire population's weight distribution. If diversity drops below a threshold, the system automatically triggers a mutation spike to prevent "genetic cloning."

---

## 📊 Performance Analysis & Reporting
The system is built for experimental verification. Upon exiting the simulation (`Esc` or `Q`), it automatically generates a research report (`ga_adaptive_results.png`) containing:
- **Best Fitness vs Generations**: Visualizes learning progress and convergence speed.
- **Mutation Strategy vs Generations**: Visualizes how the algorithm "reacted" to stagnation by scaling rate and strength.

---

## 🏗 System Architecture

### Neural Network ("The Brain")
- **Inputs (24):** 8-directional ray-casting vision (distance to walls, food, and self) + one-hot motion state.
- **Hidden Layers:** [16, 8] neurons using the `tanh` activation function.
- **Outputs (4):** Probability distribution for directions (Up, Down, Left, Right).

### Genetic Algorithm ("The Engine")
- **Initialization**: Population size of 150 unique agents.
- **Selection**: Tournament Selection (Size 5) for robust parent choosing.
- **Crossover**: Uniform Crossover (50% weight swap probability).
- **Elitism**: Top-performing individuals are carried over unchanged to ensure strictly monotonic progress.

---

## 🚀 Getting Started

### 1. Requirements
- Python 3.12+
- Libraries: `pygame`, `numpy`, `matplotlib`

### 2. Setup & Execution
```powershell
# Setup environment
python -m venv venv
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch simulation
python neuroevolution_snake.py
```

---

## 🎮 Interactive Controls
| Key | Action |
| :--- | :--- |
| `SPACE` | Pause / Resume |
| `R` | Reset Generation (Restart from Gen 1) |
| `+` / `-` | Speed Control (Adjust FPS for fast training) |
| `Q` / `ESC` | Exit & Generate Performance Report |

---
*Developed for Soft Computing Research - exploring Adaptive Evolutionary Computation and Neural Logic.*
