"""
Headless batch experiment runner for Adaptive vs Standard GA comparison.
========================================================================
Runs multiple trials of each mode without rendering, then saves
statistical results (means + confidence intervals) as a plot.

HOW TO RUN:
    python run_experiment.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import config
from snake_env import Snake
from genetic_algorithm import GeneticAlgorithm

# ── EXPERIMENT SETTINGS ──
NUM_TRIALS      = 10    # runs per mode
NUM_GENERATIONS = 100   # generations per trial

def run_trial(adaptive: bool, trial_id: int):
    """Run a single headless trial and return per-generation best fitness."""
    config.USE_ADAPTIVE_GA = adaptive
    mode = "ADAPTIVE" if adaptive else "STANDARD"
    
    ga = GeneticAlgorithm()
    fitness_curve = []

    for gen in range(NUM_GENERATIONS):
        snakes = [Snake(nn) for nn in ga.population]
        for s in snakes:
            while s.alive:
                s.step()
        ga.evaluate(snakes)
        ga.next_generation(snakes)
        fitness_curve.append(ga.best_fitness)
        
        if (gen + 1) % 25 == 0:
            print(f"  [{mode}] Trial {trial_id+1}/{NUM_TRIALS}  Gen {gen+1}/{NUM_GENERATIONS}  Best: {ga.best_fitness:.2f}")

    return fitness_curve

def main():
    results = {'adaptive': [], 'standard': []}

    # ── Run adaptive trials ──
    print("=" * 60)
    print(f"Running {NUM_TRIALS} ADAPTIVE trials ({NUM_GENERATIONS} gens each)...")
    print("=" * 60)
    for t in range(NUM_TRIALS):
        curve = run_trial(adaptive=True, trial_id=t)
        results['adaptive'].append(curve)

    # ── Run standard trials ──
    print("\n" + "=" * 60)
    print(f"Running {NUM_TRIALS} STANDARD trials ({NUM_GENERATIONS} gens each)...")
    print("=" * 60)
    for t in range(NUM_TRIALS):
        curve = run_trial(adaptive=False, trial_id=t)
        results['standard'].append(curve)

    # ── Save raw data ──
    with open('experiment_results.json', 'w') as f:
        json.dump(results, f)
    print("\nRaw data saved to experiment_results.json")

    # ── Plot comparison ──
    adaptive_arr = np.array(results['adaptive'])   # (trials, gens)
    standard_arr = np.array(results['standard'])

    gens = np.arange(1, NUM_GENERATIONS + 1)

    adapt_mean = adaptive_arr.mean(axis=0)
    adapt_std  = adaptive_arr.std(axis=0)
    std_mean   = standard_arr.mean(axis=0)
    std_std    = standard_arr.std(axis=0)

    # 95% confidence interval (approx ± 1.96 * std / sqrt(n))
    ci_factor = 1.96 / np.sqrt(NUM_TRIALS)

    plt.figure(figsize=(12, 6))
    
    # Adaptive
    plt.plot(gens, adapt_mean, color='#00dc82', linewidth=2, label='Adaptive GA (mean)')
    plt.fill_between(gens,
                     adapt_mean - adapt_std * ci_factor,
                     adapt_mean + adapt_std * ci_factor,
                     color='#00dc82', alpha=0.2, label='Adaptive 95% CI')

    # Standard
    plt.plot(gens, std_mean, color='#ff5050', linewidth=2, label='Standard GA (mean)')
    plt.fill_between(gens,
                     std_mean - std_std * ci_factor,
                     std_mean + std_std * ci_factor,
                     color='#ff5050', alpha=0.2, label='Standard 95% CI')

    plt.title(f'Adaptive vs Standard GA — {NUM_TRIALS} Trials × {NUM_GENERATIONS} Generations', fontsize=14)
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness')
    plt.legend(loc='upper left')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('experiment_comparison.png', dpi=150)
    print("Comparison plot saved to experiment_comparison.png")

    # ── Print summary stats ──
    print("\n" + "=" * 60)
    print("FINAL RESULTS (fitness at last generation):")
    print("=" * 60)
    print(f"  Adaptive:  mean={adaptive_arr[:, -1].mean():.2f}  std={adaptive_arr[:, -1].std():.2f}")
    print(f"  Standard:  mean={standard_arr[:, -1].mean():.2f}  std={standard_arr[:, -1].std():.2f}")

if __name__ == "__main__":
    main()
