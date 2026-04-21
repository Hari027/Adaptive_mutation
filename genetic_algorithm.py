# genetic_algorithm.py
import numpy as np
import random
import config
from neural_network import NeuralNetwork

class GeneticAlgorithm:
    def __init__(self):
        self.population = [NeuralNetwork() for _ in range(config.POP_SIZE)]
        self.generation  = 1
        self.best_score  = 0
        self.best_fitness= 0
        self.avg_fitness_history = []
        self.best_fitness_history= []
        # ADAPTIVE MUTATION CHANGE
        self.mutation_rate = config.MUTATION_RATE
        self.mutation_strength = config.MUTATION_STR
        self.stagnation_counter = 0
        self.last_best_fitness = 0
        self.history = {
            'fitness': [],
            'mutation': [],
            'strength': [],
            'diversity': []
        }

    def calculate_diversity(self):
        """Compute population diversity using std of chromosome weights."""
        # ADAPTIVE MUTATION CHANGE
        all_weights = np.array([nn.get_flat() for nn in self.population])
        return np.std(all_weights)

    def evaluate(self, snakes):
        for s in snakes:
            s.compute_fitness()

    def tournament_select(self, snakes):
        pool = random.sample(snakes, config.TOURNAMENT_K)
        return max(pool, key=lambda s: s.fitness).nn

    def crossover(self, p1: NeuralNetwork, p2: NeuralNetwork) -> NeuralNetwork:
        f1, f2  = p1.get_flat(), p2.get_flat()
        mask    = np.random.rand(len(f1)) < 0.5
        child_f = np.where(mask, f1, f2)
        child   = NeuralNetwork()
        child.set_flat(child_f)
        return child

    def mutate(self, nn: NeuralNetwork) -> NeuralNetwork:
        f    = nn.get_flat()
        # ADAPTIVE MUTATION CHANGE: Use dynamic mutation rate and strength
        mask = np.random.rand(len(f)) < self.mutation_rate
        f[mask] += np.random.randn(mask.sum()) * self.mutation_strength
        nn.set_flat(f)
        return nn

    def next_generation(self, snakes):
        snakes.sort(key=lambda s: s.fitness, reverse=True)
        self.best_score   = max(s.score   for s in snakes)
        self.best_fitness = snakes[0].fitness
        avg = np.mean([s.fitness for s in snakes])
        self.avg_fitness_history.append(avg)
        self.best_fitness_history.append(self.best_fitness)

        # ADAPTIVE MUTATION CHANGE: Combined logic (Strategy A & B)
        diversity = self.calculate_diversity()
        
        # Strategy A: Fitness-based
        if self.best_fitness > self.last_best_fitness:
            self.stagnation_counter = 0
            # Progress -> Decrease mutation (exploitation)
            self.mutation_rate = max(config.MUT_RATE_MIN, self.mutation_rate * config.MUT_STEP_DOWN)
            self.mutation_strength = max(config.MUT_STR_MIN, self.mutation_strength * config.MUT_STEP_DOWN)
        else:
            self.stagnation_counter += 1
            
        # Strategy B: Diversity-based or stagnation
        if self.stagnation_counter >= config.STAGNATION_N or diversity < config.DIVERSITY_THRESHOLD:
            # Need exploration -> Increase mutation rate and strength
            self.mutation_rate = min(config.MUT_RATE_MAX, self.mutation_rate * config.MUT_STEP_UP)
            self.mutation_strength = min(config.MUT_STR_MAX, self.mutation_strength * config.MUT_STEP_UP)
            self.stagnation_counter = 0 # reset to allow gradual increase/decrease
            
        self.last_best_fitness = self.best_fitness
        
        # Log to history for plotting
        self.history['fitness'].append(self.best_fitness)
        self.history['mutation'].append(self.mutation_rate)
        self.history['strength'].append(self.mutation_strength)
        self.history['diversity'].append(diversity)
        
        print(f"Gen {self.generation} | Best: {self.best_fitness:.2f} | Mut: {self.mutation_rate:.4f} | Str: {self.mutation_strength:.4f} | Div: {diversity:.4f}")

        new_pop = []
        # elitism
        for i in range(config.ELITE_K):
            new_pop.append(snakes[i].nn.clone())

        # fill rest with crossover + mutation
        while len(new_pop) < config.POP_SIZE:
            p1 = self.tournament_select(snakes)
            p2 = self.tournament_select(snakes)
            child = self.crossover(p1, p2)
            child = self.mutate(child)
            new_pop.append(child)

        self.population = new_pop
        self.generation += 1
