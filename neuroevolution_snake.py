"""
Neuroevolution Snake — Neural Network trained via Genetic Algorithm
====================================================================
No backpropagation. Weights are evolved using selection, crossover, mutation.

HOW TO RUN:
    pip install pygame numpy
    python neuroevolution_snake.py

CONTROLS:
    SPACE  — pause/resume
    R      — restart from generation 1
    +/-    — speed up / slow down simulation
    Q/ESC  — quit
"""

import pygame
import sys
import matplotlib.pyplot as plt
import config
from snake_env import Snake
from genetic_algorithm import GeneticAlgorithm
from ui_renderer import Renderer

def plot_results(ga):
    """ADAPTIVE MUTATION CHANGE: Save research graphs on exit."""
    if not ga.history['fitness']:
        return

    plt.figure(figsize=(12, 5))
    
    # Plot 1: Fitness
    plt.subplot(1, 2, 1)
    plt.plot(ga.history['fitness'], color='forestgreen', linewidth=2)
    plt.title('Agent Performance: Best Fitness vs Generations')
    plt.xlabel('Generations')
    plt.ylabel('Fitness Score')
    plt.grid(alpha=0.3)

    # Plot 2: Mutation Rate
    plt.subplot(1, 2, 2)
    plt.plot(ga.history['mutation'], color='royalblue', linewidth=2)
    plt.title('Adaptive Strategy: Mutation Rate vs Generations')
    plt.xlabel('Generations')
    plt.ylabel('Mutation Rate')
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('ga_adaptive_results.png')
    print(f"\n[SUMMARY] Research data saved to 'ga_adaptive_results.png'")
    print(f"[SUMMARY] Total Generations: {ga.generation - 1}")
    print(f"[SUMMARY] Final Diversity:   {ga.history['diversity'][-1]:.4f}" if ga.history['diversity'] else "")

def run():
    pygame.init()
    screen = pygame.display.set_mode((config.WIN_W, config.WIN_H))
    pygame.display.set_caption("Neuroevolution Snake — GA-trained Neural Network")

    try:
        font_big = pygame.font.SysFont("consolas", 18, bold=True)
        font_med = pygame.font.SysFont("consolas", 14, bold=True)
        font_sm  = pygame.font.SysFont("consolas", 12)
    except:
        font_big = pygame.font.SysFont(None, 20, bold=True)
        font_med = pygame.font.SysFont(None, 16, bold=True)
        font_sm  = pygame.font.SysFont(None, 14)

    renderer = Renderer(screen, font_big, font_med, font_sm)
    clock    = pygame.time.Clock()

    ga      = GeneticAlgorithm()
    snakes  = [Snake(nn) for nn in ga.population]

    paused  = False
    fps_cap = config.FPS_DEFAULT
    frame   = 0

    while True:
        # ── events ──
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                plot_results(ga)
                pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    plot_results(ga)
                    pygame.quit(); sys.exit()
                if event.key == pygame.K_SPACE:
                    paused = not paused
                if event.key == pygame.K_r:
                    ga     = GeneticAlgorithm()
                    snakes = [Snake(nn) for nn in ga.population]
                if event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    fps_cap = min(fps_cap + 10, 300)
                if event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    fps_cap = max(fps_cap - 10, 5)

        if not paused:
            # step all alive snakes
            for s in snakes:
                if s.alive:
                    s.step()

            # check if generation done
            if not any(s.alive for s in snakes):
                ga.evaluate(snakes)
                ga.next_generation(snakes)
                snakes = [Snake(nn) for nn in ga.population]

        # ── render ──
        screen.fill(config.C_BG)
        renderer.draw_grid()

        # draw best alive snake (or best overall)
        alive_snakes = [s for s in snakes if s.alive]
        show = sorted(alive_snakes, key=lambda s: s.score, reverse=True)[:config.RENDER_TOP_N] if alive_snakes else []
        for s in show:
            renderer.draw_snake(s)

        renderer.draw_panel(ga, clock.get_fps(), paused, snakes)
        pygame.display.flip()
        clock.tick(fps_cap)
        frame += 1


if __name__ == "__main__":
    run()
