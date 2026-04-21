# ui_renderer.py
import pygame
import config
from snake_env import Snake
from genetic_algorithm import GeneticAlgorithm

class Renderer:
    def __init__(self, screen, font_big, font_med, font_sm):
        self.screen    = screen
        self.font_big  = font_big
        self.font_med  = font_med
        self.font_sm   = font_sm

    def draw_grid(self):
        for x in range(0, config.GRID_W * config.CELL, config.CELL):
            pygame.draw.line(self.screen, config.C_GRID, (x, 0), (x, config.WIN_H))
        for y in range(0, config.WIN_H, config.CELL):
            pygame.draw.line(self.screen, config.C_GRID, (0, y), (config.GRID_W * config.CELL, y))

    def draw_snake(self, snake: Snake, alpha=255):
        for i, (cx, cy) in enumerate(snake.body):
            col = config.C_SNAKEH if i == 0 else config.C_SNAKE
            r   = pygame.Rect(cx*config.CELL+1, cy*config.CELL+1, config.CELL-2, config.CELL-2)
            pygame.draw.rect(self.screen, col, r, border_radius=4)

        # food
        fx, fy = snake.food
        fr = pygame.Rect(fx*config.CELL+3, fy*config.CELL+3, config.CELL-6, config.CELL-6)
        pygame.draw.rect(self.screen, config.C_FOOD, fr, border_radius=config.CELL//2)

    def draw_panel(self, ga: GeneticAlgorithm, fps, paused, snakes):
        px = config.GRID_W * config.CELL
        pygame.draw.rect(self.screen, config.C_PANEL, (px, 0, config.PANEL_W, config.WIN_H))
        pygame.draw.line(self.screen, config.C_ACCENT, (px, 0), (px, config.WIN_H), 2)

        y = 18
        def txt(text, font, color, indent=12):
            nonlocal y
            surf = font.render(text, True, color)
            self.screen.blit(surf, (px + indent, y))
            y += surf.get_height() + 4

        txt("NEUROEVOLUTION", self.font_big, config.C_ACCENT)
        txt("Snake  x  Genetic Algorithm", self.font_sm, config.C_DIM)
        y += 10
        pygame.draw.line(self.screen, config.C_DIM, (px+8, y), (px+config.PANEL_W-8, y)); y += 10

        txt(f"Generation   {ga.generation}", self.font_med, config.C_WHITE)
        txt(f"Population   {config.POP_SIZE}", self.font_sm, config.C_DIM)
        txt(f"Best Score   {ga.best_score}", self.font_med, config.C_GOOD)
        txt(f"Best Fitness {ga.best_fitness:.1f}", self.font_sm, config.C_DIM)

        y += 6
        pygame.draw.line(self.screen, config.C_DIM, (px+8, y), (px+config.PANEL_W-8, y)); y += 10

        alive = sum(1 for s in snakes if s.alive)
        txt(f"Alive   {alive} / {config.POP_SIZE}", self.font_med, config.C_WHITE)
        txt(f"FPS     {fps:.0f}", self.font_sm, config.C_DIM)
        if paused:
            txt("[ PAUSED ]", self.font_med, config.C_BAD)

        y += 6
        pygame.draw.line(self.screen, config.C_DIM, (px+8, y), (px+config.PANEL_W-8, y)); y += 10

        # mini fitness chart
        txt("Fitness over Generations", self.font_sm, config.C_DIM)
        self._draw_chart(px+10, y, config.PANEL_W-20, 90, ga)
        y += 96

        y += 8
        pygame.draw.line(self.screen, config.C_DIM, (px+8, y), (px+config.PANEL_W-8, y)); y += 10
        txt("Controls", self.font_sm, config.C_ACCENT)
        txt("SPACE  pause / resume", self.font_sm, config.C_DIM)
        txt("R      restart", self.font_sm, config.C_DIM)
        txt("+/-    speed up/down", self.font_sm, config.C_DIM)
        txt("Q/ESC  quit", self.font_sm, config.C_DIM)

        y += 8
        pygame.draw.line(self.screen, config.C_DIM, (px+8, y), (px+config.PANEL_W-8, y)); y += 10
        txt("Architecture", self.font_sm, config.C_ACCENT)
        arch = " -> ".join(str(s) for s in config.LAYER_SIZES)
        txt(arch, self.font_sm, config.C_DIM)
        txt(f"Mutation rate  {ga.mutation_rate:.4f}", self.font_sm, config.C_DIM)
        diversity = ga.history['diversity'][-1] if ga.history['diversity'] else 0
        txt(f"Diversity      {diversity:.4f}", self.font_sm, config.C_DIM)
        txt(f"Mutation str   {ga.mutation_strength:.4f}", self.font_sm, config.C_DIM)
        txt(f"Elite keep     {config.ELITE_K}", self.font_sm, config.C_DIM)
        txt(f"Tournament K   {config.TOURNAMENT_K}", self.font_sm, config.C_DIM)

    def _draw_chart(self, x, y, w, h, ga: GeneticAlgorithm):
        pygame.draw.rect(self.screen, (20, 24, 44), (x, y, w, h), border_radius=4)
        bh = ga.best_fitness_history
        ah = ga.avg_fitness_history
        if len(bh) < 2:
            return
        mx = max(bh) if max(bh) > 0 else 1

        def pts(hist, col):
            n = len(hist)
            points = []
            for i, v in enumerate(hist):
                px_ = x + int(i / (n-1) * w)
                py_ = y + h - int(v / mx * (h - 4)) - 2
                points.append((px_, py_))
            if len(points) > 1:
                pygame.draw.lines(self.screen, col, False, points, 2)

        pts(ah, config.C_DIM)
        pts(bh, config.C_ACCENT)

        # legend
        pygame.draw.line(self.screen, config.C_ACCENT, (x+4, y+h-8), (x+18, y+h-8), 2)
        s = self.font_sm.render("best", True, config.C_ACCENT)
        self.screen.blit(s, (x+22, y+h-14))
        pygame.draw.line(self.screen, config.C_DIM, (x+60, y+h-8), (x+74, y+h-8), 2)
        s = self.font_sm.render("avg", True, config.C_DIM)
        self.screen.blit(s, (x+78, y+h-14))
