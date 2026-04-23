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
        # Pre-create a larger font for the start menu title
        try:
            self.font_title = pygame.font.SysFont("consolas", 32, bold=True)
            self.font_btn   = pygame.font.SysFont("consolas", 22, bold=True)
        except:
            self.font_title = pygame.font.SysFont(None, 36, bold=True)
            self.font_btn   = pygame.font.SysFont(None, 26, bold=True)

        # Cached rects for start-menu hit testing
        self._toggle_rect = pygame.Rect(0, 0, 0, 0)
        self._start_rect  = pygame.Rect(0, 0, 0, 0)

    # ── START MENU ──────────────────────────────────
    def draw_start_menu(self, mouse_pos):
        """Draw a start screen with an adaptive toggle and start button."""
        sw, sh = self.screen.get_size()
        self.screen.fill(config.C_BG)

        # decorative grid (dimmed)
        for x in range(0, sw, config.CELL):
            pygame.draw.line(self.screen, config.C_GRID, (x, 0), (x, sh))
        for y in range(0, sh, config.CELL):
            pygame.draw.line(self.screen, config.C_GRID, (0, y), (sw, y))

        # --- centre card ---
        card_w, card_h = 420, 340
        cx, cy = sw // 2 - card_w // 2, sh // 2 - card_h // 2
        card_rect = pygame.Rect(cx, cy, card_w, card_h)
        # card bg with subtle border
        pygame.draw.rect(self.screen, config.C_PANEL, card_rect, border_radius=12)
        pygame.draw.rect(self.screen, config.C_ACCENT, card_rect, 2, border_radius=12)

        # title
        title = self.font_title.render("NEUROEVOLUTION", True, config.C_ACCENT)
        self.screen.blit(title, (cx + card_w // 2 - title.get_width() // 2, cy + 24))

        subtitle = self.font_med.render("Snake  ×  Genetic Algorithm", True, config.C_DIM)
        self.screen.blit(subtitle, (cx + card_w // 2 - subtitle.get_width() // 2, cy + 64))

        # divider
        pygame.draw.line(self.screen, config.C_DIM, (cx + 20, cy + 94), (cx + card_w - 20, cy + 94))

        # ── adaptive toggle ──
        tog_y = cy + 120
        label = self.font_med.render("Adaptive Mutation", True, config.C_WHITE)
        self.screen.blit(label, (cx + 40, tog_y + 4))

        # toggle track
        track_w, track_h = 56, 28
        track_x = cx + card_w - 40 - track_w
        track_rect = pygame.Rect(track_x, tog_y, track_w, track_h)
        self._toggle_rect = track_rect

        is_on = config.USE_ADAPTIVE_GA
        track_color = config.C_GOOD if is_on else (60, 65, 85)
        pygame.draw.rect(self.screen, track_color, track_rect, border_radius=14)

        # toggle knob
        knob_r = 10
        knob_x = track_x + track_w - knob_r - 6 if is_on else track_x + knob_r + 6
        knob_y = tog_y + track_h // 2
        pygame.draw.circle(self.screen, config.C_WHITE, (knob_x, knob_y), knob_r)

        # status label
        status = self.font_sm.render("ON" if is_on else "OFF", True, config.C_GOOD if is_on else config.C_BAD)
        self.screen.blit(status, (track_x + track_w + 8, tog_y + 6))

        # mode description
        if is_on:
            desc = "Mutation rate adapts to fitness & diversity"
        else:
            desc = "Fixed mutation rate (standard GA)"
        desc_surf = self.font_sm.render(desc, True, config.C_DIM)
        self.screen.blit(desc_surf, (cx + 40, tog_y + 36))

        # ── start button ──
        btn_w, btn_h = 200, 50
        btn_x = cx + card_w // 2 - btn_w // 2
        btn_y = cy + card_h - 90
        btn_rect = pygame.Rect(btn_x, btn_y, btn_w, btn_h)
        self._start_rect = btn_rect

        hover = btn_rect.collidepoint(mouse_pos)
        btn_color = (0, 240, 150) if hover else config.C_ACCENT
        pygame.draw.rect(self.screen, btn_color, btn_rect, border_radius=8)

        btn_label = self.font_btn.render("START", True, config.C_BG)
        self.screen.blit(btn_label, (btn_x + btn_w // 2 - btn_label.get_width() // 2,
                                     btn_y + btn_h // 2 - btn_label.get_height() // 2))

        # controls hint at bottom
        hint = self.font_sm.render("Click the toggle to switch GA mode", True, config.C_DIM)
        self.screen.blit(hint, (cx + card_w // 2 - hint.get_width() // 2, cy + card_h - 30))

    def hit_toggle(self, pos):
        """Return True if pos clicks the adaptive toggle."""
        return self._toggle_rect.collidepoint(pos)

    def hit_start(self, pos):
        """Return True if pos clicks the start button."""
        return self._start_rect.collidepoint(pos)

    # ── GAME UI ─────────────────────────────────────
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

        # GA mode indicator
        mode_label = "ADAPTIVE GA" if config.USE_ADAPTIVE_GA else "STANDARD GA"
        mode_color = config.C_GOOD if config.USE_ADAPTIVE_GA else config.C_BAD
        txt(f"Mode: {mode_label}", self.font_med, mode_color)
        y += 4

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

