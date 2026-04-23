# config.py

# ─────────────────────────────────────────────
#  GAME CONFIGURATION
# ─────────────────────────────────────────────
GRID_W, GRID_H   = 20, 20          # grid cells
CELL             = 24              # pixels per cell
PANEL_W          = 340             # right-side info panel
WIN_W            = GRID_W * CELL + PANEL_W
WIN_H            = GRID_H * CELL

# ─────────────────────────────────────────────
#  GA PARAMETERS
# ─────────────────────────────────────────────
POP_SIZE         = 150             # snakes per generation
MAX_STEPS        = 200             # steps before a snake "starves"
# ADAPTIVE MUTATION PARAMETERS
MUTATION_RATE    = 0.1             # Initial mutation rate
MUT_RATE_MIN     = 0.01            # Minimum allowed mutation rate
MUT_RATE_MAX     = 0.5             # Maximum allowed mutation rate
MUT_STEP_UP      = 1.5             # Factor to increase mutation during stagnation
MUT_STEP_DOWN     = 0.9             # Factor to decrease mutation during progress
STAGNATION_N     = 5               # Generations to wait before increasing mutation
DIVERSITY_THRESHOLD = 0.5          # Threshold for low population diversity
MUTATION_STR     = 0.4             # Initial mutation strength
MUT_STR_MIN      = 0.05            # Minimum allowed mutation strength
MUT_STR_MAX      = 1.5             # Maximum allowed mutation strength
ELITE_K          = 8               # top agents kept unchanged
TOURNAMENT_K     = 5               # tournament selection size
SHARING_RADIUS   = 5.0             # fitness sharing niche radius
USE_ADAPTIVE_GA  = True            # toggle adaptive mutation (set from UI)

# ─────────────────────────────────────────────
#  NN ARCHITECTURE
# ─────────────────────────────────────────────
LAYER_SIZES      = [24, 16, 8, 4]  # 24 inputs → 16 → 8 → 4 outputs

# ─────────────────────────────────────────────
#  LOOP CONFIGURATION
# ─────────────────────────────────────────────
FPS_DEFAULT      = 30
RENDER_TOP_N     = 3               # how many snakes to show per gen

# ─────────────────────────────────────────────
#  COLOURS
# ─────────────────────────────────────────────
C_BG        = (10,  12,  20)
C_GRID      = (18,  22,  38)
C_SNAKE     = (0,  220, 130)
C_SNAKEH    = (0,  255, 160)
C_FOOD      = (255, 70,  90)
C_PANEL     = (14,  17,  30)
C_ACCENT    = (0,  200, 255)
C_DIM       = (60,  70, 100)
C_WHITE     = (230, 235, 255)
C_GOOD      = (0,  220, 130)
C_BAD       = (255, 80,  80)
