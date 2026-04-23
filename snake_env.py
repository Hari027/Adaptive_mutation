# snake_env.py
import numpy as np
import random
import config
from neural_network import NeuralNetwork

DIRS = [(0,-1),(1,0),(0,1),(-1,0)]   # U R D L
DIR_UP, DIR_RIGHT, DIR_DOWN, DIR_LEFT = 0, 1, 2, 3

class Snake:
    def __init__(self, nn: NeuralNetwork):
        self.nn      = nn
        self.reset()

    def reset(self):
        cx, cy       = config.GRID_W//2, config.GRID_H//2
        self.body    = [(cx, cy), (cx-1, cy), (cx-2, cy)]
        self.dir     = DIR_RIGHT
        self.food    = self._spawn_food()
        self.alive   = True
        self.steps   = 0
        self.score   = 0
        self.fitness = 0
        self.steps_since_food = 0

    def _spawn_food(self):
        while True:
            f = (random.randint(0, config.GRID_W-1), random.randint(0, config.GRID_H-1))
            if f not in self.body:
                return f

    def get_state(self):
        """
        24-dimensional input vector:
          - 8 danger rays (distance to wall or self, normalised) × 1
          - 8 food rays (is food in that direction?) × 1
          - 4 one-hot current direction
          - 4 normalised distances to walls (up/right/down/left)
        """
        hx, hy = self.body[0]
        body_set = set(self.body[1:])  # O(1) lookups instead of O(n)
        state  = []

        # 8 rays: N NE E SE S SW W NW
        ray_dirs = [(0,-1),(1,-1),(1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1)]
        for dx, dy in ray_dirs:
            dist = 0
            x, y = hx + dx, hy + dy
            hit  = False
            while 0 <= x < config.GRID_W and 0 <= y < config.GRID_H:
                if (x, y) in body_set:
                    hit = True; break
                dist += 1
                x += dx; y += dy
            # normalised distance (1 = right next to wall/self, 0 = far)
            state.append(1.0 / (dist + 1) if not hit else 1.0)

        # 8 food rays (binary: food visible in that direction?)
        fx, fy = self.food
        for dx, dy in ray_dirs:
            x, y  = hx + dx, hy + dy
            found = False
            while 0 <= x < config.GRID_W and 0 <= y < config.GRID_H:
                if (x, y) == (fx, fy):
                    found = True; break
                if (x, y) in body_set:
                    break
                x += dx; y += dy
            state.append(1.0 if found else 0.0)

        # one-hot direction
        state += [1.0 if self.dir == d else 0.0 for d in range(4)]

        # distances to walls (normalised)
        state.append(hy / (config.GRID_H - 1))          # up
        state.append((config.GRID_W - 1 - hx) / (config.GRID_W - 1))  # right
        state.append((config.GRID_H - 1 - hy) / (config.GRID_H - 1))  # down
        state.append(hx / (config.GRID_W - 1))          # left

        return np.array(state, dtype=np.float32)

    def step(self):
        if not self.alive:
            return
        state  = self.get_state()
        output = self.nn.forward(state)
        action = int(np.argmax(output))           # 0=straight 1=right 2=left 3=reverse (ignored implicitly)

        # map action to absolute direction (can't go directly backward)
        new_dir = action
        if (new_dir + 2) % 4 == self.dir:         # would reverse
            new_dir = self.dir                     # keep going straight
        self.dir = new_dir

        dx, dy   = DIRS[self.dir]
        hx, hy   = self.body[0]
        nx, ny   = hx + dx, hy + dy

        # collision check
        if not (0 <= nx < config.GRID_W and 0 <= ny < config.GRID_H) or (nx, ny) in self.body[1:]:
            self.alive = False
            return

        self.body.insert(0, (nx, ny))
        self.steps += 1
        self.steps_since_food += 1

        if (nx, ny) == self.food:
            self.score += 1
            self.steps_since_food = 0
            self.food = self._spawn_food()
        else:
            self.body.pop()

        # starvation
        limit = config.MAX_STEPS + self.score * 50
        if self.steps_since_food > limit:
            self.alive = False

    def compute_fitness(self):
        if self.score == 0:
            self.fitness = self.steps * 0.1  # small reward for surviving at all
        else:
            self.fitness = self.score * 1000 + (self.score / self.steps) * 500
        return self.fitness
