# hetero_robot_fleet_sim.py
"""
2‑D heterogeneous robot fleet exploration simulator with simple GUI.

Robots: Legged, Drone, Boat, Rover
Environment: grid cells with semantic terrain classes and occupancy.
Planner: baseline A* (swap in D*‑Lite later).
GUI: pygame grid – each terrain type & robot drawn in colour.

Run:
    pip install pygame
    python hetero_robot_fleet_sim.py
"""

import pygame
from enum import Enum
import heapq
import math
import random
import sys

# ---------- Configuration ----------
CELL_SIZE = 20
GRID_W, GRID_H = 40, 25
FPS = 10  # simulation frames per second
# -----------------------------------

class Terrain(Enum):
    UNKNOWN = 0
    FREE = 1
    OBSTACLE = 2
    STAIRS = 3
    WATER = 4

class Capability(Enum):
    LAND = 1
    STAIRS = 2
    WATER = 3
    AIR = 4

# Colour palette (RGB tuples)
TERRAIN_COLOUR = {
    Terrain.UNKNOWN: (200, 200, 200),   # light grey
    Terrain.FREE:    (255, 255, 255),   # white
    Terrain.OBSTACLE:(0, 0, 0),         # black
    Terrain.STAIRS:  (255, 255, 0),     # yellow
    Terrain.WATER:   (0, 0, 255),       # blue
}
ROBOT_COLOUR = {
    "Legged": (0, 255, 0),      # green
    "Drone":  (255, 0, 255),    # magenta
    "Boat":   (0, 255, 255),    # cyan
    "Rover":  (255, 165, 0),    # orange
}

class Cell:
    """Grid cell with terrain semantics and log‑odds occupancy (OctoMap‑style)."""
    def __init__(self):
        self.terrain: Terrain = Terrain.UNKNOWN
        self.log_odds: float = 0.0   # not yet used for sensing

    # Traversal cost from a robot with a given capability set
    def cost(self, caps: set) -> float:
        if self.terrain == Terrain.OBSTACLE:
            return math.inf
        if self.terrain == Terrain.STAIRS and Capability.STAIRS not in caps and Capability.AIR not in caps:
            return math.inf
        if self.terrain == Terrain.WATER and Capability.WATER not in caps and Capability.AIR not in caps:
            return math.inf
        return 1.0

class GridWorld:
    """2‑D grid map with helper utilities."""
    def __init__(self, w: int, h: int):
        self.w, self.h = w, h
        self.grid: list[list[Cell]] = [[Cell() for _ in range(h)] for _ in range(w)]
        self._generate_demo_world()

    def _generate_demo_world(self):
        # Random static obstacles
        for x in range(self.w):
            for y in range(self.h):
                if random.random() < 0.1:
                    self.grid[x][y].terrain = Terrain.OBSTACLE
        # Staircase band
        for y in range(5, 10):
            self.grid[20][y].terrain = Terrain.STAIRS
        # Water pool
        for x in range(5, 10):
            for y in range(15, 20):
                self.grid[x][y].terrain = Terrain.WATER
        # Mark everything else free
        for x in range(self.w):
            for y in range(self.h):
                if self.grid[x][y].terrain == Terrain.UNKNOWN:
                    self.grid[x][y].terrain = Terrain.FREE

    def neighbours(self, node: tuple[int, int]):
        x, y = node
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.w and 0 <= ny < self.h:
                yield nx, ny

# ---------- Path planning: vanilla A* ----------

def heuristic(a: tuple[int, int], b: tuple[int, int]) -> float:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def reconstruct(came: dict, cur: tuple[int,int]):
    path = [cur]
    while cur in came:
        cur = came[cur]
        path.append(cur)
    path.reverse()
    return path

def astar(start: tuple[int,int], goal: tuple[int,int], world: GridWorld, caps: set):
    open_set: list[tuple[float, tuple[int,int]]] = []
    g = {start: 0.0}
    f = {start: heuristic(start, goal)}
    heapq.heappush(open_set, (f[start], start))
    came: dict = {}
    while open_set:
        _, cur = heapq.heappop(open_set)
        if cur == goal:
            return reconstruct(came, cur)
        for nbr in world.neighbours(cur):
            tentative = g[cur] + world.grid[nbr[0]][nbr[1]].cost(caps)
            if tentative < g.get(nbr, math.inf):
                came[nbr] = cur
                g[nbr] = tentative
                f[nbr] = tentative + heuristic(nbr, goal)
                heapq.heappush(open_set, (f[nbr], nbr))
    return []  # no path found
# -----------------------------------------------

class Robot:
    """Base class: holds position and capability set."""
    def __init__(self, name: str, x: int, y: int, caps: set[Capability]):
        self.name = name
        self.pos = (x, y)
        self.caps = caps

    def step(self, world: GridWorld, target: tuple[int,int]):
        if target == self.pos:
            return
        path = astar(self.pos, target, world, self.caps)
        if len(path) > 1:
            self.pos = path[1]  # move one step

class FleetSim:
    """Controller that spawns robots and steps the whole simulation."""
    def __init__(self):
        self.world = GridWorld(GRID_W, GRID_H)
        self.robots: list[Robot] = [
            Robot("Legged", 1, 1, {Capability.LAND, Capability.STAIRS}),
            Robot("Drone",  GRID_W - 2, 1, {Capability.AIR}),
            Robot("Boat",   1, GRID_H - 2, {Capability.WATER}),
            Robot("Rover",  GRID_W - 2, GRID_H - 2, {Capability.LAND}),
        ]

    # --- exploration policy ---
    def _unknown_cells(self, robot: Robot):
        """Return grid coords that are still unknown but traversable by this robot."""
        cells = []
        for x in range(self.world.w):
            for y in range(self.world.h):
                cell = self.world.grid[x][y]
                if cell.terrain == Terrain.UNKNOWN and cell.cost(robot.caps) < math.inf:
                    cells.append((x, y))
        return cells

    def pick_target(self, robot: Robot):
        cells = self._unknown_cells(robot)
        if not cells:
            return robot.pos
        # pick the closest unknown cell (frontier)
        return min(cells, key=lambda c: heuristic(robot.pos, c))

    def step(self):
        for robot in self.robots:
            tgt = self.pick_target(robot)
            robot.step(self.world, tgt)

# ---------- GUI using pygame ----------

def draw_grid(screen, world: GridWorld):
    for x in range(world.w):
        for y in range(world.h):
            terrain = world.grid[x][y].terrain
            colour = TERRAIN_COLOUR[terrain]
            rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, colour, rect)


def draw_robots(screen, robots: list[Robot]):
    for robot in robots:
        x, y = robot.pos
        colour = ROBOT_COLOUR[robot.name]
        rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(screen, colour, rect)


def gui_loop():
    pygame.init()
    screen = pygame.display.set_mode((GRID_W * CELL_SIZE, GRID_H * CELL_SIZE))
    pygame.display.set_caption("Heterogeneous Robot Fleet Simulator")
    clock = pygame.time.Clock()
    sim = FleetSim()

    running = True
    while running:
        # --- Event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # --- Simulation step ---
        sim.step()

        # --- Draw ---
        draw_grid(screen, sim.world)
        draw_robots(screen, sim.robots)
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()
# --------------------------------------

if __name__ == "__main__":
    gui_loop()
