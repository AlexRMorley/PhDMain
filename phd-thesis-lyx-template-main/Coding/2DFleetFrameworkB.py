# hetero_robot_fleet_sim.py
"""
2-D heterogeneous robot fleet exploration simulator with GUI controls.

Robots: Legged, Drone, Boat, Rover
Environment: grid cells with semantic terrain (STAIRS band & WATER pool & RIVER; rest FREE/OBSTACLE).
Planner: A* on each robot’s current known map, augmented with chunked risk.
GUI: pygame grid + faint cell lines + sidebar with clickable Start, Show Map, Show Survivors, Show Heat, Show Rad buttons + robot key + coverage % + time step + checklist of survivors.
Robots reveal a 3×3 area around themselves when they move (and at start), and only plan over known chunks.
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
from collections import deque
import numpy as np

# ---------- Configuration ----------
CELL_SIZE    = 10
GRID_W,GRID_H= 64,64
FPS          = 10
SIDEBAR_WIDTH= 200
MAX_BATTERY  = 1000
TEMP_LIMIT   = 80      # robots only break down at very high spots
RAD_LIMIT    = 80

# Framework risk aggregation
ZONE_CHUNKS= 4 #Zones for auction
ZONE_TARGET = 0.98
CHUNK_SIZE = 2    # 4×4 chunks
ALPHA      = 60    # global risk‐aversion multiplier
BETA = 0.5
P = 4
# -----------------------------------

class Terrain(Enum):
    UNKNOWN=0; FREE=1; OBSTACLE=2; STAIRS=3; WATER=4

class Capability(Enum):
    LAND=1; STAIRS=2; WATER=3; AIR=4

TERRAIN_COLOUR = {
    Terrain.UNKNOWN:  (200,200,200),
    Terrain.FREE:     (255,255,255),
    Terrain.OBSTACLE: (  0,  0,  0),
    Terrain.STAIRS:   (255,255,  0),
    Terrain.WATER:    (  0,  0,255),
}
ROBOT_COLOUR = {
    "Legged": (  0,255,  0),
    "Drone":  (255,  0,255),
    "Boat":   (  0,255,255),
    "Rover":  (255,165,  0),
}
SURVIVOR_COLOUR = (255,  0,  0)

from dataclasses import dataclass

@dataclass
class ZoneTask:
    zone: tuple           # (zx, zy)
    owner: str | None     # robot.name
    status: str           # "free" | "held" | "released" | "blacklisted"
    progress: float       # coverage in [0,1]
    expires_at: int       # timestep when lease ends

    last_owner: str | None = None
    last_release_reason: str = ""


# ---------- Grid & Cells ----------
class Cell:
    def __init__(self, true_terrain=None):
        self.true_terrain = true_terrain or Terrain.FREE
        self.terrain      = Terrain.UNKNOWN
        self.temperature  = 0.0
        self.radiation    = 0.0

    def cost(self, caps:set, terrain_override=None) -> float:
        t = terrain_override if terrain_override is not None else self.true_terrain

        if t is Terrain.UNKNOWN:
            # Unknown prior: allow movement but slightly penalize
            # (so they prefer known-free space)
            return 1.5
        if t is Terrain.OBSTACLE:
            return math.inf
        if t is Terrain.STAIRS and Capability.STAIRS not in caps and Capability.AIR not in caps:
            return math.inf
        if t is Terrain.WATER and Capability.WATER not in caps and Capability.AIR not in caps:
            return math.inf
        if t is Terrain.FREE and Capability.LAND not in caps and Capability.AIR not in caps:
            return math.inf
        return 1.0

class GridWorld:
    def __init__(self, w, h):
        self.w, self.h = w, h
        self.grid = [[Cell() for _ in range(h)] for _ in range(w)]
        self._generate_demo_world()
        self._initialize_temperature()
        self._initialize_radiation()

    def _generate_demo_world(self):
        for x in range(self.w):
            for y in range(self.h):
                if random.random() < 0.05:
                    self.grid[x][y] = Cell(Terrain.OBSTACLE)
        # Cove
        for x in range(5,10):
            for y in range(15,20):
                self.grid[x][y] = Cell(Terrain.WATER)
        for x in range(4,11):
            self.grid[x][9] = Cell(Terrain.OBSTACLE)
        for y in range(9,21):
            self.grid[4][y] = Cell(Terrain.OBSTACLE)
            self.grid[10][y] = Cell(Terrain.OBSTACLE)
        # River
        for y in range(20,25):
            for x in range(self.w-4):
                self.grid[x][y] = Cell(Terrain.WATER)
        for x in range(0,5):
            for y in range(20,self.h):
                self.grid[x][y] = Cell(Terrain.WATER)
        # Pool
        for x in range(35,44):
            for y in range(7,11):
                self.grid[x][y] = Cell(Terrain.WATER)
        # Small Building
        for x in range(21,27):
            self.grid[x][7] = Cell(Terrain.OBSTACLE)
            self.grid[x][12]= Cell(Terrain.OBSTACLE)
        for y in range(7,14):
            self.grid[21][y] = Cell(Terrain.OBSTACLE)
            self.grid[26][y] = Cell(Terrain.OBSTACLE)
        for y in range(9,12):
            self.grid[26][y] = Cell(Terrain.FREE)
        # Bridge
        for x in range(50,53):
            for y in range(19,26):
                self.grid[x][y] = Cell(Terrain.STAIRS)
        # Down River
        for x in range(25,31):
            for y in range(25,44):
                self.grid[x][y] = Cell(Terrain.WATER)
        for x in range(12,26):
            for y in range(40,44):
                self.grid[x][y] = Cell(Terrain.WATER)
        # Large Building
        for x in range(35,46):
            self.grid[x][40] = Cell(Terrain.OBSTACLE)
            self.grid[x][55] = Cell(Terrain.OBSTACLE)
            self.grid[x][48] = Cell(Terrain.OBSTACLE)
        for x in range(38,40):
            self.grid[x][55] = Cell(Terrain.FREE)
        for x in range(41,43):
            self.grid[x][48] = Cell(Terrain.STAIRS)
        for x in range(36,45):
            for y in range(41,48):
                self.grid[x][y] = Cell(Terrain.STAIRS)
        for y in range(40,55):
            self.grid[35][y] = Cell(Terrain.OBSTACLE)
            self.grid[45][y] = Cell(Terrain.OBSTACLE)
        for y in range(50,52):
            self.grid[45][y] = Cell(Terrain.FREE)
        # Rest free
        for x in range(self.w):
            for y in range(self.h):
                t = self.grid[x][y].true_terrain
                if t not in (Terrain.OBSTACLE, Terrain.STAIRS, Terrain.WATER):
                    self.grid[x][y] = Cell(Terrain.FREE)

    def _initialize_temperature(self):
        sources = [
            ((20, 31), 4, 100),
            ((45, 15), 3, 100),
            ((15, 55), 4, 100),
            ((50, 55), 4, 100),
            ((30, 40), 3, 100),
            ((60, 35), 3, 100),
            ((10, 35), 3, 100),
            ((14, 15), 3, 100),
            ((58, 54), 3, 100),
        ]
        for x in range(self.w):
            for y in range(self.h):
                temp = 0.0
                for (mx, my), sigma, amp in sources: #amplitude from sources
                    dx, dy = x-mx, y-my
                    d2 = (dx*dx + dy*dy)/(2*sigma*sigma)
                    temp += amp * math.exp(-(d2))
                if self.grid[x][y].true_terrain == Terrain.WATER:
                    temp = 5.0
                self.grid[x][y].temperature = temp

    def _initialize_radiation(self):
        rad_sources = [
            ((20, 31), 4, 100),
            ((40, 50), 4, 100),
            ((10, 10), 5, 100),
            ((50, 10), 4, 100),
            ((30, 40), 4, 100),
            ((31, 20), 3, 100),
            ((10, 55), 4, 100),
            ((60, 60), 3, 100),
            ((40,  3), 4, 100),
        ]
        for x in range(self.w):
            for y in range(self.h):
                rad = 0.0
                for (mx, my), sigma, amp in rad_sources:
                    dx, dy = x-mx, y-my
                    d2 = (dx*dx + dy*dy)/(2*sigma*sigma)
                    rad += amp * math.exp(-d2)
                if self.grid[x][y].true_terrain == Terrain.WATER:
                    rad = 0.0
                self.grid[x][y].radiation = rad

    def neighbours(self, u):
        x,y = u
        for dx,dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx,ny = x+dx, y+dy
            if 0 <= nx < self.w and 0 <= ny < self.h:
                yield (nx, ny)

    def cost(self, u, v, caps:set) -> float:
        cell = self.grid[v[0]][v[1]]


        return cell.cost(caps)

# ---------- A* Planner ----------
def heuristic(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

class AStar:
    def __init__(self, world, start, goal, caps,
                 chunked_risk: np.ndarray,
                 weights: np.ndarray,
                 alpha: float,
                 temp_limit: float,
                 rad_limit: float,
                 terrain_belief: np.ndarray,
                 temp_belief: np.ndarray,
                 rad_belief: np.ndarray):
        self.world   = world
        self.start   = start
        self.goal    = goal
        self.caps    = caps
        self.chunked = chunked_risk
        self.weights = weights
        self.alpha   = alpha
        self.temp_limit = temp_limit
        self.rad_limit  = rad_limit

        self.terrain_belief = terrain_belief
        self.temp_belief    = temp_belief
        self.rad_belief     = rad_belief


    def search(self):
        open_set  = [(0, self.start)]
        came_from = {}
        g_score   = {self.start: 0.0}
        f_score   = {self.start: heuristic(self.start, self.goal)}

        def norm_exposure(temp, rad, temp_limit, rad_limit):
            eT = temp / max(1e-6, temp_limit)
            eR = rad  / max(1e-6, rad_limit)
            return eT, eR

        while open_set:
            _, current = heapq.heappop(open_set)
            if current == self.goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                return path[::-1]

            for nbr in self.world.neighbours(current):
                tb = self.terrain_belief[nbr[0], nbr[1]]

                # base feasibility from BELIEF terrain
                base = self.world.grid[nbr[0]][nbr[1]].cost(self.caps, terrain_override=tb)
                if math.isinf(base):
                    continue

                # ---- conservative UNKNOWN hazard rule for fragile robots ----
                if tb is Terrain.UNKNOWN and (self.temp_limit < 9000.0 or self.rad_limit < 9000.0):
                    # if any adjacent KNOWN cell is already near-soft-limit, don't step into unknown here
                    soft_t = 0.85 * self.temp_limit
                    soft_r = 0.85 * self.rad_limit
                    danger_nearby = False
                    for nn in self.world.neighbours(nbr):
                        tb2 = self.terrain_belief[nn[0], nn[1]]
                        if tb2 is Terrain.UNKNOWN:
                            continue
                        t2 = self.temp_belief[nn[0], nn[1]]
                        r2 = self.rad_belief[nn[0], nn[1]]
                        if (not np.isnan(t2) and t2 > soft_t) or (not np.isnan(r2) and r2 > soft_r):
                            danger_nearby = True
                            break
                    if danger_nearby:
                        continue

                # lethal avoidance ONLY if we know the cell hazards
                if tb is not Terrain.UNKNOWN:
                    t = self.temp_belief[nbr[0], nbr[1]]
                    r = self.rad_belief[nbr[0], nbr[1]]
                    if (not np.isnan(t) and t > self.temp_limit) or (not np.isnan(r) and r > self.rad_limit):
                        continue

                # terrain bonuses (again based on BELIEF)
                if tb is not Terrain.UNKNOWN:
                    if self.caps == {Capability.AIR} and tb is Terrain.WATER:
                        base *= 0.5
                    elif Capability.STAIRS in self.caps and tb is Terrain.STAIRS:
                        base *= 0.2

                # chunk penalty (already belief-masked in recompute_chunked)
                cx, cy = nbr[0] // CHUNK_SIZE, nbr[1] // CHUNK_SIZE
                λ_T = self.chunked[0, cx, cy]
                λ_R = self.chunked[1, cx, cy]
                # chunk penalty: use chunk maxima (λ_T, λ_R) but normalized by THIS robot limits
                eT_c = λ_T / max(1e-6, self.temp_limit)
                eR_c = λ_R / max(1e-6, self.rad_limit)
                e_c  = max(eT_c, eR_c)

                # per-cell penalty: normalized too (only if known)
                e = 0.0
                if tb is not Terrain.UNKNOWN:
                    t = 0.0 if np.isnan(self.temp_belief[nbr[0], nbr[1]]) else float(self.temp_belief[nbr[0], nbr[1]])
                    r = 0.0 if np.isnan(self.rad_belief[nbr[0], nbr[1]])  else float(self.rad_belief[nbr[0], nbr[1]])
                    eT, eR = norm_exposure(t, r, self.temp_limit, self.rad_limit)
                    e = max(eT, eR)

                # shape: very mild when safe, steep near limit
                P = 4  # try 3–6
                chunk_pen = self.alpha * (e_c ** P)
                cell_pen  = BETA * (e ** P)

                step_cost = max(0.01, base + chunk_pen + cell_pen)
                tentative_g = g_score[current] + step_cost

                if tentative_g < g_score.get(nbr, math.inf):
                    came_from[nbr] = current
                    g_score[nbr]   = tentative_g
                    f = tentative_g + heuristic(nbr, self.goal)
                    heapq.heappush(open_set, (f, nbr))

        return []
    



# ---------- Robot (single class) ----------
class Robot:
    def __init__(self, name, x, y, caps, world,
                 raw_risk: np.ndarray,
                 weights: np.ndarray,
                 alpha: float,
                 temp_limit: float,
                 rad_limit: float):
        self.name         = name
        self.pos          = (x,y)
        self.caps         = caps
        self.world        = world
        self.raw          = raw_risk     # shape=(2,W,H)
        self.weights      = weights
        self.alpha        = alpha
        # mask of explored cells
        self.known_mask   = np.zeros((GRID_W, GRID_H), bool)
        # chunked risk over known mask
        self.chunked      = np.zeros((2,
                                      GRID_W//CHUNK_SIZE,
                                      GRID_H//CHUNK_SIZE), float)
        self.goal, self.path = None, []
        self.active       = True
        self.battery      = MAX_BATTERY
        self.death_reason = None
        self.goal_commit = 0          # ticks remaining before we allow goal switching
        self.last_pos = self.pos 
        self.temp_limit = temp_limit
        self.rad_limit  = rad_limit
        self.soft_temp = 0.85 * self.temp_limit
        self.soft_rad  = 0.85 * self.rad_limit
        self.bundle = []
        self.assigned_zones = []
        self.failed_goals = {}   # {(x,y): retry_after_timestep}
        self.role = "SCAN"
        self.dose_T = 0.0
        self.dose_R = 0.0
        self.role_locked_until = 0

        
        #Stability
        self.stuck_steps    = 0
        self.task_zone = None            # zone currently being worked on
        self.zone_lease_until = 0        # timestep until which zone is locked

        # Progress & failure tracking
        self.task_no_progress = 0
        self.task_last_known = 0

        # Zone cooldown / blacklist (prevents flip-flopping)
        self.blacklist = {}              # {(zx,zy): cooldown_until_t}

        self.current_zone     = None
        self.zone_start_batt  = None
        self.terrain_belief = np.full((GRID_W, GRID_H), Terrain.UNKNOWN, dtype=object)
        self.temp_belief    = np.full((GRID_W, GRID_H), np.nan, dtype=float)
        self.rad_belief     = np.full((GRID_W, GRID_H), np.nan, dtype=float)
        self.cached_frontiers = []
        self.frontier_refresh = 0



        


        # initial reveal & chunked build
        self.reveal()
        self.recompute_chunked()

    def reveal(self):
        R = 2                  # how many cells out you can see
        x0,y0 = self.pos
        for dx in range(-R, R+1):
            for dy in range(-R, R+1):
                if dx*dx + dy*dy <= R*R:
                    nx,ny = x0+dx, y0+dy
                    if 0 <= nx < self.world.w and 0 <= ny < self.world.h:
                        # mark that cell as seen
                        cell_true = self.world.grid[nx][ny]
                        self.terrain_belief[nx, ny] = cell_true.true_terrain
                        self.temp_belief[nx, ny]    = cell_true.temperature
                        self.rad_belief[nx, ny]     = cell_true.radiation
                        self.known_mask[nx, ny]     = True
        # x0,y0=self.pos
        # for dx in (-1,0,1):
        #     for dy in (-1,0,1):
        #         nx,ny=x0+dx,y0+dy
        #         if 0<=nx<self.world.w and 0<=ny<self.world.h:
        #                 # mark that cell as seen
        #                 self.world.grid[nx][ny].terrain = \
        #                      self.world.grid[nx][ny].true_terrain
        #                 self.known_mask[nx, ny] = True

    def recompute_chunked(self):
        maskedT = np.zeros((GRID_W, GRID_H), float)
        maskedR = np.zeros((GRID_W, GRID_H), float)

        known = self.known_mask
        maskedT[known] = np.nan_to_num(self.temp_belief[known], nan=0.0)
        maskedR[known] = np.nan_to_num(self.rad_belief[known],  nan=0.0)

        newW = GRID_W // CHUNK_SIZE
        newH = GRID_H // CHUNK_SIZE

        self.chunked[0] = maskedT.reshape(newW, CHUNK_SIZE, newH, CHUNK_SIZE).max(axis=(1,3))
        self.chunked[1] = maskedR.reshape(newW, CHUNK_SIZE, newH, CHUNK_SIZE).max(axis=(1,3))


    def set_goal(self, tgt):
        self.goal = tgt

        path = AStar(
            self.world,
            self.pos, tgt,
            self.caps,
            self.chunked,
            self.weights,
            self.alpha,
            self.temp_limit,
            self.rad_limit,
            self.terrain_belief,
            self.temp_belief,
            self.rad_belief
        ).search()
        
        if not path:
            self.failed_goals[tgt] = sim.timestep + 50
            self.goal = None
            self.path = []
            self.goal_commit = 0
            return

        
        risk, has_lethal = self.path_risk_score(path, K=12)
        if has_lethal:
            self.failed_goals[tgt] = sim.timestep + 80
            self.goal = None
            self.path = []
            self.goal_commit = 0
            return

        self.path = path

        # Soft commitment: only lock goal for a short time
        self.goal_commit = 20   # tune: 10-30

        from_zone = sim.cell_to_zone(tgt[0], tgt[1])
        self.current_zone = from_zone
        self.zone_start_batt = self.battery
        self.stuck_steps = 0
        
        from_zone = sim.cell_to_zone(tgt[0], tgt[1])
        self.current_zone = from_zone
        self.zone_start_batt = self.battery
        self.stuck_steps = 0


    def step(self):
        if self.battery <= 0:
            self.active = False
            self.death_reason = "battery depleted"
            self.path = []
            self.goal = None
            return

        if self.goal is None:
            self.path = []
            return

        # Recompute path at every step based on latest map & risk
        self.path = AStar(
            self.world,
            self.pos, self.goal,
            self.caps,
            self.chunked,
            self.weights,
            self.alpha,
            self.temp_limit,
            self.rad_limit,
            self.terrain_belief,
            self.temp_belief,
            self.rad_belief
        ).search()

        if not self.path or self.pos == self.goal or (self.stuck_steps > 0 and self.stuck_steps % 5 == 0):
            self.path = AStar(
                self.world,
                self.pos,
                self.goal,
                self.caps,
                self.chunked,
                self.weights,
                self.alpha,
                self.temp_limit,
                self.rad_limit,
                self.terrain_belief,
                self.temp_belief,
                self.rad_belief
            ).search()

        if not self.path:
            self.goal = None
            self.goal_commit = 0
            return
        
        # Emergency abort if next step is predicted lethal based on belief
        next_step = self.path[0]
        if self.predicted_cell_lethal(next_step):
            self.goal = None
            self.path = []
            self.goal_commit = 0
            return

        self.pos = self.path.pop(0)
        self.reveal()
        self.recompute_chunked()

        # Battery drain
        if self.name == "Drone":
            self.battery -= 2
        elif self.name == "Legged":
            self.battery -= 1
        elif self.name == "Boat":
            self.battery -= 2
        else:
            self.battery -= 0.5

        # Check lethal exposure at new position
        c = self.world.grid[self.pos[0]][self.pos[1]]
        self.dose_R += max(0.0, c.temperature) * 0.01
        self.dose_T += max(0.0,c.radiation) * 0.01 
        over_t = c.temperature > self.temp_limit
        over_r = c.radiation  > self.rad_limit



        if over_t or over_r:
            self.active = False
            self.path = []
            self.goal = None
            reasons = []
            if over_t: reasons.append("high temperature")
            if over_r: reasons.append("high radiation")
            self.death_reason = " & ".join(reasons)
        
        if self.goal_commit > 0:
            self.goal_commit -= 1

                
    def predicted_cell_lethal(self, cell_xy):
            """Use BELIEF hazards to decide if a cell is lethal (only if known)."""
            x, y = cell_xy
            tb = self.terrain_belief[x, y]
            if tb is Terrain.UNKNOWN:
                return False  # unknown hazard -> don't call it lethal
            t = self.temp_belief[x, y]
            r = self.rad_belief[x, y]
            if (not np.isnan(t) and t > self.temp_limit) or (not np.isnan(r) and r > self.rad_limit):
                return True
            return False

    def path_risk_score(self, path, K=12):
        """
        Look ahead K steps and compute a risk score from BELIEF hazards.
        Returns (risk_score, has_lethal).
        """
        risk = 0.0
        has_lethal = False
        for p in path[:K]:
            x, y = p
            tb = self.terrain_belief[x, y]
            if tb is Terrain.UNKNOWN:
                continue
            t = self.temp_belief[x, y]
            r = self.rad_belief[x, y]
            if (not np.isnan(t) and t > self.temp_limit) or (not np.isnan(r) and r > self.rad_limit):
                has_lethal = True
                break
            # soft penalty if close to limits
            if not np.isnan(t) and t > self.soft_temp:
                risk += (t - self.soft_temp) / max(1e-6, (self.temp_limit - self.soft_temp))
            if not np.isnan(r) and r > self.soft_rad:
                risk += (r - self.soft_rad) / max(1e-6, (self.rad_limit - self.soft_rad))
        return risk, has_lethal


# ---------- Simulation Controller ----------
class FleetSim:
    def __init__(self):
        self.world = GridWorld(GRID_W, GRID_H)

        #Zones
        self.zone_w_cells = ZONE_CHUNKS * CHUNK_SIZE
        self.zone_h_cells = ZONE_CHUNKS * CHUNK_SIZE
        self.zone_nx = GRID_W // self.zone_w_cells
        self.zone_ny = GRID_H // self.zone_h_cells

        self.found       = set()
        self.dead_robots = []

        self.debug_zone_bids = {}   # robot_name -> list of dict rows (sorted later)
        self.show_lambda_debug = False

        self.zone_tasks = {(zx,zy): ZoneTask((zx,zy), None, "free", 0.0, 0)
                   for zx in range(self.zone_nx) for zy in range(self.zone_ny)}
        
        self.timestep = 0

        self.LEASE_T = 40          # lock zone for N ticks (stops flip-flopping)
        self.COOLDOWN_T = 80       # blacklist zone for N ticks after failure
        self.ZONE_DONE = 0.90      # completion threshold (global)
        self.NO_PROGRESS_K = 25    # if no progress for K ticks => drop zone

        self.MAX_BUNDLE = 4
        self.CBBA_ITERS = 3     # 2–5 is fine
        self.REPLAN_T = 50      # already using %50

        self.debug_zone_winners = {}    # zone -> {winner, u, losers:[(robot,u)], reason}
        self.debug_cbba_rounds = []     # list of round summaries (optional)



        # build raw risk arrays
        W, H = GRID_W, GRID_H
        raw = np.zeros((2, W, H), float)
        for x in range(W):
            for y in range(H):
                raw[0,x,y] = self.world.grid[x][y].temperature
                raw[1,x,y] = self.world.grid[x][y].radiation

        # per-robot weights (Rover attracted → negative)
        weight_map = {
            "Legged": np.array([10.0, 10.0]),   # strong avoidance
            "Drone":  np.array([10.0, 10.0]),   # very strong avoidance
            "Boat":   np.array([ 0.0,  0.0]),   # no bias
            "Rover":  np.array([-2.0,-2.0]),   # seeks hotspots
        }

        starts = [(1,1), (GRID_W-2,1), (1,GRID_H-2), (GRID_W-2,GRID_H-2)]
        names  = ["Legged","Drone","Boat","Rover"]
        caps_l = [
            {Capability.LAND,Capability.STAIRS},
            {Capability.AIR},
            {Capability.WATER},
            {Capability.LAND},
        ]
        limit_map = {
            "Legged": (TEMP_LIMIT, RAD_LIMIT),         # strict
            "Drone":  (TEMP_LIMIT, RAD_LIMIT),         # strict
            "Boat":   (9999.0, 9999.0),                # ignores heat/rad
            "Rover":  (9999.0, 9999.0),                # very tolerant (or set custom)
        }

        self.robots = []
        for i,name in enumerate(names):
            temp_lim, rad_lim = limit_map[name]
            self.robots.append(Robot(
                name, *starts[i],
                caps_l[i],
                self.world,
                raw,
                weight_map[name],
                ALPHA,
                temp_limit=temp_lim,
                rad_limit=rad_lim
            ))

        free_cells = [
            (x,y) for x in range(self.world.w)
                   for y in range(self.world.h)
                   if self.world.grid[x][y].true_terrain == Terrain.FREE
        ]
        

        #self.survivors = random.sample(free_cells, 3)
        self.survivors = [
            (7, 12),
            (37, 41),
            (20, 30),
        ]
        self.found       = set()
        self.dead_robots = []
        
        self.assign_zones_cbba()

    def get_union_terrain_belief(self):
            union = np.full((GRID_W, GRID_H), Terrain.UNKNOWN, dtype=object)
            for r in self.robots:
                known = r.known_mask
                union[known] = r.terrain_belief[known]
            return union
    
    def get_union_temp_belief(self):
        T = np.full((GRID_W, GRID_H), np.nan, dtype=float)
        for r in self.robots:
            known = r.known_mask
            T[known] = r.temp_belief[known]
        return T
    
    def get_union_rad_belief(self):
        R = np.full((GRID_W, GRID_H), np.nan, dtype=float)
        for r in self.robots:
            known = r.known_mask
            R[known] = r.rad_belief[known]
        return R
    
    def cell_to_zone(self, x, y):
        zx = x // self.zone_w_cells
        zy = y // self.zone_h_cells
        if 0 <= zx < self.zone_nx and 0 <= zy < self.zone_ny:
            return (zx, zy)
        else:
            return None

    def zone_cells(self, zone):
        zx, zy = zone
        x0 = zx * self.zone_w_cells
        y0 = zy * self.zone_h_cells
        xs = range(x0, min(x0 + self.zone_w_cells, GRID_W))
        ys = range(y0, min(y0 + self.zone_h_cells, GRID_H))
        return xs, ys

    def zone_coverage(self, union, zone):
        """Coverage fraction in [0,1] using the shared union belief."""
        xs, ys = self.zone_cells(zone)
        total = 0
        known = 0
        for x in xs:
            for y in ys:
                total += 1
                if union[x, y] != Terrain.UNKNOWN:
                    known += 1
        return (known / total) if total > 0 else 1.0

    def zone_frontiers(self, union, reachable, zone):
        """Unknown reachable cells inside a zone."""
        xs, ys = self.zone_cells(zone)
        out = []
        for x in xs:
            for y in ys:
                if (x, y) in reachable and union[x, y] == Terrain.UNKNOWN:
                    out.append((x, y))
        return out


    def assign_zones_cbba(self):
        union_belief = self.get_union_terrain_belief()
        union_T = self.get_union_temp_belief()
        union_R = self.get_union_rad_belief()

        self.refresh_zone_tasks(union_belief)

        zones = [(zx, zy) for zx in range(self.zone_nx) for zy in range(self.zone_ny)]

        # reset robot intents
        for r in self.robots:
            r.bundle = []
            r.assigned_zones = []

        # reset debug
        self.debug_zone_bids = {r.name: [] for r in self.robots}
        self.debug_zone_winners = {}
        self.debug_cbba_rounds = []

        # counts used inside lambda_breakdown
        zones_assigned_count = {r.name: 0 for r in self.robots}

        # --- CBBA rounds ---
        for it in range(self.CBBA_ITERS):
            round_summary = {"iter": it, "conflicts": 0, "wins": {}}

            # 1) Build / refresh FULL bid table for each robot (debug layer 1)
            #    AND greedily add to bundle
            for r in self.robots:
                if (not r.active) or (r.battery <= 0):
                    continue

                # recompute counts from current bundles (so "load" term is meaningful)
                zones_assigned_count = {rr.name: len(rr.bundle) for rr in self.robots}

                # full table (all zones) just for debug/top-K
                robot_rows = []
                robot_bids = []  # feasible candidates for bundle selection only: (u, zone)

                for z in zones:
                    bid, row = self.zone_bid_or_skip(r, z, union_belief, union_T, union_R, zones_assigned_count)
                    robot_rows.append(row)
                    if bid is not None:
                        u, bd, _ = bid
                        robot_bids.append((u, z))

                # store debug table (complete, like your original)
                robot_rows.sort(key=lambda d: d.get("u", -1e9), reverse=True)
                self.debug_zone_bids[r.name] = robot_rows

                # greedily fill bundle (max 4)
                while len(r.bundle) < self.MAX_BUNDLE and robot_bids:
                    robot_bids.sort(reverse=True, key=lambda t: t[0])
                    best_u, best_z = robot_bids.pop(0)
                    if best_u <= -0.5:
                        break
                    if best_z not in r.bundle:
                        r.bundle.append(best_z)

            # 2) Consensus: for each zone, pick winner by max bid among robots that included it
            zone_claims = {}  # zone -> list[(robot_name, u)]
            for r in self.robots:
                zones_assigned_count = {rr.name: len(rr.bundle) for rr in self.robots}
                for z in r.bundle:
                    bid, _row = self.zone_bid_or_skip(r, z, union_belief, union_T, union_R, zones_assigned_count)
                    if bid is None:
                        continue
                    u, bd, _ = bid
                    zone_claims.setdefault(z, []).append((r.name, u))

            # decide winners + drop losers
            for z, claims in zone_claims.items():
                if len(claims) > 1:
                    round_summary["conflicts"] += 1

                claims_sorted = sorted(claims, key=lambda t: t[1], reverse=True)
                winner_name, winner_u = claims_sorted[0]
                losers = claims_sorted[1:]

                # record zone winner debug (layer 2)
                self.debug_zone_winners[z] = {
                    "winner": winner_name,
                    "u": winner_u,
                    "losers": losers,
                    "iter": it
                }

                # force losers to drop
                for r in self.robots:
                    if r.name != winner_name and z in r.bundle:
                        r.bundle = [zz for zz in r.bundle if zz != z]

            # save round summary
            for r in self.robots:
                round_summary["wins"][r.name] = len(r.bundle)
            self.debug_cbba_rounds.append(round_summary)
            print("\n[CBBA] Zone winners (conflicts only):")
            for z, d in self.debug_zone_winners.items():
                if len(d["losers"]) == 0:
                    continue
                print(f"  zone={z} winner={d['winner']} u={d['u']:.3f} losers={d['losers']}")


        # 3) Finalize: bundle -> assigned_zones, update zone_tasks to held
        # expire old holds
        for z, task in self.zone_tasks.items():
            if task.status == "held" and self.timestep >= task.expires_at:
                task.owner = None
                task.status = "released"
                task.expires_at = 0

        for r in self.robots:
            r.assigned_zones = list(r.bundle)
            for z in r.assigned_zones:
                t = self.zone_tasks[z]
                t.owner = r.name
                t.status = "held"
                t.expires_at = self.timestep + self.LEASE_T


    def zone_feasible(self, r, stats):
    # stats contains: unknown_frac, f_water, f_stairs, avgT, avgR (belief-based)

        if (not r.active) or (r.battery <= 0):
            return False

        uf = stats["unknown_frac"]
        fw = stats["f_water"]
        fs = stats["f_stairs"]

        if r.name == "Boat":
            # allow if water OR lots unknown (scouting)
            return (fw > 0.05) or (uf > 0.40)

        if r.name == "Legged":
            # avoid water-heavy zones unless stairs/unknown justify
            if fw > 0.30 and fs < 0.05 and uf < 0.60:
                return False
            return True

        if r.name == "Rover":
            # rover can go anywhere it can traverse terrain-wise (you already allow)
            return True

        if r.name == "Drone":
            return True

        return True





    def step(self):
        self.timestep += 1

        if self.timestep % 50 == 0:
            self.assign_zones_cbba()

        union_belief = self.get_union_terrain_belief()
        union_T = self.get_union_temp_belief()
        union_R = self.get_union_rad_belief()
        union = union_belief  # keep your existing variable name if you want
        any_active = False
        self.refresh_zone_tasks(union_belief)


        ZONE_TARGET = 0.80
        LEASE_K = 30
        OUT_OF_ZONE_PENALTY = 50.0

        for r in self.robots:
            if r.battery <= 0:
                # if it just died this tick, clean its task state
                if r.active:
                    r.active = False
                    r.death_reason = "battery depleted"
                    self.dead_robots.append((r.name, r.death_reason))

                r.bundle.clear()
                r.assigned_zones.clear()
                r.blacklist.clear()
                r.task_zone = None
                r.goal = None
                r.path = []
                # DO NOT return here


        for r in self.robots:
            if not r.active:
                continue

            # ---- comms merge ----
            unknown = (r.terrain_belief == Terrain.UNKNOWN)
            r.terrain_belief[unknown] = union[unknown]

            # ---- detect survivors ----
            for s in self.survivors:
                if s not in self.found and abs(r.pos[0] - s[0]) <= 2 and abs(r.pos[1] - s[1]) <= 2:
                    self.found.add(s)

            # ---- reachable set (BFS) using BELIEF ----
            dq = deque([r.pos])
            reachable = {r.pos}
            while dq:
                u = dq.popleft()
                for v in self.world.neighbours(u):
                    if v in reachable:
                        continue
                    tb = r.terrain_belief[v[0], v[1]]
                    if self.world.grid[v[0]][v[1]].cost(r.caps, terrain_override=tb) < math.inf:
                        reachable.add(v)
                        dq.append(v)
                    
            frontiers_in_zone = []
            frontiers = []
            # ---- global frontiers cached/rebuilt (ALWAYS) ----
            if r.frontier_refresh > 0 and r.cached_frontiers:
                frontiers = r.cached_frontiers
                r.frontier_refresh -= 1
            else:
                frontiers = []
                for x in range(self.world.w):
                    for y in range(self.world.h):
                        if (x, y) not in reachable:
                            continue
                        if union[x, y] != Terrain.UNKNOWN:
                            continue
                        if (x, y) == r.pos:
                            continue
                        frontiers.append((x, y))
                r.cached_frontiers = frontiers
                r.frontier_refresh = 10


            if r.task_zone is None:
                best = None
                best_cov = 1e9
                for z in r.assigned_zones:
                    if self.zone_is_blacklisted(r, z):
                        continue

                    zx, zy = z
                    stats = self.compute_zone_stats(zx, zy, union_belief, union_T, union_R)
                    if not self.zone_feasible(r, stats):
                        continue

                    cov = self.zone_coverage(union_belief, z)
                    if cov < best_cov:
                        best_cov = cov
                        best = z

                r.task_zone = best
                r.zone_lease_until = self.timestep + self.LEASE_T if best else 0
                r.task_no_progress = 0
                r.task_last_known = 0

            lease_active = (r.task_zone is not None) and (self.timestep < r.zone_lease_until)

            if r.task_zone is not None:
                zx, zy = r.task_zone
                stats = self.compute_zone_stats(zx, zy, union_belief, union_T, union_R)

                cov = self.zone_coverage(union_belief, r.task_zone)

                if cov >= self.ZONE_DONE:
                    self.release_zone(r, reason="complete")
                elif not self.zone_feasible(r, stats):
                    self.release_zone(r, reason="unsuitable")
                elif (not lease_active) and (r.task_no_progress >= self.NO_PROGRESS_K):
                    self.blacklist_zone(r, r.task_zone, reason="no_progress")
                    
                # ---- frontiers in zone ----
                frontiers_in_zone = []
                if r.task_zone is not None:
                    frontiers_in_zone = self.zone_frontiers(union, reachable, r.task_zone)


            # ---- candidates ALWAYS defined ----
            candidates = frontiers_in_zone if frontiers_in_zone else frontiers

            if not candidates:
                r.goal = None
                r.path = []
                r.stuck_steps = 0
                continue


            def frontier_score(c):
                dist = heuristic(r.pos, c)
                cx, cy = c[0] // CHUNK_SIZE, c[1] // CHUNK_SIZE
                λT = r.chunked[0, cx, cy]
                λR = r.chunked[1, cx, cy]
                eT = λT / max(1e-6, r.temp_limit)
                eR = λR / max(1e-6, r.rad_limit)
                e  = max(eT, eR)


                if r.name == "Rover":
                    score = dist - ALPHA * (e ** P)   # SEEK small, e.g. 2–10
                else:
                    score = dist + ALPHA * (e ** P)

                neigh = list(self.world.neighbours(c))
                neigh_terr = [union[nx, ny] for (nx, ny) in neigh]

                if r.name == "Boat":
                    # prioritize unknown cells that touch known water
                    if Terrain.WATER in neigh_terr:
                        score *= 0.6   # strong pull toward shore exploration

                if r.name == "Legged":
                    # prioritize unknown cells that touch stairs
                    if Terrain.STAIRS in neigh_terr:
                        score *= 0.6   # strong pull toward buildings/bridges

                if r.name == "Drone":
                    # drones are good scouts: prefer unknown near stairs/water too
                    if (Terrain.STAIRS in neigh_terr) or (Terrain.WATER in neigh_terr):
                        score *= 0.8
                return score

            need_new_goal = (r.goal is None) or (r.pos == r.goal)
            failing = (r.stuck_steps > 10) or (r.goal is None)

            # Only change goal when allowed
            should_pick_new_goal = (r.goal is None) or (r.pos == r.goal) or (r.goal_commit == 0 and r.stuck_steps > 10)

            if should_pick_new_goal:
                # ---- avoid retrying recently-failed goals ----
                candidates = [c for c in candidates if self.timestep >= r.failed_goals.get(c, 0)]
                if not candidates:
                    r.goal = None
                    r.path = []
                    r.stuck_steps = 0
                    continue
                # tgt = min(candidates, key=frontier_score)
                # r.set_goal(tgt)
                # r.stuck_steps = 0

                if r.task_zone is not None:
                    tgt = self.choose_zone_goal(r, r.task_zone, union, reachable)

                    if tgt is None:
                        self.release_zone(r, reason="no_frontier")
                        continue

                    if self.zone_coverage(union, r.task_zone) >= self.ZONE_DONE:
                        self.release_zone(r, reason="complete")
                        continue

                    r.set_goal(tgt)   # <-- THIS WAS MISSING IN YOUR CODE
                    r.stuck_steps = 0
                else:
                    if not r.cached_frontiers:
                        r.goal = None
                        r.path = []
                        continue
                    tgt = min(r.cached_frontiers, key=lambda c: heuristic(r.pos, c))
                    r.set_goal(tgt)
                    r.stuck_steps = 0

            old_dist = heuristic(r.pos, r.goal) if r.goal is not None else None
            r.step()
            if r.goal is None:
                r.goal_commit = 0

            if r.active and r.goal is not None and old_dist is not None:
                new_dist = heuristic(r.pos, r.goal)
                r.stuck_steps = r.stuck_steps + 1 if new_dist >= old_dist else 0

        if len(self.found) == len(self.survivors):
            return False  # done (found everyone)
            
        any_alive = any(rr.active and rr.battery > 0 for rr in self.robots)
        return any_alive





    
    def compute_zone_stats(self, zx, zy, union_belief, union_T, union_R):
        x0 = zx * self.zone_w_cells
        y0 = zy * self.zone_h_cells

        total = 0
        unknown = 0
        known = 0

        sumT = 0.0
        sumR = 0.0

        terrain_counts = {
            Terrain.FREE: 0,
            Terrain.WATER: 0,
            Terrain.STAIRS: 0,
            Terrain.OBSTACLE: 0,
        }

        for x in range(x0, min(x0 + self.zone_w_cells, GRID_W)):
            for y in range(y0, min(y0 + self.zone_h_cells, GRID_H)):
                total += 1
                tb = union_belief[x, y]
                if tb == Terrain.UNKNOWN:
                    unknown += 1
                else:
                    known += 1
                    if tb in terrain_counts:
                        terrain_counts[tb] += 1

                    # BELIEF hazards (A3 compliant): only count if known & not nan
                    t = union_T[x, y]
                    r = union_R[x, y]
                    if not np.isnan(t): sumT += float(t)
                    if not np.isnan(r): sumR += float(r)

        unknown_frac = (unknown / total) if total > 0 else 0.0

        if known > 0:
            avgT = sumT / known
            avgR = sumR / known
            f_water  = terrain_counts[Terrain.WATER] / known
            f_stairs = terrain_counts[Terrain.STAIRS] / known
            f_free   = terrain_counts[Terrain.FREE] / known
        else:
            avgT = avgR = 0.0
            f_water = f_stairs = f_free = 0.0

        cx = x0 + self.zone_w_cells // 2
        cy = y0 + self.zone_h_cells // 2

        return {
            "unknown_frac": unknown_frac,
            "avgT": avgT,
            "avgR": avgR,
            "f_water": f_water,
            "f_stairs": f_stairs,
            "f_free": f_free,
            "center": (cx, cy),
            "known": known,
            "total": total
        }
    
    def release_zone(self, r, reason="", update_task_status=True):
        z = r.task_zone
        if z is None:
            return

        if update_task_status and z in self.zone_tasks:
            t = self.zone_tasks[z]
            t.last_owner = t.owner
            t.last_release_reason = reason
            t.owner = None
            t.status = "released"
            t.expires_at = 0

        if z in r.assigned_zones:
            r.assigned_zones.remove(z)

        if z in r.bundle:
            r.bundle = [x for x in r.bundle if x != z]

        r.task_zone = None
        r.zone_lease_until = 0
        r.task_no_progress = 0
        r.task_last_known = 0
        r.goal = None
        r.path = []
        r.goal_commit = 0



    def blacklist_zone(self, r, z, reason=""):
        if z is None:
            return

        self.release_zone(r, reason=reason, update_task_status=False)

        until = self.timestep + self.COOLDOWN_T
        r.blacklist[z] = until

        if z in self.zone_tasks:
            t = self.zone_tasks[z]
            t.last_owner = t.owner
            t.last_release_reason = reason
            t.owner = None
            t.status = "blacklisted"
            t.expires_at = until


    def zone_is_blacklisted(self, r, z):
        until = r.blacklist.get(z, -1)
        return self.timestep < until


    def lambda_breakdown(self, r, zone_stats, zones_assigned_count, *,
                        w_info=1.0, w_risk=1.0, w_terr=1.0, lambda_cost=0.20,
                        load_penalty=0.25):
        # unpack
        unknown_frac = zone_stats["unknown_frac"]
        avgT = zone_stats["avgT"]
        avgR = zone_stats["avgR"]
        f_water  = zone_stats["f_water"]
        f_stairs = zone_stats["f_stairs"]
        f_free   = zone_stats["f_free"]
        cx, cy = zone_stats["center"]

        # travel
        dist = heuristic(r.pos, (cx, cy))
        travel_cost = dist / float(GRID_W + GRID_H)
        if r.name == "Legged":
            travel_cost *= (1.0 - 0.5 * f_stairs)

        # ---- risk term: normalized by THIS robot limits ----
        eT = avgT / max(1e-6, r.temp_limit)
        eR = avgR / max(1e-6, r.rad_limit)

        # apply per-robot weights consistently
        wT = float(r.weights[0])
        wR = float(r.weights[1])
        lam_sig = (wT * eT) + (wR * eR)

        # convert to affinity:
        # avoiders -> negative utility for higher risk
        # rover -> positive utility for higher risk
        if r.name in ("Legged", "Drone"):
            risk_affinity = -(abs(lam_sig) ** P)
        elif r.name == "Rover":
            risk_affinity = +(abs(lam_sig) ** P)
        else:
            risk_affinity = 0.0

        # terrain affinity
        terrain_affinity = 0.0
        if r.name == "Boat":
            terrain_affinity += 2.0 * f_water
            terrain_affinity -= 0.5 * f_free
        elif r.name == "Legged":
            terrain_affinity += 2.0 * f_stairs
            terrain_affinity -= 1.0 * f_water
        elif r.name == "Drone":
            terrain_affinity += 1.0 * f_water
            terrain_affinity += 0.5 * f_stairs
        elif r.name == "Rover":
            terrain_affinity -= 0.5 * f_water

        # critical tendering
        critical_bonus = 0.0
        if r.name in ("Legged", "Drone"):
            critical_bonus += 1.5 * f_stairs
        if r.name == "Boat":
            critical_bonus += 1.5 * f_water

        # info gain
        info_gain = unknown_frac

        # combine
        base_u = (
            w_info * info_gain +
            w_risk * risk_affinity +
            w_terr * terrain_affinity -
            lambda_cost * travel_cost
        )

        load_term = load_penalty * zones_assigned_count[r.name]
        u = base_u + critical_bonus - load_term

        return {
            "u": u,
            "info": w_info * info_gain,
            "risk": w_risk * risk_affinity,
            "terr": w_terr * terrain_affinity,
            "travel": -lambda_cost * travel_cost,
            "critical": critical_bonus,
            "load": -load_term,
            "dist": dist,
            "unknown_frac": unknown_frac,
            "avgT": avgT,
            "avgR": avgR,
            "f_water": f_water,
            "f_stairs": f_stairs
        }

    
    def refresh_zone_tasks(self, union_belief):
        for z, task in self.zone_tasks.items():
            task.progress = self.zone_coverage(union_belief, z)

            # auto-mark complete zones as released/free (optional)
            if task.progress >= self.ZONE_DONE and task.status != "blacklisted":
                task.owner = None
                task.status = "released"
                task.expires_at = 0
    
    def zone_is_held(self, zone):
        t = self.zone_tasks.get(zone)
        if t is None:
            return False
        return (t.status == "held") and (self.timestep < t.expires_at) and (t.owner is not None)
    
    def zone_bid(self, r, zone, union_belief, union_T, union_R):
        zx, zy = zone
        stats = self.compute_zone_stats(zx, zy, union_belief, union_T, union_R)

        # feasibility
        if not self.zone_feasible(r, stats):
            return None

        # blacklist check
        if self.zone_is_blacklisted(r, zone):
            return None

        # progress skip (done)
        prog = self.zone_coverage(union_belief, zone)
        if prog >= self.ZONE_DONE:
            return None

        # IMPORTANT: do not bid on currently-held zones (lease still active)
        t = self.zone_tasks.get(zone)
        if t and t.status == "held" and self.timestep < t.expires_at and t.owner != r.name:
            return None

        # reuse your breakdown as the bid value
        bd = self.lambda_breakdown(
            r, stats,
            zones_assigned_count={rr.name: len(rr.bundle) for rr in self.robots},
            w_info=1.0, w_risk=1.0, w_terr=1.0,
            lambda_cost=0.05,
            load_penalty=0.25
        )

        return (float(bd["u"]), bd, stats)
    
    def choose_zone_goal(self, r, zone, union, reachable):
        """Pick an active frontier inside zone; fallback to a good entry point if none."""
        if zone is None:
            return None

        frontiers = self.zone_frontiers(union, reachable, zone)
        if frontiers:
            # choose frontier that is (a) close and (b) aligned with robot preference
            def score(c):
                dist = heuristic(r.pos, c)

                cx, cy = c[0] // CHUNK_SIZE, c[1] // CHUNK_SIZE
                lamT = r.chunked[0, cx, cy]
                lamR = r.chunked[1, cx, cy]
                eT = lamT / max(1e-6, r.temp_limit)
                eR = lamR / max(1e-6, r.rad_limit)
                e  = max(eT, eR)

                # prefer shortest unless risk matters
                if r.name == "Rover":
                    return dist - 10.0 * (e ** P)   # rover seeks
                else:
                    return dist + 10.0 * (e ** P)   # others avoid mildly

            return min(frontiers, key=score)

        # no frontier inside zone: go to a reachable "entry/anchor" cell near zone center
        zx, zy = zone
        stats = self.compute_zone_stats(zx, zy, union, self.get_union_temp_belief(), self.get_union_rad_belief())
        cx, cy = stats["center"]

        # pick reachable cell closest to center (acts like entry point)
        return min(reachable, key=lambda p: heuristic(p, (cx, cy)))

    
    def zone_bid_or_skip(self, r, zone, union_belief, union_T, union_R, zones_assigned_count):
        zx, zy = zone
        stats = self.compute_zone_stats(zx, zy, union_belief, union_T, union_R)

        # completion
        cov = self.zone_coverage(union_belief, zone)
        if cov >= self.ZONE_DONE:
            return None, {"zone": zone, "u": -1e9, "skip": "complete"}

        # blacklist
        if self.zone_is_blacklisted(r, zone):
            return None, {"zone": zone, "u": -1e9, "skip": "blacklisted"}

        # held by someone else (active lease)
        t = self.zone_tasks.get(zone)
        if t and t.status == "held" and self.timestep < t.expires_at and t.owner != r.name:
            return None, {"zone": zone, "u": -1e9, "skip": f"held_by_{t.owner}"}

        # feasibility
        if not self.zone_feasible(r, stats):
            return None, {"zone": zone, "u": -1e9, "skip": "infeasible"}

        # boat constraint (optional—if you still want it)
        if r.name == "Boat" and stats.get("f_water", 0.0) <= 0.0 and stats.get("unknown_frac", 0.0) < 0.40:
            return None, {"zone": zone, "u": -1e9, "skip": "boat_no_water"}

        # normal bid breakdown
        bd = self.lambda_breakdown(
            r, stats, zones_assigned_count,
            w_info=1.0, w_risk=1.0, w_terr=1.0,
            lambda_cost=0.05, load_penalty=0.25
        )

        row = {"zone": zone, **bd, "skip": ""}
        return (float(bd["u"]), bd, stats), row


# ---------- GUI routines ----------
def draw_grid(scr, sim, show_map, show_survivors, show_heat, show_rad,
              show_risk, show_plans, show_zones, union_belief, union_T, union_R):

    world = sim.world

    # -------------------------
    # Precompute risk overlay
    # -------------------------
    if show_risk:
        chunk_W = world.w // CHUNK_SIZE
        chunk_H = world.h // CHUNK_SIZE

        chunk_max_T = np.zeros((chunk_W, chunk_H), dtype=float)
        chunk_max_R = np.zeros((chunk_W, chunk_H), dtype=float)
        chunk_known = np.zeros((chunk_W, chunk_H), dtype=int)

        for cx in range(chunk_W):
            for cy in range(chunk_H):
                xs = range(cx * CHUNK_SIZE, min((cx + 1) * CHUNK_SIZE, world.w))
                ys = range(cy * CHUNK_SIZE, min((cy + 1) * CHUNK_SIZE, world.h))

                valsT = []
                valsR = []
                for x in xs:
                    for y in ys:
                        if union_belief[x, y] != Terrain.UNKNOWN:
                            t = union_T[x, y]
                            r = union_R[x, y]
                            if not np.isnan(t):
                                valsT.append(float(t))
                            if not np.isnan(r):
                                valsR.append(float(r))

                if valsT or valsR:
                    chunk_known[cx, cy] = 1
                    chunk_max_T[cx, cy] = max(valsT) if valsT else 0.0
                    chunk_max_R[cx, cy] = max(valsR) if valsR else 0.0

        # weights for display only (not planning)
        wT, wR = 10.0, 10.0
        risk_map = wT * chunk_max_T + wR * chunk_max_R
        max_risk = float(np.max(risk_map))
        if max_risk <= 1e-9:
            max_risk = 1.0

    # -------------------------
    # Draw cells
    # -------------------------
    for x in range(world.w):
        for y in range(world.h):
            pos = (x, y)

            # survivors override
            if pos in sim.found or (show_survivors and pos in sim.survivors):
                clr = SURVIVOR_COLOUR

            # risk overlay overrides terrain/heat/rad
            elif show_risk:
                cx, cy = x // CHUNK_SIZE, y // CHUNK_SIZE
                if chunk_known[cx, cy] == 0:
                    clr = (180, 180, 180)
                else:
                    risk_norm = min(max(risk_map[cx, cy] / max_risk, 0.0), 1.0)
                    clr = (
                        int(255 * risk_norm),
                        int(255 * (1.0 - risk_norm)),
                        0
                    )

            # heat/rad views use TRUE world (as you had)
            elif show_rad:
                rr = world.grid[x][y].radiation
                if world.grid[x][y].true_terrain == Terrain.WATER:
                    clr = (0, 0, 0)
                else:
                    v = min(rr / 100.0, 1.0)
                    clr = (0, int(255 * v), 0)

            elif show_heat:
                tt = world.grid[x][y].temperature
                t_norm = min(max(tt / 200.0, 0.0), 1.0)
                clr = (int(255 * t_norm), 0, int(255 * (1.0 - t_norm)))

            # normal terrain view (true map or union belief)
            else:
                terr = world.grid[x][y].true_terrain if show_map else union_belief[x, y]
                clr = TERRAIN_COLOUR[terr]

            rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(scr, clr, rect)
            pygame.draw.rect(scr, (150, 150, 150), rect, 1)

    # -------------------------
    # Draw zones overlay
    # -------------------------
    if show_zones:
        for zx in range(sim.zone_nx):
            for zy in range(sim.zone_ny):
                x0 = zx * sim.zone_w_cells
                y0 = zy * sim.zone_h_cells
                w_cells = sim.zone_w_cells
                h_cells = sim.zone_h_cells

                owner = None
                for r in sim.robots:
                    if (zx, zy) in r.assigned_zones:
                        owner = r
                        break

                if owner is None:
                    owner_color = (120, 120, 120)
                    width = 1
                else:
                    owner_color = ROBOT_COLOUR[owner.name]
                    width = 2

                rect = pygame.Rect(
                    x0 * CELL_SIZE,
                    y0 * CELL_SIZE,
                    w_cells * CELL_SIZE,
                    h_cells * CELL_SIZE
                )
                pygame.draw.rect(scr, owner_color, rect, width)
    

        # highlight each robot's current goal zone
        for r in sim.robots:
            if r.goal is None:
                continue
            gx, gy = r.goal
            z = sim.cell_to_zone(gx, gy)
            if z is None:
                continue
            zx, zy = z
            x0 = zx * sim.zone_w_cells
            y0 = zy * sim.zone_h_cells
            rect = pygame.Rect(
                x0 * CELL_SIZE,
                y0 * CELL_SIZE,
                sim.zone_w_cells * CELL_SIZE,
                sim.zone_h_cells * CELL_SIZE
            )
            pygame.draw.rect(scr, ROBOT_COLOUR[r.name], rect, 4)


            

def draw_robots(scr, robots, show_plans= False):
    if show_plans:
        for r in robots:
            # lighter color for path cells
            base_clr = ROBOT_COLOUR[r.name]
            path_clr = tuple(max(0, min(255, int(c * 0.5))) for c in base_clr)

            # draw each planned step
            for (px, py) in r.path:
                rect = pygame.Rect(px*CELL_SIZE, py*CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(scr, path_clr, rect)

            # outline the current goal (if any)
            if r.goal is not None:
                gx, gy = r.goal
                rect = pygame.Rect(gx*CELL_SIZE, gy*CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(scr, base_clr, rect, 2)  # width=2 border

    for r in robots:
        x,y = r.pos
        clr = ROBOT_COLOUR[r.name]
        pygame.draw.rect(scr, clr, (x*CELL_SIZE, y*CELL_SIZE, CELL_SIZE, CELL_SIZE))

def gui_loop():
    global sim
    pygame.init()
    screen = pygame.display.set_mode((GRID_W*CELL_SIZE + SIDEBAR_WIDTH, GRID_H*CELL_SIZE))
    pygame.display.set_caption("Heterogeneous Robot Fleet Simulator")
    font = pygame.font.SysFont(None, 24)
    sim   = FleetSim()
    running = False
    show_map = False
    show_survivors = False
    show_heat = False
    show_rad = False
    show_risk = False
    show_plans = False
    show_zones = False
    time_step = 0
    show_lambda = False
    
    lambda_btn = pygame.Rect(GRID_W*CELL_SIZE+10, 155, 120, 20)
    start_btn = pygame.Rect(GRID_W*CELL_SIZE+10, 10, 80, 20)
    map_btn   = pygame.Rect(GRID_W*CELL_SIZE+10, 35, 100, 20)
    surv_btn  = pygame.Rect(GRID_W*CELL_SIZE+10, 60, 100, 20)
    #heat_btn  = pygame.Rect(GRID_W*CELL_SIZE+10,85,100,20)
    #rad_btn   = pygame.Rect(GRID_W*CELL_SIZE+10,110,100,20)
    risk_btn  = pygame.Rect(GRID_W*CELL_SIZE+10, 85, 120, 20)
    plans_btn = pygame.Rect(GRID_W*CELL_SIZE+10, 110,120,20)
    zones_btn = pygame.Rect(GRID_W*CELL_SIZE+10, 130,120,20)
    clock = pygame.time.Clock()

    while True:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                if start_btn.collidepoint(ev.pos):
                    running = not running
                if map_btn.collidepoint(ev.pos):
                    show_map = not show_map
                if surv_btn.collidepoint(ev.pos):
                    show_survivors = not show_survivors
                # if heat_btn.collidepoint(ev.pos):
                #     show_heat = not show_heat
                #     if show_heat: show_rad = False
                # if rad_btn.collidepoint(ev.pos):
                #     show_rad = not show_rad
                #     if show_rad: show_heat = False
                if risk_btn.collidepoint(ev.pos):
                    show_risk = not show_risk
                if show_risk:
                    show_heat = False
                    show_rad  = False
                    show_plans = False
                if plans_btn.collidepoint(ev.pos):
                    show_plans = not show_plans
                    if show_plans:
                        show_heat = False
                        show_rad  = False
                if zones_btn.collidepoint(ev.pos):
                    show_zones = not show_zones
                    if show_zones:
                        show_risk  = False
                        show_plans = False
                        show_heat  = False
                        show_rad   = False
                if lambda_btn.collidepoint(ev.pos):
                    show_lambda = not show_lambda

        if running:
            if not sim.step():
                running=False
                print("Exploration Complete!")
                if sim.dead_robots:
                    print("Robots incapacitated during run:")
                    for name, reason in sim.dead_robots:
                        print(f" - {name} stopped due to {reason}")
            time_step += 1

        union_belief = sim.get_union_terrain_belief()
        union_T = sim.get_union_temp_belief()
        union_R = sim.get_union_rad_belief()
        draw_grid(screen, sim, show_map, show_survivors, show_heat, show_rad,
                  show_risk, show_plans, show_zones, union_belief, union_T, union_R)
        draw_robots(screen, sim.robots,show_plans)

        # sidebar
        pygame.draw.rect(screen, (255,255,255),
                         (GRID_W*CELL_SIZE, 0, SIDEBAR_WIDTH, GRID_H*CELL_SIZE))
        for btn, label in [
            (start_btn, 'Pause' if running else 'Start'),
            (map_btn,   'Hide Map' if show_map else 'Show Map'),
            (surv_btn,  'Hide Survi' if show_survivors else 'Show Survi'),
            #(heat_btn,  'Hide Heat' if show_heat else 'Show Heat'),
            #(rad_btn,   'Hide Rad' if show_rad else 'Show Rad'),
            (risk_btn, 'Hide Risk' if show_risk else 'Show Risk'),
            (plans_btn, 'Hide Plans' if show_plans else 'Show Plans'),
            (zones_btn, 'Hide Zones' if show_zones else 'Show Zones'),
            (lambda_btn, 'Hide λ' if show_lambda else 'Show λ'),

        ]:
            pygame.draw.rect(screen, (200,200,200), btn)
            screen.blit(font.render(label, True, (0,0,0)), (btn.x+10, btn.y+5))

        # stats
        screen.blit(font.render(f"Step: {time_step}", True, (0,0,0)),
                    (GRID_W*CELL_SIZE+10, plans_btn.y+50))
        total = sim.world.w * sim.world.h

        disc = int(np.sum(union_belief != Terrain.UNKNOWN))

        pct   = disc/total*100
        screen.blit(font.render(f"Coverage: {pct:.1f}%", True, (0,0,0)),
                    (GRID_W*CELL_SIZE+10, plans_btn.y+80))
        y = plans_btn.y + 110
        screen.blit(font.render("Battery:", True, (0,0,0)),
                    (GRID_W*CELL_SIZE+10, y)); y += 25
        for r in sim.robots:
            screen.blit(font.render(f"{r.name}: {r.battery:.1f}", True, (0,0,0)),
                        (GRID_W*CELL_SIZE+10, y))
            y += 25
            # ---- λ debug panel ----
        if show_lambda:
            y += 10
            screen.blit(font.render("λ top zones:", True, (0,0,0)),
                        (GRID_W*CELL_SIZE+10, y))
            y += 20

            K = 3
            for r in sim.robots:
                rows = sim.debug_zone_bids.get(r.name, [])
                screen.blit(font.render(f"{r.name}:", True, (0,0,0)),
                            (GRID_W*CELL_SIZE+10, y))
                y += 18

                for row in rows[:K]:
                    z = row["zone"]
                    txt = (
                        f"z{z} u={row['u']:.2f} "
                        f"i{row['info']:.1f} "
                        f"t{row['terr']:.1f} "
                        f"r{row['risk']:.1f}"
                    )
                    screen.blit(font.render(txt, True, (0,0,0)),
                                (GRID_W*CELL_SIZE+14, y))
                    y += 18
                y += 6


        # key & survivors
        screen.blit(font.render("Robot Key:", True, (0,0,0)),
                    (GRID_W*CELL_SIZE+10, y)); y += 25
        for nm, clr in ROBOT_COLOUR.items():
            pygame.draw.rect(screen, clr,
                             (GRID_W*CELL_SIZE+10, y, 20, 20))
            screen.blit(font.render(nm, True, (0,0,0)),
                        (GRID_W*CELL_SIZE+40, y))
            y += 25
        screen.blit(font.render("Survivors:", True, (0,0,0)),
                    (GRID_W*CELL_SIZE+10, y)); y += 25
        for idx, pos in enumerate(sim.survivors, start=1):
            found = pos in sim.found
            mark  = "✔" if found else "✖"
            col   = (0,200,0) if found else (200,0,0)
            screen.blit(font.render(f"{mark} S{idx} {pos}", True, col),
                        (GRID_W*CELL_SIZE+10, y))
            y += 25

        pygame.display.flip()
        clock.tick(FPS)

if __name__ == "__main__":
    gui_loop()


#Notes we want the robots to priorities close unknown cells over far away ones, to avoid wasting energy backtracking
#We dont want points to be cosntantly considered aka the drone example