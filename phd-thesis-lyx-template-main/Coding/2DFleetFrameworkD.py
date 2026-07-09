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
CELL_SIZE    = 6
GRID_W,GRID_H= 128,128
FPS          = 60
SIDEBAR_WIDTH= 260
MAX_BATTERY  = 1000
TEMP_LIMIT   = 1000      # robots only break down at very high spots
RAD_LIMIT    = 1000

# Framework risk aggregation
ZONE_CHUNKS= 8 #Zones for auction
ZONE_TARGET = 0.98
CHUNK_SIZE = 2    # 4×4 chunks
ALPHA      = 60    # global risk‐aversion multiplier
BETA = 0.5
P = 4
# -----------------------------------

# --- Random hazard field settings ---
N_HOTSPOTS_TEMP = 18
N_HOTSPOTS_RAD  = 18

# Binomial amplitude: amp = AMP_SCALE * Binomial(n, p)
# (choose n,p so that AMP_SCALE*n can exceed 100 comfortably)
TEMP_AMP_N = 30
TEMP_AMP_P = 0.35
TEMP_AMP_SCALE = 6.0     # max ~ 30*6 = 180

RAD_AMP_N = 28
RAD_AMP_P = 0.30
RAD_AMP_SCALE = 7.0      # max ~ 28*7 = 196

SIGMA_MIN = 2.5
SIGMA_MAX = 9.0

UNKNOWN_STEP_PENALTY = 1.2   # 0.5–3.0 (bigger = avoid unknown more)
UNKNOWN_HAZARD_PRIOR = 0.35  # fraction of chunk risk to assume in unknown


class Role(Enum):
    SCOUT = 1
    SCAN  = 2
    LOITER= 3
    RELAY = 4


class Terrain(Enum):
    UNKNOWN=0; FREE=1; OBSTACLE=2; STAIRS=3; WATER=4; BRIDGE=5

class Capability(Enum):
    LAND=1; STAIRS=2; WATER=3; AIR=4

TERRAIN_COLOUR = {
    Terrain.UNKNOWN:  (200,200,200),
    Terrain.FREE:     (255,255,255),
    Terrain.OBSTACLE: (  0,  0,  0),
    Terrain.STAIRS:   (255,255,  0),
    Terrain.WATER:    (  0,  0,255),
    Terrain.BRIDGE: (139,69,19)
}
T_UNKNOWN = 0
T_FREE    = 1
T_OBS     = 2
T_STAIRS  = 3
T_WATER   = 4
T_BRIDGE = 5

TERR_TO_CODE = {
    Terrain.UNKNOWN:  T_UNKNOWN,
    Terrain.FREE:     T_FREE,
    Terrain.OBSTACLE: T_OBS,
    Terrain.STAIRS:   T_STAIRS,
    Terrain.WATER:    T_WATER,
    Terrain.BRIDGE: T_BRIDGE
}

TERRAIN_COLOUR_CODE = {
    T_UNKNOWN: (200,200,200),
    T_FREE:    (255,255,255),
    T_OBS:     (0,0,0),
    T_STAIRS:  (255,255,0),
    T_WATER:   (0,0,255),
    T_BRIDGE: (139,69,19)
}

CAP_LAND   = 1 << 0
CAP_STAIRS = 1 << 1
CAP_WATER  = 1 << 2
CAP_AIR    = 1 << 3

def caps_to_mask(caps:set) -> int:
    m = 0
    if Capability.LAND in caps:   m |= CAP_LAND
    if Capability.STAIRS in caps: m |= CAP_STAIRS
    if Capability.WATER in caps:  m |= CAP_WATER
    if Capability.AIR in caps:    m |= CAP_AIR
    return m
# --- Terrain enum -> fast uint8 code ---

def terrain_to_code(t):
    # true_terrain in your sim is already stored as int codes (T_FREE, T_WATER, etc.)
    if isinstance(t, (int, np.integer)):
        return int(t)
    return TERR_TO_CODE.get(t, T_UNKNOWN)


def traversable_code(tb: int, caps_mask: int) -> bool:
    has_land   = (caps_mask & CAP_LAND) != 0
    has_stairs = (caps_mask & CAP_STAIRS) != 0
    has_water  = (caps_mask & CAP_WATER) != 0
    has_air    = (caps_mask & CAP_AIR) != 0

    if tb == T_OBS:
        return False
    if tb == T_STAIRS and (not has_stairs) and (not has_air):
        return False
    if tb == T_WATER and (not has_water) and (not has_air):
        return False
    if tb == T_BRIDGE and (not has_stairs) and (not has_air):
        return False
    if tb == T_FREE and (not has_land) and (not has_air):
        return False
    # T_UNKNOWN allowed (your A* has extra rules for it)
    return True


NBR4 = ((1,0), (-1,0), (0,1), (0,-1))

ROBOT_COLOUR = {
    "Legged": (  0,255,  0),
    "Drone":  (255,  0,255),
    "Boat":   (  0,255,255),
    "Rover":  (255,165,  0),
}

def robot_type(name: str) -> str:
    # names are like "Legged0", "Drone5", etc.
    for t in ("Legged", "Drone", "Boat", "Rover"):
        if name.startswith(t):
            return t
    return name  # fallback

SURVIVOR_COLOUR = (255,  0,  0)

from dataclasses import dataclass

@dataclass
class ZoneTask:
    zone: tuple           # (zx, zy)
    owners: list[str]           # robot names holding the zone (capacity-limited)
    status: str           # "free" | "held" | "released" | "blacklisted"
    progress: float       # coverage in [0,1]
    expires_at: int       # timestep when lease ends

    last_owner: str | None = None
    last_release_reason: str = ""


# ---------- Grid & Cells ----------
class Cell:
    def __init__(self, true_terrain=None):
        self.true_terrain = true_terrain or T_FREE
        self.terrain      = T_UNKNOWN
        self.temperature  = 0.0
        self.radiation    = 0.0

    def cost(self, caps:set, terrain_override=None) -> float:
        t = terrain_override if terrain_override is not None else self.true_terrain

        if t is T_UNKNOWN:
            # Unknown prior: allow movement but slightly penalize
            # (so they prefer known-free space)
            return 1.5
        if t is T_OBS:
            return math.inf
        if t is T_STAIRS and Capability.STAIRS not in caps and Capability.AIR not in caps:
            return math.inf
        if t is T_WATER and Capability.WATER not in caps and Capability.AIR not in caps:
            return math.inf
        # BRIDGE should be like FREE land
        if t is T_BRIDGE and Capability.STAIRS not in caps and Capability.AIR not in caps:
            return math.inf
        if t is T_FREE and Capability.LAND not in caps and Capability.AIR not in caps:
            return math.inf
        return 1.0

class GridWorld:
    def __init__(self, w, h):
        self.w, self.h = w, h
        self.grid = [[Cell() for _ in range(h)] for _ in range(w)]
        self._generate_demo_world()
        self._initialize_temperature()
        self._initialize_radiation()

    def _clamp(self, v, lo, hi):
        return lo if v < lo else hi if v > hi else v

    def _pick_edge_point(self):
        side = random.randint(0, 3)
        if side == 0:   return (0, random.randint(0, self.h-1))          # left
        if side == 1:   return (self.w-1, random.randint(0, self.h-1))    # right
        if side == 2:   return (random.randint(0, self.w-1), 0)           # top
        return (random.randint(0, self.w-1), self.h-1)                    # bottom

    def _pick_other_edge_point(self, p0):
        # Ensure p1 is on a DIFFERENT edge than p0
        x0, y0 = p0
        if x0 == 0:         forbidden = 0
        elif x0 == self.w-1: forbidden = 1
        elif y0 == 0:       forbidden = 2
        else:               forbidden = 3

        while True:
            side = random.randint(0, 3)
            if side == forbidden:
                continue
            if side == 0:   return (0, random.randint(0, self.h-1))
            if side == 1:   return (self.w-1, random.randint(0, self.h-1))
            if side == 2:   return (random.randint(0, self.w-1), 0)
            return (random.randint(0, self.w-1), self.h-1)

    def _river_path_points(self, p0, p1, n_ctrl=3):
        """
        Build a smooth polyline from p0(edge) to p1(edge) using intermediate control points.
        We then "sample" between them for continuous path coverage.
        """
        x0, y0 = p0
        x1, y1 = p1

        pts = [(float(x0), float(y0))]

        # direction vector from p0 to p1
        vx, vy = (x1 - x0), (y1 - y0)

        for k in range(1, n_ctrl+1):
            t = k / (n_ctrl+1)

            # base point along straight line
            bx = x0 + t * vx
            by = y0 + t * vy

            # perpendicular jitter magnitude (scaled by map size)
            mag = 0.10 * min(self.w, self.h)  # smoother
            jitter = random.uniform(-mag, mag)


            # perpendicular direction
            # if vx,vy is near zero, just jitter both
            if abs(vx) + abs(vy) < 1e-6:
                px, py = random.uniform(-mag, mag), random.uniform(-mag, mag)
            else:
                # perp = (-vy, vx)
                px, py = -vy, vx
                norm = math.sqrt(px*px + py*py)
                px, py = px / norm, py / norm

            cx = bx + jitter * px
            cy = by + jitter * py

            # clamp control points inside map interior
            cx = self._clamp(cx, 2, self.w-3)
            cy = self._clamp(cy, 2, self.h-3)

            pts.append((cx, cy))

        pts.append((float(x1), float(y1)))
        return pts
    
    def _place_bridge_on_component(self, comp_cells, bridge_w=5):
        if not comp_cells:
            return
        # pick a mid-ish cell in this component (closest to map center)
        cx0, cy0 = self.w // 2, self.h // 2
        comp_cells = sorted(comp_cells, key=lambda p: abs(p[0]-cx0) + abs(p[1]-cy0))
        sx, sy = comp_cells[len(comp_cells)//2]

        # estimate river direction locally
        def is_water(x, y):
            return 0 <= x < self.w and 0 <= y < self.h and self.grid[x][y].true_terrain == T_WATER

        horiz = (is_water(sx-1, sy) or is_water(sx+1, sy))
        vert  = (is_water(sx, sy-1) or is_water(sx, sy+1))

        if horiz and not vert:
            dx, dy = 0, 1
        elif vert and not horiz:
            dx, dy = 1, 0
        else:
            dx, dy = (1, 0) if random.random() < 0.5 else (0, 1)

        # --- extend bridge span until leaving water both directions ---
        def true_t(x, y):
            if 0 <= x < self.w and 0 <= y < self.h:
                return self.grid[x][y].true_terrain
            return T_OBS

        span = [(sx, sy)]

        bx, by = sx, sy
        while True:
            nbx, nby = bx + dx, by + dy
            if not (0 <= nbx < self.w and 0 <= nby < self.h): break
            span.append((nbx, nby))
            bx, by = nbx, nby
            if true_t(bx, by) is not T_WATER: break

        bx, by = sx, sy
        while True:
            nbx, nby = bx - dx, by - dy
            if not (0 <= nbx < self.w and 0 <= nby < self.h): break
            span.insert(0, (nbx, nby))
            bx, by = nbx, nby
            if true_t(bx, by) is not T_WATER: break

        # extend a little onto land on both ends
        for _ in range(2):
            fx, fy = span[-1]
            nx, ny = fx + dx, fy + dy
            if 0 <= nx < self.w and 0 <= ny < self.h: span.append((nx, ny))
            bx, by = span[0]
            nx, ny = bx - dx, by - dy
            if 0 <= nx < self.w and 0 <= ny < self.h: span.insert(0, (nx, ny))

        # stamp STAIRS thickly
        half_w = bridge_w // 2
        for (bx, by) in span:
            for j in range(-half_w, half_w + 1):
                px = bx + j * (1 - dx)
                py = by + j * (1 - dy)
                if 0 <= px < self.w and 0 <= py < self.h:
                    self.grid[px][py] = Cell(T_BRIDGE)




    def _sample_polyline(self, pts, step=0.5):
        """
        Densely sample a polyline so we can paint continuous river cells.
        step in 'cell units' (smaller => denser)
        """
        out = []
        for i in range(len(pts)-1):
            x0, y0 = pts[i]
            x1, y1 = pts[i+1]
            dx, dy = x1-x0, y1-y0
            dist = math.sqrt(dx*dx + dy*dy)
            if dist < 1e-6:
                continue
            n = max(2, int(dist / step))
            for k in range(n):
                t = k / (n-1)
                out.append((x0 + t*dx, y0 + t*dy))
        return out

    def _paint_disc(self, cx, cy, radius):
        """
        Paint a filled disc of WATER. This gives width.
        """
        r2 = radius * radius
        x0 = int(cx - radius - 1)
        x1 = int(cx + radius + 1)
        y0 = int(cy - radius - 1)
        y1 = int(cy + radius + 1)
        for x in range(x0, x1+1):
            for y in range(y0, y1+1):
                if 0 <= x < self.w and 0 <= y < self.h:
                    dx = x - cx
                    dy = y - cy
                    if dx*dx + dy*dy <= r2:
                        self.grid[x][y] = Cell(T_WATER)

    def _carve_river_edge_to_edge(self, base_width=4, n_ctrl=3):
        """
        Realistic continuous river:
        - picks p0 on one edge, p1 on another edge
        - creates smooth control polyline with perpendicular jitter
        - samples densely and paints a variable-width disc at each sample
        """
        p0 = self._pick_edge_point()
        p1 = self._pick_other_edge_point(p0)

        ctrl = self._river_path_points(p0, p1, n_ctrl=n_ctrl)
        samples = self._sample_polyline(ctrl, step=0.30)  # denser sampling = smoother river


        # width variation along river (wider mid-stream, slight variation)
        for i, (fx, fy) in enumerate(samples):
            t = i / max(1, (len(samples)-1))
            # make it slightly wider in the middle, narrower near edges
            w = base_width * (0.75 + 0.6 * math.sin(math.pi * t))
            w += random.uniform(-0.8, 0.8)  # natural variation
            w = max(2.0, w)

            self._paint_disc(int(round(fx)), int(round(fy)), radius=w/2.0)

        # return endpoints for possible bridge placement
        return p0, p1

    def _rect_clear(self, x0, y0, w, h, pad=2):
        # returns True if area (with padding) is all FREE (not WATER, not OBSTACLE, not STAIRS)
        xa = max(0, x0 - pad)
        xb = min(self.w, x0 + w + pad)
        ya = max(0, y0 - pad)
        yb = min(self.h, y0 + h + pad)
        for x in range(xa, xb):
            for y in range(ya, yb):
                tt = self.grid[x][y].true_terrain
                if tt != T_FREE:
                    return False
        return True

    def _stamp_house(self, hx, hy, house_w, house_h):
        # walls
        for x in range(hx, hx + house_w):
            self.grid[x][hy] = Cell(T_OBS)
            self.grid[x][hy + house_h - 1] = Cell(T_OBS)
        for y in range(hy, hy + house_h):
            self.grid[hx][y] = Cell(T_OBS)
            self.grid[hx + house_w - 1][y] = Cell(T_OBS)

        # interior FULL STAIRS (yellow fill)
        for x in range(hx+1, hx + house_w - 1):
            for y in range(hy+1, hy + house_h - 1):
                self.grid[x][y] = Cell(T_STAIRS)
        # --- Door: wider + STAIRS-yellow threshold so legged can enter ---
        door_y = hy + house_h - 1          # bottom wall
        door_x = hx + house_w // 2
        door_w = 3                         # width 3

        for dx in range(-(door_w//2), door_w//2 + 1):
            x = door_x + dx
            if hx <= x < hx + house_w:
                self.grid[x][door_y] = Cell(T_STAIRS)     # yellow doorway
                # also ensure the cell just inside is stairs (already is, but safe)
                if door_y - 1 > hy:
                    self.grid[x][door_y - 1] = Cell(T_STAIRS)




    def _stamp_box(self, bx, by, bw, bh):
        for x in range(bx, bx + bw):
            self.grid[x][by] = Cell(T_OBS)
            self.grid[x][by + bh - 1] = Cell(T_OBS)
        for y in range(by, by + bh):
            self.grid[bx][y] = Cell(T_OBS)
            self.grid[bx + bw - 1][y] = Cell(T_OBS)

            # door
        door_x = bx + bw // 2
        self.grid[door_x][by] = Cell(T_FREE)


    def _generate_demo_world(self):
        # start everything FREE
        for x in range(self.w):
            for y in range(self.h):
                self.grid[x][y] = Cell(T_FREE)

        # sprinkle obstacles lightly
        for x in range(self.w):
            for y in range(self.h):
                if random.random() < 0.04:
                    self.grid[x][y] = Cell(T_OBS)

        # -----------------------------
        # MULTI-RIVER WATER GENERATION
        # -----------------------------
        # -----------------------------
        #n_rivers = random.randint(1, 2)
        n_rivers = 1
        river_endpoints = []
        for _ in range(n_rivers):
            base_width = random.choice([4, 5, 6])      # wide rivers
            n_ctrl = random.choice([2, 3, 4])          # bendiness
            p0, p1 = self._carve_river_edge_to_edge(base_width=base_width, n_ctrl=n_ctrl)
            river_endpoints.append((p0, p1))

        # optional tributary: a thinner edge-to-edge river (looks like a secondary river)
        if random.random() < 0.7:
            base_width = random.choice([3, 4])
            n_ctrl = random.choice([3, 4])
            p0, p1 = self._carve_river_edge_to_edge(base_width=base_width, n_ctrl=n_ctrl)
            river_endpoints.append((p0, p1))

        # -----------------------------
        # GUARANTEED BRIDGE (STAIRS) that crosses a river
        # -----------------------------
        # place multiple bridges
        # Guaranteed bridges: one per water component (i.e., each river network)
        comps = self._water_components()
        for comp in comps:
            self._place_bridge_on_component(comp, bridge_w=5)




        # -----------------------------
        # GUARANTEED “UPSTAIRS HOUSE” (non-overlapping)
        # -----------------------------
        house_w, house_h = 14, 14

        placed = False
        for _try in range(200):
            hx = random.randint(2, self.w - house_w - 2)
            hy = random.randint(2, self.h - house_h - 2)

            # spread it away from center a bit (optional bias)
            # (keeps it from clustering near other stuff)
            if abs(hx - self.w//2) < 12 and abs(hy - self.h//2) < 12:
                continue

            if self._rect_clear(hx, hy, house_w, house_h, pad=4):
                self._stamp_house(hx, hy, house_w, house_h)
                placed = True
                break

        if not placed:
            # fallback: just do it somewhere (rare)
            hx = 4
            hy = self.h - house_h - 4
            self._stamp_house(hx, hy, house_w, house_h)


        # -----------------------------
        # EMPTY BOX ROOMS (non-overlapping + spread)
        # -----------------------------
        n_boxes = 10
        made = 0
        tries = 0
        while made < n_boxes and tries < 2500:
            tries += 1
            bw = random.randint(6, 18)
            bh = random.randint(6, 18)
            bx = random.randint(2, self.w - bw - 2)
            by = random.randint(2, self.h - bh - 2)

            # reject boxes too close to edges (spreads them out)
            if bx < 6 or by < 6 or bx > self.w - bw - 6 or by > self.h - bh - 6:
                continue

            if not self._rect_clear(bx, by, bw, bh, pad=2):
                continue

            self._stamp_box(bx, by, bw, bh)
            made += 1


    def _sample_hotspots(self, n_hotspots, amp_n, amp_p, amp_scale):
        """
        Returns list of hotspots: [((mx,my), sigma, amp), ...]
        Amp is binomially distributed so you get many medium peaks + some large peaks.
        """
        hs = []
        for _ in range(n_hotspots):
            mx = random.randint(0, self.w - 1)
            my = random.randint(0, self.h - 1)
            sigma = random.uniform(SIGMA_MIN, SIGMA_MAX)

            # binomial amplitude (can exceed 100)
            # uses numpy so distribution is correct
            amp = float(np.random.binomial(amp_n, amp_p)) * float(amp_scale)

            # avoid amp=0 dead hotspots
            if amp < 1.0:
                amp = 1.0

            hs.append(((mx, my), sigma, amp))
        return hs

    def _initialize_temperature(self):
        # Random binomial hotspots
        sources = self._sample_hotspots(
            N_HOTSPOTS_TEMP,
            TEMP_AMP_N, TEMP_AMP_P, TEMP_AMP_SCALE
        )

        for x in range(self.w):
            for y in range(self.h):
                temp = 0.0
                for (mx, my), sigma, amp in sources:
                    dx, dy = x - mx, y - my
                    d2 = (dx*dx + dy*dy) / (2.0 * sigma * sigma)
                    temp += amp * math.exp(-d2)

                # keep your water cooling rule
                if self.grid[x][y].true_terrain == T_WATER:
                    temp = 5.0

                self.grid[x][y].temperature = temp


    def _initialize_radiation(self):
        # Random binomial hotspots
        rad_sources = self._sample_hotspots(
            N_HOTSPOTS_RAD,
            RAD_AMP_N, RAD_AMP_P, RAD_AMP_SCALE
        )

        for x in range(self.w):
            for y in range(self.h):
                rad = 0.0
                for (mx, my), sigma, amp in rad_sources:
                    dx, dy = x - mx, y - my
                    d2 = (dx*dx + dy*dy) / (2.0 * sigma * sigma)
                    rad += amp * math.exp(-d2)

                # keep your water radiation rule
                if self.grid[x][y].true_terrain == T_WATER:
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
    
    def _water_components(self):
        """Return list of water connected components (4-neigh)."""
        visited = set()
        comps = []
        for x in range(self.w):
            for y in range(self.h):
                if self.grid[x][y].true_terrain != T_WATER:
                    continue
                if (x, y) in visited:
                    continue
                # BFS component
                q = deque([(x, y)])
                visited.add((x, y))
                comp = []
                while q:
                    cx, cy = q.popleft()
                    comp.append((cx, cy))
                    for nx, ny in ((cx-1,cy),(cx+1,cy),(cx,cy-1),(cx,cy+1)):
                        if 0 <= nx < self.w and 0 <= ny < self.h:
                            if self.grid[nx][ny].true_terrain == T_WATER and (nx, ny) not in visited:
                                visited.add((nx, ny))
                                q.append((nx, ny))
                comps.append(comp)
        return comps


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
                 rad_belief: np.ndarray,
                 radio_shadow: np.ndarray,
                 relay_ok: dict,
                 zone_of_cell_fn):
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
        self.radio_shadow = radio_shadow
        self.relay_ok = relay_ok
        self.zone_of_cell = zone_of_cell_fn

    @staticmethod
    def astar_fast(
        start, goal,
        caps_mask,                 # bitmask, explained below
        terrain_belief_u8,         # (W,H) uint8
        temp_belief_f32,           # (W,H) float32 (nan for unknown)
        rad_belief_f32,            # (W,H) float32
        chunked_risk,              # (2, Wc, Hc) float32
        weights,                   # (2,) float32
        alpha, beta, P,
        temp_limit, rad_limit,
        radio_shadow_bool,         # (W,H) bool
        relay_ok,                  # dict {zone:bool}
        cell_to_zone,              # fn(x,y)->(zx,zy)
        CHUNK_SIZE,
    ):
        W, H = terrain_belief_u8.shape
        sx, sy = start
        gx, gy = goal

        if start == goal:
            return []

        INF = 1e30
        gscore = np.full((W, H), INF, dtype=np.float32)
        gscore[sx, sy] = 0.0

        # parent pointers (store previous cell)
        px = np.full((W, H), -1, dtype=np.int16)
        py = np.full((W, H), -1, dtype=np.int16)

        closed = np.zeros((W, H), dtype=np.uint8)

        def h(x, y):
            return abs(x - gx) + abs(y - gy)

        heap = [(h(sx, sy), 0.0, sx, sy)]  # (f, g, x, y)

        # precompute for speed
        inv_T = 1.0 / max(1e-6, temp_limit)
        inv_R = 1.0 / max(1e-6, rad_limit)

        wT = max(0.0, float(weights[0]))
        wR = max(0.0, float(weights[1]))

        soft_t = 0.85 * temp_limit
        soft_r = 0.85 * rad_limit

        # capability checks via bitmask (faster than set membership)
        CAP_LAND   = 1 << 0
        CAP_STAIRS = 1 << 1
        CAP_WATER  = 1 << 2
        CAP_AIR    = 1 << 3

        has_land   = (caps_mask & CAP_LAND) != 0
        has_stairs = (caps_mask & CAP_STAIRS) != 0
        has_water  = (caps_mask & CAP_WATER) != 0
        has_air    = (caps_mask & CAP_AIR) != 0

        land_robot = has_land and (not has_air) and (not has_water)
        boat_robot = has_water and (not has_air)

        while heap:
            f, g, x, y = heapq.heappop(heap)
            if closed[x, y]:
                continue
            closed[x, y] = 1

            if (x, y) == (gx, gy):
                # reconstruct
                path = []
                cx, cy = gx, gy
                while (cx, cy) != (sx, sy):
                    path.append((cx, cy))
                    ncx, ncy = int(px[cx, cy]), int(py[cx, cy])
                    if ncx < 0:
                        return []
                    cx, cy = ncx, ncy
                path.reverse()
                return path

            for dx, dy in NBR4:
                nx, ny = x + dx, y + dy
                if nx < 0 or nx >= W or ny < 0 or ny >= H:
                    continue
                if closed[nx, ny]:
                    continue

                tb = int(terrain_belief_u8[nx, ny])

                # ---------- feasibility by terrain ----------
                # obstacle always blocked
                if tb == T_OBS:
                    continue
                # stairs blocked unless STAIRS or AIR
                if tb == T_STAIRS and (not has_stairs) and (not has_air):
                    continue
                # water blocked unless WATER or AIR
                if tb == T_WATER and (not has_water) and (not has_air):
                    continue
                # bridge blocked unless LAND or AIR
                if tb == T_BRIDGE and (not has_stairs) and (not has_air):
                    continue
                # free land blocked unless LAND or AIR
                if tb == T_FREE and (not has_land) and (not has_air):
                    continue

                # ---------- special UNKNOWN rules ----------
                if tb == T_UNKNOWN:
                    if land_robot:
                        # avoid UNKNOWN adjacent to known water
                        water_adj = False
                        for ddx, ddy in NBR4:
                            ax, ay = nx + ddx, ny + ddy
                            if 0 <= ax < W and 0 <= ay < H:
                                if terrain_belief_u8[ax, ay] == T_WATER:
                                    water_adj = True
                                    break
                        if water_adj:
                            continue

                    if boat_robot:
                        # only allow UNKNOWN if adjacent to known water
                        water_adj = False
                        for ddx, ddy in NBR4:
                            ax, ay = nx + ddx, ny + ddy
                            if 0 <= ax < W and 0 <= ay < H:
                                if terrain_belief_u8[ax, ay] == T_WATER:
                                    water_adj = True
                                    break
                        if not water_adj:
                            continue

                    # conservative hazard halo check (fragile robots)
                    if (temp_limit < 9000.0) or (rad_limit < 9000.0):
                        danger = False
                        for ddx, ddy in NBR4:
                            ax, ay = nx + ddx, ny + ddy
                            if 0 <= ax < W and 0 <= ay < H:
                                if terrain_belief_u8[ax, ay] == T_UNKNOWN:
                                    continue
                                t2 = temp_belief_f32[ax, ay]
                                r2 = rad_belief_f32[ax, ay]
                                if (not np.isnan(t2) and t2 > soft_t) or (not np.isnan(r2) and r2 > soft_r):
                                    danger = True
                                    break
                        if danger:
                            continue

                else:
                    # lethal avoidance if known
                    t = temp_belief_f32[nx, ny]
                    r = rad_belief_f32[nx, ny]
                    if (not np.isnan(t) and t > temp_limit) or (not np.isnan(r) and r > rad_limit):
                        continue

                # ---------- radio shadow gate ----------
                # ---------- radio shadow gate (neighbor-permeable near borders) ----------
                # ---------- (neighbor-permeable ONLY near borders) ----------
                if radio_shadow_bool[nx, ny]:
                    z = cell_to_zone(nx, ny)
                    if z is not None and (not relay_ok.get(z, False)):

                        zx, zy = z

                        # zone bounds in cell coordinates
                        ZW = ZONE_CHUNKS * CHUNK_SIZE
                        ZH = ZONE_CHUNKS * CHUNK_SIZE
                        x0 = zx * ZW
                        y0 = zy * ZH
                        x1 = x0 + ZW - 1
                        y1 = y0 + ZH - 1

                        BORDER = 3  # how many cells from the border counts as "near"

                        ok = False  # IMPORTANT: start False

                        for dx, dy in NBR4:
                            nz = (zx + dx, zy + dy)

                            # must exist + must have relay
                            if not relay_ok.get(nz, False):
                                continue

                            # allow entry ONLY if the candidate cell is close to the shared border
                            if nz == (zx - 1, zy) and (nx - x0) < BORDER:
                                ok = True; break
                            if nz == (zx + 1, zy) and (x1 - nx) < BORDER:
                                ok = True; break
                            if nz == (zx, zy - 1) and (ny - y0) < BORDER:
                                ok = True; break
                            if nz == (zx, zy + 1) and (y1 - ny) < BORDER:
                                ok = True; break

                        if not ok:
                            continue



                # ---------- costs ----------
                base = 1.0
                # (optional bonuses)
                if tb == T_WATER and has_air and (caps_mask == CAP_AIR):
                    base *= 0.5
                elif tb == T_STAIRS and has_stairs:
                    base *= 0.2

                cx, cy = nx // CHUNK_SIZE, ny // CHUNK_SIZE
                lamT = float(chunked_risk[0, cx, cy])
                lamR = float(chunked_risk[1, cx, cy])
                e_c = max(lamT * inv_T, lamR * inv_R)

                # ---------- costs ----------
                base = 1.0

                # small terrain bonuses (your existing ones)
                if tb == T_WATER and has_air and (caps_mask == CAP_AIR):
                    base *= 0.5
                elif tb == T_STAIRS and has_stairs:
                    base *= 0.2

                cx, cy = nx // CHUNK_SIZE, ny // CHUNK_SIZE
                lamT = float(chunked_risk[0, cx, cy])
                lamR = float(chunked_risk[1, cx, cy])
                e_c = max(lamT * inv_T, lamR * inv_R)

                # cell-level hazard e (only if cell is known)
                if tb != T_UNKNOWN:
                    t = 0.0 if np.isnan(temp_belief_f32[nx, ny]) else float(temp_belief_f32[nx, ny])
                    r = 0.0 if np.isnan(rad_belief_f32[nx, ny])  else float(rad_belief_f32[nx, ny])
                    eT = t * inv_T
                    eR = r * inv_R
                    e  = max(wT * eT, wR * eR)
                    unk_pen = 0.0
                else:
                    # UNKNOWN: assume some risk from the chunk + add explicit exploration penalty
                    e = UNKNOWN_HAZARD_PRIOR * e_c
                    unk_pen = UNKNOWN_STEP_PENALTY

                step = base + unk_pen + alpha * (e_c ** P) + beta * (e ** P)
                ng = g + step

                if ng < gscore[nx, ny]:
                    gscore[nx, ny] = ng
                    px[nx, ny] = x
                    py[nx, ny] = y
                    heapq.heappush(heap, (ng + h(nx, ny), ng, nx, ny))

        return []

    def search(self):
        caps_mask = caps_to_mask(self.caps)
        return AStar.astar_fast(
            start=self.start,
            goal=self.goal,
            caps_mask=caps_mask,
            terrain_belief_u8=self.terrain_belief,
            temp_belief_f32=self.temp_belief,
            rad_belief_f32=self.rad_belief,
            chunked_risk=self.chunked,
            weights=self.weights,
            alpha=self.alpha,
            beta=BETA,
            P=P,
            temp_limit=self.temp_limit,
            rad_limit=self.rad_limit,
            radio_shadow_bool=self.radio_shadow,
            relay_ok=self.relay_ok,
            cell_to_zone=self.zone_of_cell,
            CHUNK_SIZE=CHUNK_SIZE,
        )

    



# ---------- Robot (single class) ----------
class Robot:
    def __init__(self, name, x, y, caps, world, sim,
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
        self.sim = sim
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
        self.role = Role.SCAN
        self.role_locked_until = 0
        self.reveal_R = 2  # base reveal radius

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
        self.terrain_belief = np.full((GRID_W, GRID_H), T_UNKNOWN, dtype=np.uint8)
        self.temp_belief    = np.full((GRID_W, GRID_H), np.nan, dtype=np.float32)
        self.rad_belief     = np.full((GRID_W, GRID_H), np.nan, dtype=np.float32)
        self.cached_frontiers = []
        self.frontier_refresh = 40

        self.reachable_cache = None
        self.reachable_refresh = 40
        # in Robot.__init__
        self.survival_p = 1.0
        self.zone_frontier_signal = 0.0   # in [0,1], updated each tick from reachable frontiers
        self.zone_frontier_count  = 0     # optional debug



        # initial reveal & chunked build
        if self.reveal():
            self.recompute_chunked()

    def reveal(self):
        changed = False
        R = self.reveal_R
        x0, y0 = self.pos

        for dx in range(-R, R+1):
            for dy in range(-R, R+1):
                if dx*dx + dy*dy <= R*R:
                    nx, ny = x0+dx, y0+dy
                    if 0 <= nx < self.world.w and 0 <= ny < self.world.h:
                        if not self.known_mask[nx, ny]:
                            changed = True
                            self.known_mask[nx, ny] = True

                            cell = self.world.grid[nx][ny]
                            self.terrain_belief[nx, ny] = terrain_to_code(cell.true_terrain)
                            self.temp_belief[nx, ny]    = cell.temperature
                            self.rad_belief[nx, ny]     = cell.radiation

        return changed



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
        if tgt == self.pos:
            self.goal = None
            self.path = []
            self.goal_commit = 0
            return

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
            self.rad_belief,
            radio_shadow=self.sim.radio_shadow,
            relay_ok=self.sim.relay_ok,
            zone_of_cell_fn=self.sim.cell_to_zone
        ).search()
        
        if not path:
            self.failed_goals[tgt] = self.sim.timestep + 80
            self.goal = None
            self.path = []
            self.goal_commit = 0
            return

        
        risk, has_lethal = self.path_risk_score(path, K=12)
        if has_lethal:
            self.failed_goals[tgt] = self.sim.timestep + 80
            self.goal = None
            self.path = []
            self.goal_commit = 0
            return

        self.path = path

        # Soft commitment: only lock goal for a short time
        self.goal_commit = 20   # tune: 10-30

        from_zone = self.sim.cell_to_zone(tgt[0], tgt[1])
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

        # Replan ONLY if path is empty or invalid under current belief
        if self.path_has_error():
            new_path = AStar(
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
                self.rad_belief,
                radio_shadow=self.sim.radio_shadow,
                relay_ok=self.sim.relay_ok,
                zone_of_cell_fn=self.sim.cell_to_zone
            ).search()

            if not new_path:
                # Can't reach goal anymore -> drop it and cooldown it
                if self.goal is not None:
                    self.failed_goals[self.goal] = self.sim.timestep + 80
                self.goal = None
                self.path = []
                self.goal_commit = 0
                return

            self.path = new_path

        # Emergency abort if next step is predicted lethal based on belief
        next_step = self.path[0]
        if self.predicted_cell_lethal(next_step):
            self.goal = None
            self.path = []
            self.goal_commit = 0
            return

        # ---- execute next step ----
        prev_pos = self.pos
        attempted = self.path[0]          # cell we intend to step into
        self.pos = self.path.pop(0)

        true_t = self.world.grid[self.pos[0]][self.pos[1]].true_terrain


        # ---- verify against TRUE terrain (hard safety gate) ----
        true_t = self.world.grid[self.pos[0]][self.pos[1]].true_terrain

        def illegal_for_caps(t: Terrain, caps: set) -> bool:
                # OBSTACLE always illegal
                if t is T_OBS:
                    return True

                # Water illegal unless WATER or AIR
                if t is T_WATER and (Capability.WATER not in caps) and (Capability.AIR not in caps):
                    return True

                # Stairs illegal unless STAIRS or AIR
                if t is T_STAIRS and (Capability.STAIRS not in caps) and (Capability.AIR not in caps):
                    return True

                # Free land illegal unless LAND or AIR
                if t is T_FREE and (Capability.LAND not in caps) and (Capability.AIR not in caps):
                    return True

                return False

            # Boat rule: boats must stay on water
        if (Capability.WATER in self.caps) and (Capability.AIR not in self.caps):
            if true_t is not T_WATER:
                self.terrain_belief[self.pos[0], self.pos[1]] = true_t
                self.known_mask[self.pos[0], self.pos[1]] = True

                bad_cell = self.pos

                self.pos = prev_pos

                # KEEP the goal, just force replanning
                self.path = []
                self.goal_commit = 0  # optional

                self.mark_failed_goal(bad_cell, cooldown=120)
                return


        # General rule
        if illegal_for_caps(true_t, self.caps):
            self.terrain_belief[self.pos[0], self.pos[1]] = true_t
            self.known_mask[self.pos[0], self.pos[1]] = True

            bad_cell = self.pos

            self.pos = prev_pos

            # KEEP the goal, just force replanning
            self.path = []
            self.goal_commit = 0  # optional

            self.mark_failed_goal(bad_cell, cooldown=120)
            return

        
        # --- RADIO SHADOW RULE ---
        if self.sim.radio_shadow[self.pos[0], self.pos[1]]:
            z = self.sim.cell_to_zone(self.pos[0], self.pos[1])
            if not self.sim.relay_ok_extended(z):
                self.active = False
                self.goal = None
                self.path = []
                self.death_reason = "comms lost in radio shadow"
                return



        self.reveal()
        self.recompute_chunked()

        # Battery drain
        role_mult = {Role.SCOUT: 1.4, Role.SCAN: 1.0, Role.RELAY: 1.1, Role.LOITER: 0.6}[self.role]
        # then apply to battery decrement (multiply the decrement)

        if self.name.startswith("Drone"):
            self.battery -= 2*role_mult
        elif self.name.startswith("Legged"):
            self.battery -= 1*role_mult
        elif self.name.startswith("Boat"):
            self.battery -= 2*role_mult
        else:
            self.battery -= 0.5*role_mult



        # Check lethal exposure at new position
        c = self.world.grid[self.pos[0]][self.pos[1]]
        self.dose_T += max(0.0, c.temperature) * 0.01
        self.dose_R += max(0.0, c.radiation) * 0.01

        over_t = c.temperature > self.temp_limit
        over_r = c.radiation  > self.rad_limit
                # in step after dose updates
        k = 2.0
        lam = 300.0   # tune
        dose = self.dose_T + self.dose_R
        self.survival_p = math.exp(- (dose/lam)**k )


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
    

    def mark_failed_goal(self, cell, cooldown=60):
            # avoid repeated re-tries of the same bad target
        self.failed_goals[cell] = self.sim.timestep + cooldown


                
    def predicted_cell_lethal(self, cell_xy):
            """Use BELIEF hazards to decide if a cell is lethal (only if known)."""
            x, y = cell_xy
            tb = int(self.terrain_belief[x, y])
            if tb == T_UNKNOWN:
                return False  # unknown hazard -> don't call it lethal
            t = self.temp_belief[x, y]
            r = self.rad_belief[x, y]
            if (not np.isnan(t) and t > self.temp_limit) or (not np.isnan(r) and r > self.rad_limit):
                return True
            return False
    
    def cell_traversable_under_belief(self, xy) -> bool:
        x, y = xy
        tb = self.terrain_belief[x, y]

        # match the same terrain feasibility rules A* uses
        base = self.world.grid[x][y].cost(self.caps, terrain_override=tb)
        if math.isinf(base):
            return False

        # belief-based lethal check (only if known)
        if tb is not T_UNKNOWN:
            t = self.temp_belief[x, y]
            r = self.rad_belief[x, y]
            if (not np.isnan(t) and t > self.temp_limit) or (not np.isnan(r) and r > self.rad_limit):
                return False

        # radio shadow feasibility check (same as A*)
        if self.sim.radio_shadow[x, y]:
            z = self.sim.cell_to_zone(x, y)
            if (z is not None) and (not self.sim.relay_ok.get(z, False)):
                return False

        return True


    def path_has_error(self) -> bool:
        """Return True if current stored path is invalid under *current belief*."""
        if not self.path:
            return True

        # If the next step isn't traversable anymore, we must replan.
        if not self.cell_traversable_under_belief(self.path[0]):
            return True

        # OPTIONAL: light validation a few steps ahead (keeps it cheap)
        # This avoids following a path that newly revealed an obstacle further ahead.
        LOOKAHEAD = 8
        for p in self.path[:LOOKAHEAD]:
            if not self.cell_traversable_under_belief(p):
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
            if tb is T_UNKNOWN:
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

        self.radio_shadow = np.zeros((GRID_W, GRID_H), dtype=bool)

        #Zones
        self.zone_w_cells = ZONE_CHUNKS * CHUNK_SIZE
        self.zone_h_cells = ZONE_CHUNKS * CHUNK_SIZE
        self.zone_nx = GRID_W // self.zone_w_cells
        self.zone_ny = GRID_H // self.zone_h_cells

        self.found       = set()
        self.dead_robots = []

        self.debug_zone_bids = {}   # robot_name -> list of dict rows (sorted later)
        self.show_lambda_debug = False

        self.zone_tasks = {(zx,zy): ZoneTask((zx,zy), [], "free", 0.0, 0)
                   for zx in range(self.zone_nx) for zy in range(self.zone_ny)}

        
        self.timestep = 0

        self.LEASE_T = 40          # lock zone for N ticks (stops flip-flopping)
        self.COOLDOWN_T = 80       # blacklist zone for N ticks after failure
        self.ZONE_DONE = 1      # completion threshold (global)
        self.NO_PROGRESS_K = 25    # if no progress for K ticks => drop zone

        self.MAX_BUNDLE = 4
        self.CBBA_ITERS = 3     # 2–5 is fine
        self.REPLAN_T = 50      # already using %50

        self.debug_zone_winners = {}    # zone -> {winner, u, losers:[(robot,u)], reason}
        self.debug_cbba_rounds = []     # list of round summaries (optional)

        self.PRINT_CBBA_DEBUG = False

        self.union_belief = None
        self.union_T = None
        self.union_R = None

        self.ZONE_CAPACITY = 2
        self._zone_stats_cache = {}
        self._zone_stats_cache_key = None






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
        # --- robot templates ---
        robot_templates = [
            ("Legged", {Capability.LAND, Capability.STAIRS}, np.array([10.0, 10.0]), (TEMP_LIMIT, RAD_LIMIT)),
            ("Drone",  {Capability.AIR},                  np.array([10.0, 10.0]), (TEMP_LIMIT, RAD_LIMIT)),
            ("Boat",   {Capability.WATER},                np.array([ 0.0,  0.0]), (9999.0, 9999.0)),
            ("Rover",  {Capability.LAND},                 np.array([-2.0, -2.0]), (9999.0, 9999.0)),
        ]
        water_cells = [
            (x,y) for x in range(self.world.w)
                for y in range(self.world.h)
                if self.world.grid[x][y].true_terrain == T_WATER
        ]

        # how many total robots (suggest 8–16 for 128x128)
        N_ROBOTS = 12
        robots_per_cluster = 3

        clusters = [
            (6, 6),                              # top-left
            (GRID_W - 7, 6),                      # top-right
            (6, GRID_H - 7),                      # bottom-left
            (GRID_W - 7, GRID_H - 7),             # bottom-right
        ]

        # helper: jitter around a center
        def jitter(center, r=6):
            cx, cy = center
            x = max(1, min(GRID_W - 2, cx + random.randint(-r, r)))
            y = max(1, min(GRID_H - 2, cy + random.randint(-r, r)))
            return x, y

        # optional: quadrant-filtered water cells for boats
        def in_quadrant(x, y, qx, qy):
            # qx: 0 left / 1 right, qy: 0 top / 1 bottom
            if qx == 0 and x > GRID_W // 2: return False
            if qx == 1 and x < GRID_W // 2: return False
            if qy == 0 and y > GRID_H // 2: return False
            if qy == 1 and y < GRID_H // 2: return False
            return True

        water_by_quad = {}
        for qx in (0, 1):
            for qy in (0, 1):
                water_by_quad[(qx, qy)] = [(x, y) for (x, y) in water_cells if in_quadrant(x, y, qx, qy)]

        # build robots with fixed totals
        robot_id = 0
        self.robots = []

        desired = {"Legged": 3, "Drone": 4, "Boat": 2, "Rover": 3}  # sum=12
        spawn_list = []
        for t, n in desired.items():
            spawn_list += [t] * n
        random.shuffle(spawn_list)

        # map type -> template
        tpl = {tname: (tname, caps, weights, lims) for (tname, caps, weights, lims) in robot_templates}

        for i, tname in enumerate(spawn_list):
            center = clusters[i % len(clusters)]
            qx = 0 if center[0] < GRID_W // 2 else 1
            qy = 0 if center[1] < GRID_H // 2 else 1

            _, caps, weights, (tlim, rlim) = tpl[tname]
            name = f"{tname}{robot_id}"
            robot_id += 1

            sx, sy = jitter(center, r=8)

            if tname == "Boat" and water_cells:
                pool = water_by_quad.get((qx, qy), [])
                sx, sy = random.choice(pool) if pool else random.choice(water_cells)
            else:
                for _ in range(30):
                    sx, sy = jitter(center, r=10)
                    tt = self.world.grid[sx][sy].true_terrain
                    if tt == T_FREE or (tt == T_STAIRS and Capability.STAIRS in caps):
                        break

            self.robots.append(Robot(
                name, sx, sy, caps,
                self.world, sim=self,
                raw_risk=raw, weights=weights, alpha=ALPHA,
                temp_limit=tlim, rad_limit=rlim
            ))




        # spawn points around edges (spread out)
        spawn_points = []
        margin = 2
        for k in range(N_ROBOTS):
            side = k % 4
            if side == 0:   spawn_points.append((margin, margin + (k*7) % (GRID_H-2*margin)))
            elif side == 1: spawn_points.append((GRID_W-1-margin, margin + (k*7) % (GRID_H-2*margin)))
            elif side == 2: spawn_points.append((margin + (k*7) % (GRID_W-2*margin), margin))
            else:           spawn_points.append((margin + (k*7) % (GRID_W-2*margin), GRID_H-1-margin))


        free_cells = [
            (x,y) for x in range(self.world.w)
                   for y in range(self.world.h)
                   if self.world.grid[x][y].true_terrain == T_FREE
        ]




        def near_cells(center, radius=8):
            cx, cy = center
            out = []
            for x in range(max(0, cx-radius), min(self.world.w, cx+radius+1)):
                for y in range(max(0, cy-radius), min(self.world.h, cy+radius+1)):
                    if self.world.grid[x][y].true_terrain == T_FREE:
                        out.append((x, y))
            return out

        # --- critical survivors near features ---
        # 1) near bridge
        bridge_guess = (int(self.world.w*0.75), self.world.h//2)
        bridge_pool = near_cells(bridge_guess, radius=10)

        # 2) near stairs-house (roughly lower-right quadrant)
        house_guess = (int(self.world.w*0.55), int(self.world.h*0.75))
        house_pool = near_cells(house_guess, radius=12)

        # 3) near river shore (left side)
        shore_guess = (self.world.w//6 + 2, self.world.h//2 + 6)
        shore_pool = near_cells(shore_guess, radius=10)

        critical = []
        for pool in (bridge_pool, house_pool, shore_pool):
            if pool:
                critical.append(random.choice(pool))

        # fill the rest randomly, avoiding duplicates
        remaining = [c for c in free_cells if c not in critical]
        self.survivors = critical + random.sample(remaining, 18 - len(critical))


        self.found       = set()
        self.dead_robots = []
        
        self.assign_zones_cbba()
        self.build_radio_shadow()
        # after build_radio_shadow()
        self.relay_ok = {(zx, zy): False for zx in range(self.zone_nx) for zy in range(self.zone_ny)}
        for zx in range(self.zone_nx):
            for zy in range(self.zone_ny):
                self.relay_ok[(zx, zy)] = True



    # ---- initialize union caches so GUI can render before first step ----
        self.union_belief = self.get_union_terrain_belief()
        self.union_T = self.get_union_temp_belief()
        self.union_R = self.get_union_rad_belief()




    def get_union_terrain_belief(self):
        union = np.full((GRID_W, GRID_H), T_UNKNOWN, dtype=np.uint8)
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
                if union[x, y] != T_UNKNOWN:
                    known += 1
        return (known / total) if total > 0 else 1.0

    def zone_frontiers(self, union, reachable, zone, robot=None):
        xs, ys = self.zone_cells(zone)
        out = []

        is_boat = (robot is not None and robot.name.startswith("Boat") and
                (Capability.WATER in robot.caps) and (Capability.AIR not in robot.caps))

        for x in xs:
            for y in ys:
                if (x, y) not in reachable:
                    continue

                if is_boat:
                    # Boat frontiers = WATER cells that can reveal unknown neighbors
                    if union[x, y] == T_WATER:
                        if any(union[nx, ny] == T_UNKNOWN for (nx, ny) in self.world.neighbours((x, y))):
                            out.append((x, y))
                else:
                    # Everyone else: unknown cells in reachable
                    if union[x, y] == T_UNKNOWN:
                        out.append((x, y))

        return out




    def assign_zones_cbba(self):
        union_belief = self.get_union_terrain_belief()
        union_T = self.get_union_temp_belief()
        union_R = self.get_union_rad_belief()

        zone_stats_cache = {}
        def get_stats(z):
            if z not in zone_stats_cache:
                zx, zy = z
                zone_stats_cache[z] = self.compute_zone_stats(zx, zy, union_belief, union_T, union_R)
            return zone_stats_cache[z]


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
                    bid, _row = self.zone_bid_or_skip(r, z, union_belief, union_T, union_R, zones_assigned_count, stats=get_stats(z))
                    if bid is None:
                        continue
                    u, bd, _ = bid
                    zone_claims.setdefault(z, []).append((r.name, u))

            # decide winners + drop losers
            for z, claims in zone_claims.items():
                if len(claims) > 1:
                    round_summary["conflicts"] += 1

                claims_sorted = sorted(claims, key=lambda t: t[1], reverse=True)
                winners = claims_sorted[:self.ZONE_CAPACITY]
                winner_names = {nm for (nm, _u) in winners}
                losers = claims_sorted[self.ZONE_CAPACITY:]

                # record debug
                self.debug_zone_winners[z] = {
                    "winners": winners,
                    "losers": losers,
                    "iter": it
                }

                # force losers to drop
                for r in self.robots:
                    if (r.name not in winner_names) and (z in r.bundle):
                        r.bundle = [zz for zz in r.bundle if zz != z]


            # save round summary
            for r in self.robots:
                round_summary["wins"][r.name] = len(r.bundle)
            self.debug_cbba_rounds.append(round_summary)
            if getattr(self, "PRINT_CBBA_DEBUG", False):
                print("\n[CBBA] Zone winners (conflicts only):")
                for z, d in self.debug_zone_winners.items():
                    if len(d["losers"]) == 0:
                        continue
                    print(f"  zone={z} winner={d['winner']} u={d['u']:.3f} losers={d['losers']}")



        # 3) Finalize: bundle -> assigned_zones, update zone_tasks to held
        # expire old holds
        for z, task in self.zone_tasks.items():
            if task.status == "held" and self.timestep >= task.expires_at:
                task.owners = []
                task.status = "released"
                task.expires_at = 0


        # clear owners first
        for z, t in self.zone_tasks.items():
            if t.status != "blacklisted":
                t.owners = []

        for r in self.robots:
            r.assigned_zones = list(r.bundle)
            for z in r.assigned_zones:
                t = self.zone_tasks[z]
                if t.status == "blacklisted":
                    continue
                if r.name not in t.owners and len(t.owners) < self.ZONE_CAPACITY:
                    t.owners.append(r.name)
                t.status = "held" if t.owners else "free"
                t.expires_at = self.timestep + self.LEASE_T

        # ---- Fallback: ensure every active robot gets at least 1 zone ----

        for r in self.robots:
            if (not r.active) or (r.battery <= 0):
                continue
            if len(r.bundle) == 0:
                z = self.best_fallback_zone(r, union_belief, union_T, union_R, allow_held=False)
                if z is None:
                    # last resort: even allow zones held by others
                    z = self.best_fallback_zone(r, union_belief, union_T, union_R, allow_held=True)
                if z is not None:
                    r.bundle.append(z)



    def zone_feasible(self, r, stats):
    # stats contains: unknown_frac, f_water, f_stairs, avgT, avgR (belief-based)

        if (not r.active) or (r.battery <= 0):
            return False

        uf = stats["unknown_frac"]
        fw = stats["f_water"]
        fs = stats["f_stairs"]

        if r.name.startswith("Boat"):
            # allow if water OR lots unknown (scouting)
            return (fw > 0.05) or (uf > 0.40)

        if r.name.startswith("Legged"):
            # avoid water-heavy zones unless stairs/unknown justify
            if fw > 0.30 and fs < 0.05 and uf < 0.60:
                return False
            return True

        if r.name.startswith("Rover"):
            # rover can go anywhere it can traverse terrain-wise (you already allow)
            return True

        if r.name.startswith("Drone"):
            return True

        return True





    def step(self):
        self.timestep += 1
        if self.timestep % 50 == 0:
            self.assign_zones_cbba()

        union_belief = self.union_belief
        union_T = self.union_T
        union_R = self.union_R
        union = union_belief

        self.refresh_zone_tasks(union_belief)


        for rr in self.robots:
            if rr.active:
                rr.reveal_R = 3 if rr.role == Role.SCOUT else 2


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
                                # --- ROLE DECISION: once per zone per tick (or every few ticks) ---
        # --- ROLE DECISION ---
        zone_to_robots = {}
        for rr in self.robots:
            if rr.active and rr.task_zone is not None:
                zone_to_robots.setdefault(rr.task_zone, []).append(rr)

        for z, rs in zone_to_robots.items():
            zx, zy = z
            stats = self.compute_zone_stats_cached(zx, zy, union_belief, union_T, union_R)
            self.decide_roles_in_zone(z, rs, stats)

        relay_first = [r for r in self.robots if r.active and r.role == Role.RELAY]
        others      = [r for r in self.robots if r.active and r.role != Role.RELAY]

        def process_robot(r):
            if not r.active:
                return

            # ---- comms merge (terrain + hazards) ----
            uTerr = self.union_belief
            uT    = self.union_T
            uR    = self.union_R

            recv = (r.terrain_belief == T_UNKNOWN) & (uTerr != T_UNKNOWN)
            if np.any(recv):
                r.terrain_belief[recv] = uTerr[recv]
                r.temp_belief[recv]    = uT[recv]
                r.rad_belief[recv]     = uR[recv]
                r.known_mask[recv]     = True
                r.recompute_chunked()



            # ---- detect survivors ----
            for s in self.survivors:
                if s not in self.found and abs(r.pos[0] - s[0]) <= 2 and abs(r.pos[1] - s[1]) <= 2:
                    self.found.add(s)

            # ---- reachable set (BFS) using BELIEF ----
            if r.reachable_refresh > 0 and r.reachable_cache is not None:
                reachable = r.reachable_cache
                r.reachable_refresh -= 1
            else:
                dq = deque([r.pos])
                reachable = {r.pos}
                while dq:
                    u = dq.popleft()
                    for v in self.world.neighbours(u):
                        if v in reachable:
                            continue

                        tb = r.terrain_belief[v[0], v[1]]

                        # --- MATCH A* RULES so "reachable" means "A* reachable" ---
                        # Land robots: avoid UNKNOWN cells adjacent to known water
                        if (Capability.LAND in r.caps) and (Capability.AIR not in r.caps) and (Capability.WATER not in r.caps):
                            if tb is T_UNKNOWN:
                                water_adj = any(r.terrain_belief[nn[0], nn[1]] is T_WATER for nn in self.world.neighbours(v))
                                if water_adj:
                                    continue

                        # Boats: only allow UNKNOWN if adjacent to known water
                        if (Capability.WATER in r.caps) and (Capability.AIR not in r.caps):
                            if tb is T_UNKNOWN:
                                water_adj = any(r.terrain_belief[nn[0], nn[1]] is T_WATER for nn in self.world.neighbours(v))
                                if not water_adj:
                                    continue

                        if not traversable_code(int(tb), caps_to_mask(r.caps)):
                            continue

                        # NEW: shadow feasibility must match A*
                        if self.radio_shadow[v[0], v[1]]:
                            z = self.cell_to_zone(v[0], v[1])
                            if not self.relay_ok_extended(z):
                                continue

                        reachable.add(v)
                        dq.append(v)


                r.reachable_cache = reachable
                r.reachable_refresh = 12   # <= tune: 8–20

                    
            frontiers_in_zone = []
            frontiers = []
            # ---- global frontiers cached/rebuilt (ALWAYS) ----
            if r.frontier_refresh > 0 and r.cached_frontiers is not None:
                frontiers = r.cached_frontiers
                r.frontier_refresh -= 1
            else:
                # Build frontiers by iterating reachable only (FAST)
                frontiers = []
                is_boat = r.name.startswith("Boat") and (Capability.WATER in r.caps) and (Capability.AIR not in r.caps)

                for (x, y) in reachable:
                    if (x, y) == r.pos:
                        continue

                    if is_boat:
                        if union[x, y] == T_WATER and any(union[nx, ny] == T_UNKNOWN for (nx, ny) in self.world.neighbours((x, y))):
                            frontiers.append((x, y))
                    else:
                        if union[x, y] == T_UNKNOWN:
                            frontiers.append((x, y))


                r.cached_frontiers = frontiers
                r.frontier_refresh = 12

            # If no unknown-reachable cells remain, target boundary-known cells that touch unknown
            if not frontiers:
                boundary = []
                for (x, y) in reachable:
                    if union[x, y] == T_UNKNOWN:
                        continue
                    # if any neighbor is unknown, this is a "reveal frontier"
                    if any(union[nx, ny] == T_UNKNOWN for (nx, ny) in self.world.neighbours((x, y))):
                        boundary.append((x, y))
                frontiers = boundary
                r.cached_frontiers = frontiers
                r.frontier_refresh = 12




            if r.task_zone is None:
                best = None
                best_cov = 1e9
                for z in r.assigned_zones:
                    if self.zone_is_blacklisted(r, z):
                        continue

                    zx, zy = z
                    stats = self.compute_zone_stats_cached(zx, zy, union_belief, union_T, union_R)
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
                stats = self.compute_zone_stats_cached(zx, zy, union_belief, union_T, union_R)


                cov = self.zone_coverage(union_belief, r.task_zone)

                if cov >= self.ZONE_DONE:
                    self.release_zone(r, reason="complete")
                elif not self.zone_feasible(r, stats):
                    self.release_zone(r, reason="unsuitable")
                elif (not lease_active) and (r.task_no_progress >= self.NO_PROGRESS_K):
                    self.blacklist_zone(r, r.task_zone, reason="no_progress")
                elif r.task_zone is not None:
                    # if zone has no reachable frontiers for this robot, drop it
                    z_front = self.zone_frontiers(union, reachable, r.task_zone)
                    if not z_front:
                        self.release_zone(r, reason="no_frontier_reachable")

                    
                # ---- frontiers in zone ----
                frontiers_in_zone = []
                if r.task_zone is not None:
                    frontiers_in_zone = self.zone_frontiers(union, reachable, r.task_zone)


            # ---- candidates ALWAYS defined ----
            candidates = frontiers_in_zone if frontiers_in_zone else frontiers

            if not candidates:
                # Only allow "idle" if LOITER in a designated zone
                if r.task_zone is not None and r.role in (Role.LOITER, Role.RELAY):
                    tgt = self.choose_zone_goal(r, r.task_zone, union, reachable)
                    if tgt is not None:
                        r.set_goal(tgt)
                        r.stuck_steps = 0
                        # let it keep moving/loitering in-zone
                    else:
                        r.goal = None
                        r.path = []
                        r.stuck_steps = 0
                        return
                else:
                    r.goal = None
                    r.path = []
                    r.stuck_steps = 0
                    return

            remaining = len(self.survivors) - len(self.found)
            global_cov = np.mean(union != T_UNKNOWN)

            # after computing frontiers_in_zone
            r.zone_frontier_count = len(frontiers_in_zone)
            r.zone_frontier_signal = min(1.0, len(frontiers_in_zone) / 25.0)  # 25 is a tunable "enough frontiers" scale


            ENDGAME = (remaining <= 2) or (global_cov > 0.97)

            if ENDGAME:
                # force all active robots to chase any remaining reachable unknowns
                candidates = [c for c in frontiers if self.timestep >= r.failed_goals.get(c, 0)]



            def frontier_score(c):
                dist = heuristic(r.pos, c)
                cx, cy = c[0] // CHUNK_SIZE, c[1] // CHUNK_SIZE
                λT = r.chunked[0, cx, cy]
                λR = r.chunked[1, cx, cy]
                eT = λT / max(1e-6, r.temp_limit)
                eR = λR / max(1e-6, r.rad_limit)
                e  = max(eT, eR)


                if r.name.startswith("Rover"):
                    score = dist - ALPHA * (e ** P)   # SEEK small, e.g. 2–10
                else:
                    score = dist + ALPHA * (e ** P)

                neigh = list(self.world.neighbours(c))
                neigh_terr = [union[nx, ny] for (nx, ny) in neigh]

                if r.name.startswith("Boat"):
                    # prioritize unknown cells that touch known water
                    if T_WATER in neigh_terr:
                        score *= 0.6   # strong pull toward shore exploration

                if r.name.startswith("Legged"):
                    # prioritize unknown cells that touch stairs
                    if T_STAIRS in neigh_terr:
                        score *= 0.6   # strong pull toward buildings/bridges

                if r.name.startswith("Drone"):
                    # drones are good scouts: prefer unknown near stairs/water too
                    if (T_STAIRS in neigh_terr) or (T_WATER in neigh_terr):
                        score *= 0.8
                return score


            need_new_goal = (r.goal is None) or (r.pos == r.goal)
            failing = (r.stuck_steps > 10) or (r.goal is None)

            # Only change goal when allowed
            should_pick_new_goal = (r.goal is None) or (r.pos == r.goal) or (r.goal_commit == 0 and r.stuck_steps > 10)

            if should_pick_new_goal:
                candidates = [c for c in candidates if self.timestep >= r.failed_goals.get(c, 0)]
                if not candidates:
                    r.goal = None; r.path = []; r.stuck_steps = 0
                    return

                if r.task_zone is not None:
                    tgt = self.choose_zone_goal(r, r.task_zone, union, reachable)

                    if tgt is None:
                        self.release_zone(r, reason="no_frontier")
                        return

                    if self.zone_coverage(union, r.task_zone) >= self.ZONE_DONE:
                        self.release_zone(r, reason="complete")
                        return
                else:
                    if not r.cached_frontiers:
                        r.goal = None
                        r.path = []
                        return
                    tgt = min(r.cached_frontiers, key=lambda c: heuristic(r.pos, c))


                if tgt is None:
                    r.goal = None; r.path = []; r.stuck_steps = 0
                    return

                r.set_goal(tgt)
                r.stuck_steps = 0
                
            



            old_dist = heuristic(r.pos, r.goal) if r.goal is not None else None
            r.step()

            if r.task_zone is not None:
                # measure progress as known cells in that zone (using union)
                cov = self.zone_coverage(union_belief, r.task_zone)
                known_now = int(cov * 10000)  # cheap discretization

                if r.task_last_known == 0:
                    r.task_last_known = known_now

                if known_now <= r.task_last_known:
                    r.task_no_progress += 1
                else:
                    r.task_no_progress = 0
                    r.task_last_known = known_now

            if r.goal is None:
                r.goal_commit = 0

            if r.active and r.goal is not None and old_dist is not None:
                new_dist = heuristic(r.pos, r.goal)
                r.stuck_steps = r.stuck_steps + 1 if new_dist >= old_dist else 0

        for r in self.robots:
            if r.active and r.role == Role.RELAY:
                process_robot(r)

        # 2) recompute relay_ok AFTER relays have moved/anchored
        # --- relay_ok with neighbor-zone permeability ---
        raw_ok = {}
        for zx in range(self.zone_nx):
            for zy in range(self.zone_ny):
                raw_ok[(zx, zy)] = self.zone_has_outside_relay((zx, zy))

        relay_ok = {}
        for z in raw_ok:
            ok = raw_ok[z]
            if not ok:
                # allow help from NBR4 neighbors
                for nz in self.zone_neighbors4(z):
                    if raw_ok.get(nz, False):
                        ok = True
                        break
            relay_ok[z] = ok

        self.relay_ok = relay_ok

        ## Late game strange auctioning
        ## Are the robots moving while relaying or do we just not turn the circle off?

        # 3) move everyone else (they now plan with correct relay_ok)
        for r in self.robots:
            if r.active and r.role != Role.RELAY:
                process_robot(r)

        if len(self.found) == len(self.survivors):
            return False

        # update union caches ONCE per tick (end)
        self.union_belief = self.get_union_terrain_belief()
        self.union_T      = self.get_union_temp_belief()
        self.union_R      = self.get_union_rad_belief()

        any_alive = any(rr.active and rr.battery > 0 for rr in self.robots)
        if not any_alive and len(self.found) < len(self.survivors):
            print(f"[STOP] All robots dead at t={self.timestep}, survivors missing={len(self.survivors)-len(self.found)}")
        return any_alive
        
    def compute_zone_stats_cached(self, zx, zy, union_belief, union_T, union_R):
        # Cache key must change when union changes (timestep is fine because you refresh union each tick)
        key = self.timestep
        if self._zone_stats_cache_key != key:
            self._zone_stats_cache_key = key
            self._zone_stats_cache = {}

        z = (zx, zy)
        if z in self._zone_stats_cache:
            return self._zone_stats_cache[z]

        # IMPORTANT: call the *non-cached* version
        stats = self.compute_zone_stats(zx, zy, union_belief, union_T, union_R)
        self._zone_stats_cache[z] = stats
        return stats


    
    def compute_zone_stats(self, zx, zy, union_belief, union_T, union_R):
        x0 = zx * self.zone_w_cells
        y0 = zy * self.zone_h_cells

        total = 0
        unknown = 0
        known = 0
        shadow = 0
        shadow_total = 0
        sumT = 0.0
        sumR = 0.0

        terrain_counts = {
            T_FREE: 0,
            T_WATER: 0,
            T_STAIRS: 0,
            T_OBS: 0,
        }

        for x in range(x0, min(x0 + self.zone_w_cells, GRID_W)):
            for y in range(y0, min(y0 + self.zone_h_cells, GRID_H)):
                total += 1
                tb = union_belief[x, y]
                if tb == T_UNKNOWN:
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

                if union_belief[x, y] != T_UNKNOWN:
                    if self.radio_shadow[x, y]:
                        shadow_total += 1

        unknown_frac = (unknown / total) if total > 0 else 0.0

        if known > 0:
            avgT = sumT / known
            avgR = sumR / known
            f_water  = terrain_counts[T_WATER] / known
            f_stairs = terrain_counts[T_STAIRS] / known
            f_free   = terrain_counts[T_FREE] / known
        else:
            avgT = avgR = 0.0
            f_water = f_stairs = f_free = 0.0

        cx = x0 + self.zone_w_cells // 2
        cy = y0 + self.zone_h_cells // 2

        shadow_frac_total = (shadow_total / total) if total > 0 else 0.0
        shadow_frac_known = (shadow / known) if known > 0 else 0.0

        return {
            "unknown_frac": unknown_frac,
            "avgT": avgT,
            "avgR": avgR,
            "f_water": f_water,
            "f_stairs": f_stairs,
            "f_free": f_free,
            "center": (cx, cy),
            "known": known,
            "total": total,
            "shadow_frac": shadow_frac_known,
            "shadow_frac_total": shadow_frac_total
        }
    
    def release_zone(self, r, reason="", update_task_status=True):
        z = r.task_zone
        if z is None:
            return

        if update_task_status and z in self.zone_tasks:
            t = self.zone_tasks[z]
            t.last_owner = r.name
            if r.name in t.owners:
                t.owners.remove(r.name)
            t.last_release_reason = reason

            # if nobody left holding it, mark released
            if len(t.owners) == 0:
                t.status = "released"
                t.expires_at = 0
            else:
                t.status = "held"


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
            t.last_owner = r.name
            if r.name in t.owners:
                t.owners.remove(r.name)
            t.last_release_reason = reason




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


        #shadow
        sf = zone_stats.get("shadow_frac_total", 0.0)
        shadow_bonus = 0.8 * sf   # tune: 0.4–1.5

        # travel
        dist = heuristic(r.pos, (cx, cy))
        travel_cost = dist / float(GRID_W + GRID_H)
        if r.name.startswith("Legged"):
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
        if r.name.startswith(("Legged", "Drone")):
            risk_affinity = -(abs(lam_sig) ** P)
        elif r.name.startswith("Rover"):
            risk_affinity = +(abs(lam_sig) ** P)
        else:
            risk_affinity = 0.0

        # terrain affinity
        terrain_affinity = 0.0
        if r.name.startswith("Boat"):
            terrain_affinity += 2.0 * f_water
            terrain_affinity -= 0.5 * f_free
        elif r.name.startswith("Legged"):
            terrain_affinity += 2.0 * f_stairs
            terrain_affinity -= 1.0 * f_water
        elif r.name.startswith("Drone"):
            terrain_affinity += 1.0 * f_water
            terrain_affinity += 0.5 * f_stairs
        elif r.name.startswith("Rover"):
            terrain_affinity -= 0.5 * f_water

        # critical tendering
        critical_bonus = 0.0
        if r.name.startswith(("Legged", "Drone")):
            critical_bonus += 1.5 * f_stairs
        if r.name.startswith("Boat"):
            critical_bonus += 1.5 * f_water

        # info gain
        info_gain = unknown_frac

        # combine
        base_u = (
            w_info * info_gain +
            w_risk * risk_affinity +
            w_terr * terrain_affinity -
            lambda_cost * travel_cost + shadow_bonus
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
                task.owners = []
                task.status = "released"
                task.expires_at = 0
    
    def zone_is_held(self, zone):
        t = self.zone_tasks.get(zone)
        if t is None:
            return False
        return (t.status == "held") and (self.timestep < t.expires_at) and (len(t.owners) > 0)
    
    def zone_bid(self, r, zone, union_belief, union_T, union_R):
        zx, zy = zone
        stats = self.compute_zone_stats_cached(zx, zy, union_belief, union_T, union_R)

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
        if t and t.status == "held" and self.timestep < t.expires_at:
            # allow bidding if there is capacity remaining OR already one of the holders
            if (r.name not in t.owners) and (len(t.owners) >= self.ZONE_CAPACITY):
                return None, {"zone": zone, "u": -1e9, "skip": f"held_full"}


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
        if zone is None or not reachable:
            return None

        frontiers = self.zone_frontiers(union, reachable, zone, robot=r)

        zx, zy = zone
        stats = self.compute_zone_stats_cached(zx, zy, self.union_belief, self.union_T, self.union_R)
        cx, cy = stats["center"]

        def nearest_to_center(exclude=None):
            pool = reachable
            if exclude is not None:
                pool = [p for p in reachable if p != exclude]
            if not pool:
                return None
            return min(pool, key=lambda p: heuristic(p, (cx, cy)))

        # If there are no frontiers in-zone, don't crash—go to zone anchor.
        if not frontiers:
            return nearest_to_center()

        if r.role == Role.LOITER:
            tgt = nearest_to_center(exclude=r.pos)
            return tgt
        
        if r.role == Role.RELAY:
            # Anchor: reachable, outside shadow, adjacent to shadow
            anchors = []
            for p in reachable:
                x, y = p
                if self.radio_shadow[x, y]:
                    continue
                if any(self.radio_shadow[nx, ny] for (nx, ny) in self.world.neighbours(p)):
                    anchors.append(p)

                if anchors:
                    zx, zy = zone

                    def zone_shadow_need(z):
                        # how badly does this zone need relay help? use shadow fraction
                        zsx, zsy = z
                        st = self.compute_zone_stats_cached(zsx, zsy, self.union_belief, self.union_T, self.union_R)
                        sf = st.get("shadow_frac_total", 0.0)
                        return sf

                    def relay_score(p):
                        x, y = p

                        # prefer being near zone borders (so you can "support" neighbors)
                        x0 = zx * self.zone_w_cells
                        y0 = zy * self.zone_h_cells
                        x1 = x0 + self.zone_w_cells - 1
                        y1 = y0 + self.zone_h_cells - 1

                        dist_to_border = min(x - x0, x1 - x, y - y0, y1 - y)  # 0 means on border
                        border_bonus = -2.0 * dist_to_border  # lower dist => better

                        # prefer positions that help MORE neighboring shadow zones
                        help = 0.0
                        for dx, dy in NBR4:
                            nz = (zx + dx, zy + dy)
                            if 0 <= nz[0] < self.zone_nx and 0 <= nz[1] < self.zone_ny:
                                help += zone_shadow_need(nz)

                        # also stay somewhat close to zone center so you don’t drift off
                        center_cost = 0.25 * heuristic(p, (cx, cy))

                        return center_cost + border_bonus - 4.0 * help

                    return min(anchors, key=relay_score)


            # fallback: nearest to center outside shadow
            pool = [p for p in reachable if not self.radio_shadow[p[0], p[1]]]
            if pool:
                return min(pool, key=lambda p: heuristic(p, (cx, cy)))
            return None



        if r.role == Role.SCOUT:
            # pick frontier with many unknown neighbors (high info)
            def scout_score(c):
                unk = sum(1 for (nx, ny) in self.world.neighbours(c) if union[nx, ny] == T_UNKNOWN)
                return 0.6 * heuristic(r.pos, c) - 2.0 * unk
            return min(frontiers, key=scout_score)

        # default SCAN: your existing dist+risk idea (keep it simple)
        def scan_score(c):
            dist = heuristic(r.pos, c)

            cx2, cy2 = c[0] // CHUNK_SIZE, c[1] // CHUNK_SIZE
            lamT = r.chunked[0, cx2, cy2]
            lamR = r.chunked[1, cx2, cy2]
            eT = lamT / max(1e-6, r.temp_limit)
            eR = lamR / max(1e-6, r.rad_limit)
            e  = max(eT, eR)

            if r.name.startswith("Rover"):
                return dist - 10.0 * (e ** P)
            else:
                return dist + 10.0 * (e ** P)

        return min(frontiers, key=scan_score)


    
    def zone_bid_or_skip(self, r, zone, union_belief, union_T, union_R, zones_assigned_count, stats=None):
        if stats is None:
            zx, zy = zone
            stats = self.compute_zone_stats(zx, zy, union_belief, union_T, union_R)


        # completion
        cov = self.zone_coverage(union_belief, zone)
        if cov >= self.ZONE_DONE:
            return None, {"zone": zone, "u": -1e9, "skip": "complete"}

        # blacklist
        if self.zone_is_blacklisted(r, zone):
            return None, {"zone": zone, "u": -1e9, "skip": "blacklisted"}

        t = self.zone_tasks.get(zone)
        if t and t.status == "held" and self.timestep < t.expires_at:
            if (r.name not in t.owners) and (len(t.owners) >= self.ZONE_CAPACITY):
                return None, {"zone": zone, "u": -1e9, "skip": "held_full"}


        # feasibility
        if not self.zone_feasible(r, stats):
            return None, {"zone": zone, "u": -1e9, "skip": "infeasible"}

        # boat constraint (optional—if you still want it)
        if r.name.startswith("Boat") and stats.get("f_water", 0.0) <= 0.0 and stats.get("unknown_frac", 0.0) < 0.40:
            return None, {"zone": zone, "u": -1e9, "skip": "boat_no_water"}

        # normal bid breakdown
        bd = self.lambda_breakdown(
            r, stats, zones_assigned_count,
            w_info=1.0, w_risk=1.0, w_terr=1.0,
            lambda_cost=0.05, load_penalty=0.25
        )

        row = {"zone": zone, **bd, "skip": ""}
        return (float(bd["u"]), bd, stats), row
    
    def best_fallback_zone(self, r, union_belief, union_T, union_R, allow_held=False):
        """Pick the best feasible zone for a robot (even if utility is low)."""
        best_z = None
        best_u = -1e18

        zones_assigned_count = {rr.name: len(rr.bundle) for rr in self.robots}

        for z in [(zx, zy) for zx in range(self.zone_nx) for zy in range(self.zone_ny)]:
            # skip completed
            if self.zone_coverage(union_belief, z) >= self.ZONE_DONE:
                continue
            # skip blacklisted
            if self.zone_is_blacklisted(r, z):
                continue

            # respect holds unless allow_held
            t = self.zone_tasks.get(z)
            if (not allow_held) and t and t.status == "held" and self.timestep < t.expires_at and t.owners != r.name:
                continue

            stats = self.compute_zone_stats(z[0], z[1], union_belief, union_T, union_R)
            if not self.zone_feasible(r, stats):
                continue

            bd = self.lambda_breakdown(
                r, stats, zones_assigned_count,
                w_info=1.0, w_risk=1.0, w_terr=1.0,
                lambda_cost=0.05, load_penalty=0.25
            )
            u = float(bd["u"])
            if u > best_u:
                best_u = u
                best_z = z

        return best_z

    
    def role_utility(self, r: Robot, role: Role, zone, role_counts, stats):
        # stats from compute_zone_stats (unknown_frac, avgT, avgR, f_water, f_stairs, center...)
        uf = stats["unknown_frac"]
        eT = stats["avgT"] / max(1e-6, r.temp_limit)
        eR = stats["avgR"] / max(1e-6, r.rad_limit)

        risk = max(eT, eR)

        # robot-specific exploration potential (0 if no reachable frontiers)
        m = getattr(r, "zone_frontier_signal", 0.0)




        # --- info term ---
        if role == Role.SCOUT:
            U_info = 1.6 * uf
        elif role == Role.SCAN:
            U_info = 1.2 * uf
        elif role == Role.RELAY:
            U_info = 0.6 * uf
        else:  # LOITER
            U_info = 0.1 * uf

        # --- safety term (rovers less sensitive) ---
        if r.name.startswith("Rover"):
            U_safe = -0.2 * (risk**2)
        elif r.name.startswith("Boat"):  # boats ignore rad/temp in your limits, risk≈0
            U_safe = -0.1 * (risk**2)
        else:
            U_safe = -1.0 * (risk**2)

        # --- team congestion / coupling ---
        k = 0.7
        U_team = -k * role_counts.get(role, 0)

        # --- capability rules as soft penalties ---
        U_rule = 0.0
        if r.name.startswith("Boat") and role in (Role.RELAY, Role.LOITER):
            # boat relay/loiter can be ok if zone has water; else penalize
            if stats["f_water"] < 0.05 and stats["unknown_frac"] < 0.4:
                U_rule -= 2.0

        if r.name.startswith("Drone") and role == Role.SCOUT:
            U_info += 0.5  # drones are great scouts
        
        if role == Role.LOITER:
            U_rule -= 4.0 * m
            U_rule -= 1.5 * uf
        
        sf_total = stats.get("shadow_frac_total", 0.0)  # NEW
        sf_known  = stats.get("shadow_frac", 0.0)

        sf = max(sf_total, sf_known)

        if role == Role.RELAY:
            U_info += 4.0 * sf * (0.5 + 0.5 * m)

        
        relay_count = role_counts.get(Role.RELAY, 0)
        if relay_count == 0 and sf > 0.05 and role in (Role.SCOUT, Role.SCAN):
            U_rule -= 5.0 * sf * (0.5 + 0.5 * m)


        return U_info + U_safe + U_team + U_rule
    
    def decide_roles_in_zone(self, zone, robots_in_zone, stats):
        # initialize
        for r in robots_in_zone:
            if self.timestep < r.role_locked_until:
                continue
            r.role = Role.SCAN

        roles = [Role.SCOUT, Role.SCAN, Role.RELAY, Role.LOITER]

        for _ in range(3):  # 3 rounds best-response
            # count
            role_counts = {}
            for r in robots_in_zone:
                role_counts[r.role] = role_counts.get(r.role, 0) + 1

            for r in robots_in_zone:
                if self.timestep < r.role_locked_until:
                    continue

                # remove self from counts (so you don't punish yourself twice)
                role_counts[r.role] -= 1

                best_role = r.role
                best_u = -1e9
                for role in roles:
                    u = self.role_utility(r, role, zone, role_counts, stats)
                    if u > best_u:
                        best_u, best_role = u, role

                # assign + update counts
                r.role = best_role
                role_counts[r.role] = role_counts.get(r.role, 0) + 1

                # lock to prevent oscillation
                r.role_locked_until = self.timestep + 15

    def build_radio_shadow(self):
        rs = np.zeros((GRID_W, GRID_H), dtype=bool)

        # 1) Building shadow: STAIRS are indoor (house + bridges in your world)
        for x in range(GRID_W):
            for y in range(GRID_H):
                if self.world.grid[x][y].true_terrain == T_STAIRS:
                    rs[x, y] = True

        # 2) One big random blob somewhere on the map
        cx = random.randint(GRID_W//5, GRID_W - GRID_W//5)
        cy = random.randint(GRID_H//5, GRID_H - GRID_H//5)
        rad = random.randint(14, 26)  # blob size

        r2 = rad * rad
        for x in range(max(0, cx-rad), min(GRID_W, cx+rad+1)):
            for y in range(max(0, cy-rad), min(GRID_H, cy+rad+1)):
                dx, dy = x - cx, y - cy
                if dx*dx + dy*dy <= r2:
                    rs[x, y] = True

        # 3) Hard guarantee: if somehow still empty, force a small center block
        if not np.any(rs):
            mx, my = GRID_W//2, GRID_H//2
            rs[mx-3:mx+4, my-3:my+4] = True

        self.radio_shadow = rs



    def zone_has_outside_relay(self, zone):
        if zone is None:
            return False

        for r in self.robots:
            if not r.active:
                continue
            if r.role != Role.RELAY:
                continue

            x, y = r.pos
            if self.cell_to_zone(x, y) != zone:
                continue

            # relay must be OUTSIDE shadow
            if not self.radio_shadow[x, y]:
                return True

        return False
    
    def zone_neighbors4(self, z):
        zx, zy = z
        out = []
        for dx, dy in NBR4:  # you already have NBR4 = ((1,0),(-1,0),(0,1),(0,-1))
            nz = (zx + dx, zy + dy)
            if 0 <= nz[0] < self.zone_nx and 0 <= nz[1] < self.zone_ny:
                out.append(nz)
        return out
    
    def cell_near_shared_border(self, x, y, z, nz, border=3):
        """
        True if cell (x,y) in zone z is within 'border' cells of the border shared with neighbor zone nz.
        Only handles 4-neighbor zones (no diagonals).
        """
        zx, zy = z
        nx, ny = nz

        x0 = zx * self.zone_w_cells
        y0 = zy * self.zone_h_cells
        x1 = x0 + self.zone_w_cells - 1
        y1 = y0 + self.zone_h_cells - 1

        # neighbor is LEFT of z
        if nx == zx - 1 and ny == zy:
            return (x - x0) < border
        # neighbor is RIGHT of z
        if nx == zx + 1 and ny == zy:
            return (x1 - x) < border
        # neighbor is UP of z
        if nx == zx and ny == zy - 1:
            return (y - y0) < border
        # neighbor is DOWN of z
        if nx == zx and ny == zy + 1:
            return (y1 - y) < border

        return False
    
    def relay_ok_extended(self, z):
        if z is None:
            return False
        if self.relay_ok.get(z, False):
            return True
        zx, zy = z
        for dx, dy in NBR4:
            zz = (zx + dx, zy + dy)
            if self.relay_ok.get(zz, False):
                return True
        return False



def build_grid_surface(sim, show_map, show_survivors, show_heat, show_rad, show_risk,
                       union_belief, union_T, union_R):
    world = sim.world
    surf = pygame.Surface((world.w * CELL_SIZE, world.h * CELL_SIZE))

    # Precompute risk overlay only if needed
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
                        if union_belief[x, y] != T_UNKNOWN:
                            t = union_T[x, y]
                            r = union_R[x, y]
                            if not np.isnan(t): valsT.append(float(t))
                            if not np.isnan(r): valsR.append(float(r))

                if valsT or valsR:
                    chunk_known[cx, cy] = 1
                    chunk_max_T[cx, cy] = max(valsT) if valsT else 0.0
                    chunk_max_R[cx, cy] = max(valsR) if valsR else 0.0

        risk_map = 10.0 * chunk_max_T + 10.0 * chunk_max_R
        max_risk = float(np.max(risk_map)) if float(np.max(risk_map)) > 1e-9 else 1.0

    # Draw cells ONCE into surf
    for x in range(world.w):
        for y in range(world.h):
            pos = (x, y)

            if pos in sim.found or (show_survivors and pos in sim.survivors):
                clr = SURVIVOR_COLOUR

            elif show_risk:
                cx, cy = x // CHUNK_SIZE, y // CHUNK_SIZE
                if chunk_known[cx, cy] == 0:
                    clr = (180, 180, 180)
                else:
                    risk_norm = min(max(risk_map[cx, cy] / max_risk, 0.0), 1.0)
                    clr = (int(255 * risk_norm), int(255 * (1.0 - risk_norm)), 0)

            elif show_rad:
                rr = world.grid[x][y].radiation
                if world.grid[x][y].true_terrain == T_WATER:
                    clr = (0, 0, 0)
                else:
                    v = min(rr / 100.0, 1.0)
                    clr = (0, int(255 * v), 0)

            elif show_heat:
                tt = world.grid[x][y].temperature
                t_norm = min(max(tt / 200.0, 0.0), 1.0)
                clr = (int(255 * t_norm), 0, int(255 * (1.0 - t_norm)))

            terr = world.grid[x][y].true_terrain if show_map else union_belief[x, y]
            # if show_map -> terr is Terrain enum; else -> terr is uint8 code

            clr = TERRAIN_COLOUR_CODE[int(terr)]


            rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(surf, clr, rect)

    # Optional: draw faint grid lines MUCH cheaper: every 4 cells instead of every cell
    step = 4 * CELL_SIZE
    wpx = world.w * CELL_SIZE
    hpx = world.h * CELL_SIZE
    line_col = (170, 170, 170)
    for px in range(0, wpx, step):
        pygame.draw.line(surf, line_col, (px, 0), (px, hpx), 1)
    for py in range(0, hpx, step):
        pygame.draw.line(surf, line_col, (0, py), (wpx, py), 1)

    return surf

def draw_grid(scr, sim, show_zones):
    # only zones overlay here (map itself is blitted from a buffer surface)
    if not show_zones:
        return

    for zx in range(sim.zone_nx):
        for zy in range(sim.zone_ny):
            x0 = zx * sim.zone_w_cells
            y0 = zy * sim.zone_h_cells

            rect = pygame.Rect(
                x0 * CELL_SIZE,
                y0 * CELL_SIZE,
                sim.zone_w_cells * CELL_SIZE,
                sim.zone_h_cells * CELL_SIZE
            )

            # Get owners from zone_tasks (capacity-aware)
            t = sim.zone_tasks.get((zx, zy))
            owner_names = list(t.owners) if (t is not None and hasattr(t, "owners")) else []

            # Resolve owner robots (keep stable order)
            owner_robots = []
            for nm in owner_names:
                rr = next((r for r in sim.robots if r.name == nm), None)
                if rr is not None:
                    owner_robots.append(rr)

            if len(owner_robots) == 0:
                # unowned
                pygame.draw.rect(scr, (120, 120, 120), rect, 1)

            elif len(owner_robots) == 1:
                # single owner
                c1 = ROBOT_COLOUR[robot_type(owner_robots[0].name)]
                pygame.draw.rect(scr, c1, rect, 2)

            else:
                # two owners: outer outline + inner outline
                c1 = ROBOT_COLOUR[robot_type(owner_robots[0].name)]
                c2 = ROBOT_COLOUR[robot_type(owner_robots[1].name)]

                # outer
                pygame.draw.rect(scr, c1, rect, 3)

                # inner inset rectangle
                inset = 6  # pixels; adjust to taste
                inner = rect.inflate(-inset, -inset)
                if inner.width > 0 and inner.height > 0:
                    pygame.draw.rect(scr, c2, inner, 3)

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
        pygame.draw.rect(scr, ROBOT_COLOUR[robot_type(r.name)], rect, 4)


            

def draw_robots(scr, robots, show_plans= False):
    if show_plans:
        for r in robots:
            # lighter color for path cells
            base_clr = ROBOT_COLOUR[robot_type(r.name)]
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
        clr = ROBOT_COLOUR[robot_type(r.name)]
        pygame.draw.rect(scr, clr, (x*CELL_SIZE, y*CELL_SIZE, CELL_SIZE, CELL_SIZE))

        if getattr(r, "role", None) == Role.RELAY:
                cx = x*CELL_SIZE + CELL_SIZE//2
                cy = y*CELL_SIZE + CELL_SIZE//2

                # halo ring
                pygame.draw.circle(scr, (255, 255, 0), (cx, cy), 8, 2)  # yellow ring

def build_shadow_surface(sim):
    """Precompute a semi-transparent overlay for radio shadow cells."""
    world = sim.world
    surf = pygame.Surface((world.w * CELL_SIZE, world.h * CELL_SIZE), pygame.SRCALPHA)

    # RGBA: dark tint with alpha
    shadow_col = (40, 40, 40, 110)

    rs = sim.radio_shadow  # bool grid
    for x in range(world.w):
        for y in range(world.h):
            if rs[x, y]:
                rect = pygame.Rect(x*CELL_SIZE, y*CELL_SIZE, CELL_SIZE, CELL_SIZE)
                surf.fill(shadow_col, rect)

    return surf


def gui_loop():
    global sim
    pygame.init()
    screen = pygame.display.set_mode((GRID_W*CELL_SIZE + SIDEBAR_WIDTH, GRID_H*CELL_SIZE))
    pygame.display.set_caption("Heterogeneous Robot Fleet Simulator")
    font = pygame.font.SysFont(None, 24)
    sim   = FleetSim()

    shadow_surface = build_shadow_surface(sim)
    shadow_dirty = False

    running = False
    show_map = False
    show_survivors = False
    show_heat = False
    show_rad = False
    show_risk = False
    show_plans = False
    show_zones = False
    show_shadow = False
    time_step = 0
    show_lambda = False
    grid_surface = None
    last_view_key = None
    last_union_step = -1

    
    lambda_btn = pygame.Rect(GRID_W*CELL_SIZE+10, 155, 120, 20)
    start_btn = pygame.Rect(GRID_W*CELL_SIZE+10, 10, 80, 20)
    map_btn   = pygame.Rect(GRID_W*CELL_SIZE+10, 35, 100, 20)
    surv_btn  = pygame.Rect(GRID_W*CELL_SIZE+10, 60, 100, 20)
    #heat_btn  = pygame.Rect(GRID_W*CELL_SIZE+10,85,100,20)
    #rad_btn   = pygame.Rect(GRID_W*CELL_SIZE+10,110,100,20)
    risk_btn  = pygame.Rect(GRID_W*CELL_SIZE+10, 85, 120, 20)
    plans_btn = pygame.Rect(GRID_W*CELL_SIZE+10, 110,120,20)
    zones_btn = pygame.Rect(GRID_W*CELL_SIZE+10, 130,120,20)
    shadow_btn = pygame.Rect(GRID_W*CELL_SIZE+10, 175, 120, 20)  # below λ button




    clock = pygame.time.Clock()

    def view_key():
        return (show_map, show_survivors, show_heat, show_rad, show_risk)

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
                        show_heat  = False
                        show_rad   = False
                if lambda_btn.collidepoint(ev.pos):
                    show_lambda = not show_lambda
                if shadow_btn.collidepoint(ev.pos):
                    show_shadow = not show_shadow
                    if show_shadow:
                        show_risk  = False
                        show_heat  = False
                        show_rad   = False

        if running:
            # 1 sim step per frame for smooth 60fps visuals
            if not sim.step():
                running = False
                print("Exploration Complete!")
                if sim.dead_robots:
                    print("Robots incapacitated during run:")
                    for name, reason in sim.dead_robots:
                        print(f" - {name} stopped due to {reason}")
                    # no break needed here
            time_step += 1


        union_belief = sim.union_belief
        union_T = sim.union_T
        union_R = sim.union_R


        vk = view_key()

        # rebuild buffer if:
        # - first time
        # - view toggles changed
        # - union changed AND we're not showing true map/heat/rad/risk (i.e., union matters)
        union_matters = (not show_map) and (not show_heat) and (not show_rad) and (not show_risk)
        union_changed = (sim.timestep != last_union_step)

        if (grid_surface is None) or (vk != last_view_key) or (union_matters and union_changed):
            grid_surface = build_grid_surface(sim, show_map, show_survivors, show_heat, show_rad, show_risk,
                                            union_belief, union_T, union_R)
            last_view_key = vk
            last_union_step = sim.timestep
        # draw base grid once
        screen.blit(grid_surface, (0, 0))

        # then shadow overlay on top (so it stays visible)
        if show_shadow:
            if shadow_dirty or shadow_surface is None:
                shadow_surface = build_shadow_surface(sim)
                shadow_dirty = False
            screen.blit(shadow_surface, (0, 0))

        # overlays
        draw_grid(screen, sim, show_zones)
        draw_robots(screen, sim.robots, show_plans)


        # draw_grid(screen, sim, show_map, show_survivors, show_heat, show_rad,
        #           show_risk, show_plans, show_zones, union_belief, union_T, union_R)
        # draw_robots(screen, sim.robots,show_plans)

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
            (shadow_btn, 'Hide Shadow' if show_shadow else 'Show Shadow'),

        ]:
            pygame.draw.rect(screen, (200,200,200), btn)
            screen.blit(font.render(label, True, (0,0,0)), (btn.x+10, btn.y+5))

        # stats
        screen.blit(font.render(f"Step: {time_step}", True, (0,0,0)),
                    (GRID_W*CELL_SIZE+10, plans_btn.y+50))
        total = sim.world.w * sim.world.h

        disc = int(np.sum(union_belief != T_UNKNOWN))

        pct   = disc/total*100
        screen.blit(font.render(f"Coverage: {pct:.1f}%", True, (0,0,0)),
                    (GRID_W*CELL_SIZE+10, plans_btn.y+80))
        
        y = plans_btn.y + 110
        screen.blit(font.render("Battery (avg by type):", True, (0,0,0)),
                    (GRID_W*CELL_SIZE+10, y)); y += 22

        groups = {"Legged": [], "Drone": [], "Boat": [], "Rover": []}
        alive_counts = {"Legged": 0, "Drone": 0, "Boat": 0, "Rover": 0}

        for r in sim.robots:
            t = robot_type(r.name)
            if t in groups:
                groups[t].append(r.battery)
                if r.active and r.battery > 0:
                    alive_counts[t] += 1

        for t in ("Legged", "Drone", "Boat", "Rover"):
            vals = groups[t]
            avg = (sum(vals) / len(vals)) if vals else 0.0
            alive = alive_counts[t]
            total = len(vals)
            screen.blit(font.render(f"{t}: {avg:.1f} ({alive}/{total} alive)", True, (0,0,0)),
                        (GRID_W*CELL_SIZE+10, y))
            y += 20



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