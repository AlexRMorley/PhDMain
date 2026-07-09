"""
2-D heterogeneous robot fleet exploration simulator with GUI controls.

Robots: Legged, Drone, Boat, Rover
Environment: grid cells with semantic terrain (STAIRS/WATER/RIVER/BRIDGE; rest FREE/OBSTACLE).
Planner: A* on each robot's known map, augmented with chunked risk.
GUI: pygame grid + sidebar controls.

Architecture (medium restructure):
  - Robot.tick()       : single entry point per robot per sim-step
  - FleetSim.step()    : orchestrates tick order, shared state, CBBA cadence
  - CBBA               : fixed consensus loop, stable bundle (no mid-task reset)
  - ZoneFrontierCache  : per-tick shared cache so robots don't re-BFS the same zone
  - Collision          : reservation table; blocked robot replans to offset goal
    # Do temperature and radiation and explore radial relaying (Speed issue)
  # Code has a deadlock at continuing after 2000 steps.
"""

import pygame
from enum import Enum
import heapq
import math
import random
import sys
from collections import deque
import numpy as np
from dataclasses import dataclass, field

# ── Configuration ────────────────────────────────────────────────────────────
CELL_SIZE     = 6
GRID_W        = 128
GRID_H        = 128
FPS           = 60
SIDEBAR_WIDTH = 260
MAX_BATTERY   = 1000

# Risk / planner
CHUNK_SIZE  = 2
ZONE_CHUNKS = 8          # zone = ZONE_CHUNKS * CHUNK_SIZE cells wide/tall
ALPHA       = 60.0
BETA        = 0.5
P           = 4
ZONE_DONE   = 0.98

# Hazard field
N_HOTSPOTS_TEMP  = 18;  TEMP_AMP_N = 30; TEMP_AMP_P = 0.35; TEMP_AMP_SCALE = 6.0
N_HOTSPOTS_RAD   = 18;  RAD_AMP_N  = 28; RAD_AMP_P  = 0.30; RAD_AMP_SCALE  = 7.0
SIGMA_MIN = 2.5;  SIGMA_MAX = 9.0
TEMP_LIMIT = 1000.0;  RAD_LIMIT = 1000.0

# Exploration planner shaping
UNK_PEN_SCOUT  = 0.25;  UNK_PEN_OTHER  = 0.70
INFO_W_SCOUT   = 0.35;  INFO_W_OTHER   = 0.15   # reward per unknown neighbour revealed
UNK_PRIOR      = 0.25
SOFT_FRAC      = 0.90
MIN_STEP_COST  = 0.05

# Traffic / anti-conga
TRAFFIC_LOOKAHEAD = 25
TRAFFIC_W         = 0.25

# CBBA / zone lease
CBBA_ITERS     = 3
MAX_BUNDLE     = 4
ZONE_CAPACITY  = 2
LEASE_T        = 50
COOLDOWN_T     = 80
NO_PROGRESS_K  = 30
IDLE_RESCUE_K  = 15   # ticks idle with no goal before forcing a CBBA rebid

# Relay
RELAY_MIN_HOLD     = 50        # ticks a relay must stay before being demoted
RELAY_SEP_RADIUS   = 10
RELAY_SEP_PENALTY  = 6.0
RELAY_COUNT_PENALTY= 2.5
RELAY_MAX_PER_ZONE = 1
RELAY_MAX_FLEET_FRAC = 0.30   # at most 30% of active fleet can be relays
RELAY_IDLE_TICKS   = 40       # demote relay if no robot has been inside bubble for this many ticks

# Comms — centralised planner model
# Robots broadcast instantly to base when in open air.
# Shadow robots with relay coverage get a small chain-latency delay.
RELAY_COMMS_DELAY  = 2        # ticks of latency through a relay chain
CONF_TAU           = 200      # ticks — e-folding time for scan confidence decay
CONF_UNCERTAIN     = 0.25     # below this confidence a cell is treated as uncertain
ZONE_PERSONAL_THRESH = 0.40   # fraction of a zone a robot must have personally
                               # scanned to use shared stats in CBBA

# ── Terrain codes ─────────────────────────────────────────────────────────────
T_UNKNOWN = 0; T_FREE = 1; T_OBS = 2; T_STAIRS = 3; T_WATER = 4; T_BRIDGE = 5

TERRAIN_COLOUR_CODE = {
    T_UNKNOWN: (200, 200, 200),
    T_FREE:    (255, 255, 255),
    T_OBS:     (  0,   0,   0),
    T_STAIRS:  (255, 255,   0),
    T_WATER:   (  0,   0, 255),
    T_BRIDGE:  (139,  69,  19),
}

# ── Capability bitmasks ───────────────────────────────────────────────────────
CAP_LAND   = 1 << 0
CAP_STAIRS = 1 << 1
CAP_WATER  = 1 << 2
CAP_AIR    = 1 << 3

class Capability(Enum):
    LAND = 1; STAIRS = 2; WATER = 3; AIR = 4

def caps_to_mask(caps: set) -> int:
    m = 0
    if Capability.LAND   in caps: m |= CAP_LAND
    if Capability.STAIRS in caps: m |= CAP_STAIRS
    if Capability.WATER  in caps: m |= CAP_WATER
    if Capability.AIR    in caps: m |= CAP_AIR
    return m

# Pre-built traversability lookup: _TRAV_LUT[terrain_code][caps_mask & 0xF] -> bool
def _build_trav_lut():
    lut = [[False]*16 for _ in range(8)]
    for tb in range(8):
        for mask in range(16):
            lut[tb][mask] = traversable_code(tb, mask)
    return lut

def traversable_code(tb: int, mask: int) -> bool:
    has_land   = bool(mask & CAP_LAND)
    has_stairs = bool(mask & CAP_STAIRS)
    has_water  = bool(mask & CAP_WATER)
    has_air    = bool(mask & CAP_AIR)
    if tb == T_OBS:                                              return False
    if tb == T_STAIRS and not has_stairs and not has_air:        return False
    if tb == T_WATER  and not has_water  and not has_air:        return False
    if tb == T_BRIDGE and not (has_land or has_water or has_air or has_stairs): return False
    if tb == T_FREE   and not has_land   and not has_air:        return False
    return True  # T_UNKNOWN allowed (A* handles separately)

_TRAV_LUT = _build_trav_lut()

# ── Roles ─────────────────────────────────────────────────────────────────────
class Role(Enum):
    SCOUT = 1; SCAN = 2; LOITER = 3; RELAY = 4

# ── Robot colours ─────────────────────────────────────────────────────────────
ROBOT_COLOUR = {
    "Legged": (  0, 255,   0),
    "Drone":  (255,   0, 255),
    "Boat":   (  0, 255, 255),
    "Rover":  (255, 165,   0),
}

NBR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

# Pre-computed neighbour offsets as a numpy array for fast vectorised ops
_NBR4_DX = np.array([1, -1, 0,  0], dtype=np.int16)
_NBR4_DY = np.array([0,  0, 1, -1], dtype=np.int16)

def robot_type(name: str) -> str:
    for t in ("Legged", "Drone", "Boat", "Rover"):
        if name.startswith(t): return t
    return name

# ─────────────────────────────────────────────────────────────────────────────
# Zone task bookkeeping
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ZoneTask:
    zone:       tuple
    owners:     list = field(default_factory=list)
    status:     str  = "free"      # free | held | released | blacklisted
    progress:   float = 0.0
    expires_at: int   = 0
    last_owner: str   = None
    last_release_reason: str = ""

# ─────────────────────────────────────────────────────────────────────────────
# World generation
# ─────────────────────────────────────────────────────────────────────────────
class GridWorld:
    def __init__(self, w, h):
        self.w, self.h = w, h
        self.grid = [[self._cell(T_FREE) for _ in range(h)] for _ in range(w)]
        self.river_centrelines = []   # list of centreline point lists, one per river
        self._generate()
        self._init_temperature()
        self._init_radiation()

    # ── internal cell struct (plain dict for speed) ──
    @staticmethod
    def _cell(terrain=T_FREE, temp=0.0, rad=0.0):
        return {"t": terrain, "temp": temp, "rad": rad}

    def terrain(self, x, y):  return self.grid[x][y]["t"]
    def temp(self,    x, y):  return self.grid[x][y]["temp"]
    def rad(self,     x, y):  return self.grid[x][y]["rad"]

    def neighbours(self, pos):
        x, y = pos
        for dx, dy in NBR4:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.w and 0 <= ny < self.h:
                yield (nx, ny)

    # ── terrain cost ──
    def cell_cost(self, x, y, caps_mask: int, terrain_override=None) -> float:
        t = terrain_override if terrain_override is not None else self.grid[x][y]["t"]
        if t == T_OBS:    return math.inf
        if t == T_STAIRS and not (caps_mask & (CAP_STAIRS | CAP_AIR)): return math.inf
        if t == T_WATER  and not (caps_mask & (CAP_WATER  | CAP_AIR)): return math.inf
        if t == T_FREE   and not (caps_mask & (CAP_LAND   | CAP_AIR)): return math.inf
        return 1.0

    # ── world generation helpers ──
    def _clamp(self, v, lo, hi): return max(lo, min(hi, v))

    def _pick_edge(self):
        s = random.randint(0, 3)
        if s == 0: return (0, random.randint(0, self.h-1))
        if s == 1: return (self.w-1, random.randint(0, self.h-1))
        if s == 2: return (random.randint(0, self.w-1), 0)
        return (random.randint(0, self.w-1), self.h-1)

    def _pick_other_edge(self, p0):
        x0, y0 = p0
        if   x0 == 0:        forbidden = 0
        elif x0 == self.w-1: forbidden = 1
        elif y0 == 0:        forbidden = 2
        else:                forbidden = 3
        while True:
            s = random.randint(0, 3)
            if s == forbidden: continue
            if s == 0: return (0, random.randint(0, self.h-1))
            if s == 1: return (self.w-1, random.randint(0, self.h-1))
            if s == 2: return (random.randint(0, self.w-1), 0)
            return (random.randint(0, self.w-1), self.h-1)

    def _ctrl_pts(self, p0, p1, n=4):
        """
        Generate smooth control points for a meandering river.
        Uses correlated perpendicular jitter so the river curves gently
        rather than zigzagging.  More control points = more natural bends.
        """
        x0, y0 = float(p0[0]), float(p0[1])
        x1, y1 = float(p1[0]), float(p1[1])
        vx, vy = x1-x0, y1-y0
        norm = math.sqrt(vx*vx + vy*vy) or 1.0
        px, py = -vy/norm, vx/norm        # perpendicular unit vector
        mag = 0.07 * min(self.w, self.h)  # max meander amplitude

        pts = [(x0, y0)]
        # Correlated jitter: each step is a random-walk drift in the perp direction
        # so the river meanders smoothly rather than bouncing randomly
        drift = 0.0
        for k in range(1, n+1):
            t = k / (n+1)
            drift += random.gauss(0, mag * 0.4)
            drift  = max(-mag, min(mag, drift))   # clamp drift
            bx = x0 + t*vx + drift*px
            by = y0 + t*vy + drift*py
            pts.append((self._clamp(bx, 8, self.w-9),
                        self._clamp(by, 8, self.h-9)))
        pts.append((x1, y1))
        return pts

    def _catmull_rom(self, pts, steps_per_seg=20):
        """Catmull-Rom spline — more steps per segment for smoother curves."""
        out = []
        p = [(2*pts[0][0]-pts[1][0], 2*pts[0][1]-pts[1][1])] + pts +             [(2*pts[-1][0]-pts[-2][0], 2*pts[-1][1]-pts[-2][1])]
        for i in range(1, len(p)-2):
            p0, p1, p2, p3 = p[i-1], p[i], p[i+1], p[i+2]
            for s in range(steps_per_seg):
                t = s / steps_per_seg
                t2, t3 = t*t, t*t*t
                x = 0.5*((2*p1[0]) + (-p0[0]+p2[0])*t +
                         (2*p0[0]-5*p1[0]+4*p2[0]-p3[0])*t2 +
                         (-p0[0]+3*p1[0]-3*p2[0]+p3[0])*t3)
                y = 0.5*((2*p1[1]) + (-p0[1]+p2[1])*t +
                         (2*p0[1]-5*p1[1]+4*p2[1]-p3[1])*t2 +
                         (-p0[1]+3*p1[1]-3*p2[1]+p3[1])*t3)
                out.append((x, y))
        return out

    def _paint_disc(self, cx, cy, radius):
        r2 = radius*radius
        for x in range(int(cx-radius-1), int(cx+radius+2)):
            for y in range(int(cy-radius-1), int(cy+radius+2)):
                if 0 <= x < self.w and 0 <= y < self.h:
                    if (x-cx)**2 + (y-cy)**2 <= r2:
                        self.grid[x][y]["t"] = T_WATER

    def _carve_river(self, base_width=5, n_ctrl=4):
        p0 = self._pick_edge()
        p1 = self._pick_other_edge(p0)
        pts = self._ctrl_pts(p0, p1, n_ctrl)
        samples = self._catmull_rom(pts, steps_per_seg=20)
        n = max(1, len(samples)-1)
        phase = random.uniform(0, math.pi)
        for i, (fx, fy) in enumerate(samples):
            t = i / n
            w = base_width * (0.85 + 0.3*math.sin(2*math.pi*t + phase))
            w += random.gauss(0, 0.2)
            self._paint_disc(int(round(fx)), int(round(fy)), max(2.5, w)/2.0)

        # Fill any non-water islands left inside the river body.
        # Flood-fill water outward from every edge water cell; any water cell
        # not reachable from the edge is interior — fill it too.  Then invert:
        # any non-water cell completely enclosed by water becomes water.
        self._fill_river_islands()

        # Store centreline (deduplicated integer points)
        seen = set(); cl = []
        for (fx, fy) in samples:
            p = (int(round(fx)), int(round(fy)))
            if p not in seen and 0 <= p[0] < self.w and 0 <= p[1] < self.h:
                seen.add(p); cl.append(p)
        self.river_centrelines.append(cl)
        return p0, p1

    def _fill_river_islands(self):
        """
        Clean up the river body after carving:
        1. Flood-fill enclosed islands (non-water cells not reachable from map edge → water).
        2. Morphological closing: remove peninsula stubs — any non-water cell with
           >= 3 water/bridge neighbours is a thin finger; convert it to water.
           Repeat until stable so chains of stubs are fully eroded.
        """
        W, H = self.w, self.h

        # Pass 1: flood-fill enclosed islands
        visited = np.zeros((W, H), dtype=bool)
        q = deque()
        for x in range(W):
            for y in [0, H-1]:
                if self.grid[x][y]["t"] != T_WATER and not visited[x, y]:
                    visited[x, y] = True; q.append((x, y))
        for y in range(H):
            for x in [0, W-1]:
                if not visited[x, y] and self.grid[x][y]["t"] != T_WATER:
                    visited[x, y] = True; q.append((x, y))
        while q:
            x, y = q.popleft()
            for dx, dy in NBR4:
                nx, ny = x+dx, y+dy
                if 0 <= nx < W and 0 <= ny < H and not visited[nx, ny]:
                    if self.grid[nx][ny]["t"] != T_WATER:
                        visited[nx, ny] = True; q.append((nx, ny))
        for x in range(W):
            for y in range(H):
                if not visited[x, y] and self.grid[x][y]["t"] != T_WATER:
                    self.grid[x][y]["t"] = T_WATER

        # Pass 2: erode peninsula stubs and orphaned obstacles (repeat until stable)
        for _ in range(6):
            changed = False
            for x in range(1, W-1):
                for y in range(1, H-1):
                    t = self.grid[x][y]["t"]
                    if t in (T_WATER, T_BRIDGE):
                        continue
                    water_nbrs = sum(
                        1 for dx, dy in NBR4
                        if self.grid[x+dx][y+dy]["t"] in (T_WATER, T_BRIDGE))
                    if water_nbrs >= 3:
                        self.grid[x][y]["t"] = T_WATER
                        changed = True
            if not changed:
                break

    def _place_bridges(self, n_bridges=2, bridge_w=5, min_spacing=40):
        """
        Place axis-aligned bridges perpendicular to the river's dominant flow.

        For each river centreline:
          1. Compute the river's dominant flow axis (horizontal vs vertical)
             from the average direction of the centreline.
          2. The bridge runs on the PERPENDICULAR axis:
             - Mostly-horizontal river  → bridge is a vertical column of cells
             - Mostly-vertical river    → bridge is a horizontal row of cells
          3. Scan every candidate row/column along that perpendicular axis,
             find the narrowest water crossing with clear land banks on both sides.
          4. Pick up to n_bridges spaced >= min_spacing apart.

        Result: always a clean rectangular bridge stamp, never diagonal.
        """
        W, H  = self.w, self.h
        bank  = 4    # clear-land cells required beyond each bank edge

        water = np.zeros((W, H), dtype=bool)
        for x in range(W):
            for y in range(H):
                water[x, y] = (self.grid[x][y]["t"] == T_WATER)

        candidates = []   # (span, axis, slice_pos, water_start, water_end)

        for cl in self.river_centrelines:
            if len(cl) < 4: continue

            # Dominant flow direction from first→last centreline point
            fx0, fy0 = cl[0]; fx1, fy1 = cl[-1]
            adx = abs(fx1 - fx0); ady = abs(fy1 - fy0)
            total = adx + ady or 1.0
            # If neither axis dominates clearly, try both and keep all candidates
            # (the narrowest-first sort will select the better one).
            # bridge_axis 'col' = vertical strip (fixed x), 'row' = horizontal strip (fixed y)
            axes_to_try = []
            if adx >= ady * 1.3:        # clearly horizontal flow → vertical bridge
                axes_to_try = ['col']
            elif ady >= adx * 1.3:      # clearly vertical flow → horizontal bridge
                axes_to_try = ['row']
            else:                        # close to diagonal → try both, pick narrower
                axes_to_try = ['col', 'row']

            for bridge_axis in axes_to_try:
                if bridge_axis == 'col':
                    # Bridge is a vertical strip at a fixed x value
                    for x in range(bank+1, W-bank-1):
                        ys_water = np.where(water[x, :])[0]
                        if len(ys_water) == 0: continue
                        rs = int(ys_water[0]); rp = rs; runs = []
                        for cy in ys_water[1:]:
                            if cy > rp+1: runs.append((rs, rp)); rs = int(cy)
                            rp = int(cy)
                        runs.append((rs, rp))
                        for ry0, ry1 in runs:
                            span = ry1 - ry0 + 1
                            if span > H * 0.5: continue
                            # Bank above: either clear land, or river starts at map top edge
                            top_ok = (ry0 == 0 or
                                      (ry0-bank >= 0 and not np.any(water[x, max(0,ry0-bank):ry0])))
                            # Bank below: either clear land, or river exits at map bottom edge
                            bot_ok = (ry1 == H-1 or
                                      (ry1+bank < H and not np.any(water[x, ry1+1:min(H,ry1+bank+1)])))
                            if top_ok and bot_ok:
                                candidates.append((span, 'col', x, ry0, ry1))
                else:
                    # Bridge is a horizontal strip at a fixed y value
                    for y in range(bank+1, H-bank-1):
                        xs_water = np.where(water[:, y])[0]
                        if len(xs_water) == 0: continue
                        rs = int(xs_water[0]); rp = rs; runs = []
                        for cx in xs_water[1:]:
                            if cx > rp+1: runs.append((rs, rp)); rs = int(cx)
                            rp = int(cx)
                        runs.append((rs, rp))
                        for rx0, rx1 in runs:
                            span = rx1 - rx0 + 1
                            if span > W * 0.5: continue
                            left_ok  = (rx0 == 0 or
                                        (rx0-bank >= 0 and not np.any(water[max(0,rx0-bank):rx0, y])))
                            right_ok = (rx1 == W-1 or
                                        (rx1+bank < W and not np.any(water[rx1+1:min(W,rx1+bank+1), y])))
                            if left_ok and right_ok:
                                candidates.append((span, 'row', y, rx0, rx1))

        if not candidates: return

        candidates.sort(key=lambda c: c[0])   # narrowest first within each region

        def bridge_pos(c):
            """Return the axis position (the fixed x or y of the crossing)."""
            return c[2]   # for both 'col' (fixed x) and 'row' (fixed y)

        # For n_bridges=2: place one in each half of the map along the crossing axis.
        # This guarantees even coverage — robots anywhere on the map are within
        # half-map-width of a bridge.
        # For n_bridges=1 (or when one half has no candidates): best overall.
        axis_vals = [bridge_pos(c) for c in candidates]
        lo, hi = min(axis_vals), max(axis_vals)
        mid = (lo + hi) // 2

        chosen = []
        if n_bridges >= 2:
            left_cands  = [c for c in candidates if bridge_pos(c) <= mid]
            right_cands = [c for c in candidates if bridge_pos(c) >  mid]
            if left_cands:  chosen.append(left_cands[0])
            if right_cands: chosen.append(right_cands[0])
            # If only one half has candidates, pick 2 best overall spaced apart
            if len(chosen) < 2:
                chosen = []
                for c in candidates:
                    if len(chosen) >= n_bridges: break
                    if all(abs(bridge_pos(c)-bridge_pos(p)) >= 20 for p in chosen):
                        chosen.append(c)
        else:
            chosen = candidates[:1]

        hw = bridge_w // 2
        for cand in chosen:
            span, axis, pos, w0, w1 = cand
            if axis == 'col':
                x = pos
                # Extend stamp to map edge if river exits there
                y_lo = 0 if w0 == 0 else max(0, w0 - bank)
                y_hi = H if w1 == H-1 else min(H, w1 + bank + 1)
                for bx in range(max(0, x-hw), min(W, x+hw+1)):
                    for by in range(y_lo, y_hi):
                        if self.grid[bx][by]["t"] in (T_WATER, T_FREE):
                            self.grid[bx][by]["t"] = T_BRIDGE
            else:
                y = pos
                x_lo = 0 if w0 == 0 else max(0, w0 - bank)
                x_hi = W if w1 == W-1 else min(W, w1 + bank + 1)
                for by in range(max(0, y-hw), min(H, y+hw+1)):
                    for bx in range(x_lo, x_hi):
                        if self.grid[bx][by]["t"] in (T_WATER, T_FREE):
                            self.grid[bx][by]["t"] = T_BRIDGE

    def _fix_land_pinches(self, min_corridor=4):
        """
        Ensure every land/bridge cell reachable by at least one robot type is connected.
        Only carves through T_OBS (debris/rubble) — never through T_WATER.
        Rivers are intentional barriers; only bridges cross them.
        If a land pocket is isolated by water with no bridge, it stays isolated
        (it was cut off by the river, which is correct).
        """
        W, H = self.w, self.h
        # Passable = anything robots can traverse (land, stairs, bridge — not water or OBS)
        PASSABLE = {T_FREE, T_STAIRS, T_BRIDGE}

        def land_components():
            all_land = [(x,y) for x in range(W) for y in range(H)
                        if self.grid[x][y]["t"] in PASSABLE]
            visited = set(); comps = []
            for s in all_land:
                if s in visited: continue
                comp = []; q = deque([s]); visited.add(s)
                while q:
                    x,y = q.popleft(); comp.append((x,y))
                    for dx,dy in NBR4:
                        nx,ny = x+dx,y+dy
                        if (0<=nx<W and 0<=ny<H and (nx,ny) not in visited
                                and self.grid[nx][ny]["t"] in PASSABLE):
                            visited.add((nx,ny)); q.append((nx,ny))
                comps.append(set(comp))
            comps.sort(key=len, reverse=True)
            return comps

        for _pass in range(10):
            comps = land_components()
            if len(comps) <= 1:
                break
            main = comps[0]
            fixed_any = False

            for iso in comps[1:]:
                # BFS through T_OBS only (never water) to find path to main component
                parent = {}
                front = list(iso)
                for c in front: parent[c] = None
                q = deque(front)
                found = None
                while q and found is None:
                    x,y = q.popleft()
                    for dx,dy in NBR4:
                        nx,ny = x+dx,y+dy
                        if not (0<=nx<W and 0<=ny<H): continue
                        if (nx,ny) in parent: continue
                        if (nx,ny) in main:
                            parent[(nx,ny)] = (x,y); found = (nx,ny); break
                        # Only expand through OBS debris — not water
                        if self.grid[nx][ny]["t"] == T_OBS:
                            parent[(nx,ny)] = (x,y); q.append((nx,ny))
                    if found: break

                if found is None:
                    # Cannot connect without crossing water — leave isolated
                    # (the river legitimately separates this pocket)
                    continue

                # Carve path through OBS only
                cur = found
                while parent[cur] is not None and parent[cur] not in iso:
                    cur = parent[cur]
                    if self.grid[cur[0]][cur[1]]["t"] == T_OBS:
                        self.grid[cur[0]][cur[1]]["t"] = T_FREE
                fixed_any = True

            if not fixed_any:
                break

    def _water_components(self):
        visited = set(); comps = []
        for x in range(self.w):
            for y in range(self.h):
                if self.grid[x][y]["t"] != T_WATER or (x,y) in visited: continue
                q = deque([(x,y)]); visited.add((x,y)); comp = []
                while q:
                    cx,cy = q.popleft(); comp.append((cx,cy))
                    for nx,ny in ((cx-1,cy),(cx+1,cy),(cx,cy-1),(cx,cy+1)):
                        if 0<=nx<self.w and 0<=ny<self.h and self.grid[nx][ny]["t"]==T_WATER and (nx,ny) not in visited:
                            visited.add((nx,ny)); q.append((nx,ny))
                comps.append(comp)
        return comps

    def _rect_clear(self, x0, y0, w, h, pad=1):
        """Check if area is clear of water, bridges, and other buildings."""
        for x in range(max(0,x0-pad), min(self.w,x0+w+pad)):
            for y in range(max(0,y0-pad), min(self.h,y0+h+pad)):
                t = self.grid[x][y]["t"]
                # Only block on water, bridges, and existing buildings (stairs/walls)
                # Scattered obstacle debris is fine — buildings replace it
                if t in (T_WATER, T_BRIDGE, T_STAIRS): return False
        return True

    def _stamp_house(self, hx, hy, hw, hh):
        """Staircase building: clear footprint, outer obstacle walls, interior stairs, wide door."""
        # First clear the entire footprint (removes debris obstacles)
        for x in range(hx, hx+hw):
            for y in range(hy, hy+hh):
                self.grid[x][y]["t"] = T_FREE
        # Outer walls
        for x in range(hx, hx+hw):
            self.grid[x][hy]["t"]      = T_OBS
            self.grid[x][hy+hh-1]["t"] = T_OBS
        for y in range(hy, hy+hh):
            self.grid[hx][y]["t"]      = T_OBS
            self.grid[hx+hw-1][y]["t"] = T_OBS
        # Interior stairs
        for x in range(hx+1, hx+hw-1):
            for y in range(hy+1, hy+hh-1):
                self.grid[x][y]["t"] = T_STAIRS
        # Wide door (3 cells)
        door_x = hx + hw//2; door_y = hy + hh - 1
        for dx in range(-(3//2), 3//2+1):
            x = door_x + dx
            if hx <= x < hx+hw:
                self.grid[x][door_y]["t"] = T_STAIRS
                if door_y-1 > hy: self.grid[x][door_y-1]["t"] = T_STAIRS

    def _stamp_box(self, bx, by, bw, bh):
        for x in range(bx, bx+bw):
            self.grid[x][by]["t"]      = T_OBS
            self.grid[x][by+bh-1]["t"] = T_OBS
        for y in range(by, by+bh):
            self.grid[bx][y]["t"]  = T_OBS
            self.grid[bx+bw-1][y]["t"] = T_OBS
        self.grid[bx+bw//2][by]["t"] = T_FREE  # door

    def _generate(self):
        W, H = self.w, self.h

        # ── scattered obstacles (debris / rubble) ──
        for x in range(W):
            for y in range(H):
                if random.random() < 0.03:
                    self.grid[x][y]["t"] = T_OBS

        # ── river system: always exactly one main river, sometimes a tributary ──
        # Force the main river to cross the map properly (opposite edges).
        self._carve_river(base_width=random.randint(5,7), n_ctrl=4)
        if random.random() < 0.45:
            self._carve_river(base_width=random.randint(3,4), n_ctrl=3)

        # Clean up river immediately after carving (before anything else touches the grid)
        self._fill_river_islands()

        # ── place bridges: 2, one per map half ──
        self._place_bridges(n_bridges=2, bridge_w=5, min_spacing=40)

        # ── fix any isolated land pockets created by river + bridges ──
        # Must run AFTER bridges so the new T_BRIDGE cells are counted as passable
        self._fix_land_pinches(min_corridor=4)

        # ── buildings: 2-4 staircase buildings spread across quadrants ──        # Divide the map into quadrants and place at most one building per quadrant.
        # This guarantees spatial distribution rather than corner-clustering.
        quadrants = [
            (W//8,        H//8,        W//2-10, H//2-10),   # top-left
            (W//2+5,      H//8,        W-W//8,  H//2-10),   # top-right
            (W//8,        H//2+5,      W//2-10, H-H//8),    # bottom-left
            (W//2+5,      H//2+5,      W-W//8,  H-H//8),    # bottom-right
        ]
        random.shuffle(quadrants)
        n_stair_buildings = random.randint(2, 4)
        placed_houses = []
        for qx0, qy0, qx1, qy1 in quadrants:
            if len(placed_houses) >= n_stair_buildings: break
            hw = random.randint(10, 16); hh = random.randint(10, 16)
            placed = False
            for _ in range(120):
                hx = random.randint(qx0, max(qx0, qx1-hw))
                hy = random.randint(qy0, max(qy0, qy1-hh))
                if self._rect_clear(hx, hy, hw, hh, pad=4):
                    self._stamp_house(hx, hy, hw, hh)
                    placed_houses.append((hx, hy, hw, hh))
                    placed = True; break
            # If random placement failed, try the quadrant corner as fallback
            if not placed:
                hx = qx0 + 2; hy = qy0 + 2
                hw2 = min(hw, qx1-hx-2); hh2 = min(hh, qy1-hy-2)
                if hw2 >= 6 and hh2 >= 6 and self._rect_clear(hx, hy, hw2, hh2, pad=2):
                    self._stamp_house(hx, hy, hw2, hh2)
                    placed_houses.append((hx, hy, hw2, hh2))

        # ── outbuildings / hollow box structures near each staircase building ──
        for (hx, hy, hw, hh) in placed_houses:
            n_boxes = random.randint(1, 3)
            for _ in range(n_boxes):
                bw = random.randint(5, 10); bh = random.randint(5, 10)
                for attempt in range(60):
                    offset_x = random.randint(-22, 22)
                    offset_y = random.randint(-22, 22)
                    bx = hx + hw//2 + offset_x - bw//2
                    by = hy + hh//2 + offset_y - bh//2
                    bx = max(4, min(W-bw-4, bx))
                    by = max(4, min(H-bh-4, by))
                    if self._rect_clear(bx, by, bw, bh, pad=2):
                        self._stamp_box(bx, by, bw, bh); break

        # ── final connectivity pass — catches any pockets from building walls ──
        self._fix_land_pinches(min_corridor=4)


    def _sample_hotspots(self, n, amp_n, amp_p, amp_scale):
        hs = []
        for _ in range(n):
            mx = random.randint(0, self.w-1); my = random.randint(0, self.h-1)
            sigma = random.uniform(SIGMA_MIN, SIGMA_MAX)
            amp = max(1.0, float(np.random.binomial(amp_n, amp_p)) * amp_scale)
            hs.append(((mx, my), sigma, amp))
        return hs

    def _init_temperature(self):
        hs = self._sample_hotspots(N_HOTSPOTS_TEMP, TEMP_AMP_N, TEMP_AMP_P, TEMP_AMP_SCALE)
        for x in range(self.w):
            for y in range(self.h):
                v = sum(amp * math.exp(-((x-mx)**2+(y-my)**2)/(2*s*s)) for (mx,my),s,amp in hs)
                if self.grid[x][y]["t"] in (T_WATER, T_BRIDGE): v = 5.0
                self.grid[x][y]["temp"] = v

    def _init_radiation(self):
        hs = self._sample_hotspots(N_HOTSPOTS_RAD, RAD_AMP_N, RAD_AMP_P, RAD_AMP_SCALE)
        for x in range(self.w):
            for y in range(self.h):
                v = sum(amp * math.exp(-((x-mx)**2+(y-my)**2)/(2*s*s)) for (mx,my),s,amp in hs)
                if self.grid[x][y]["t"] in (T_WATER, T_BRIDGE): v = 0.0
                self.grid[x][y]["rad"] = v

# ─────────────────────────────────────────────────────────────────────────────
# A* planner (self-contained static method)
# ─────────────────────────────────────────────────────────────────────────────
class AStar:
    @staticmethod
    def search(start, goal, caps_mask,
               terrain_u8, temp_f32, rad_f32,
               chunked_risk, temp_limit, rad_limit,
               radio_shadow, relay_ok_fn, cell_to_zone_fn,
               global_cov, unk_pen, info_w, unk_prior,
               alpha_mult, beta_mult, soft_frac,
               traffic_u16, traffic_w):

        W, H = terrain_u8.shape
        if start == goal: return []
        sx, sy = start; gx, gy = goal

        INF = 1e30
        gscore = np.full((W, H), INF, dtype=np.float32)
        gscore[sx, sy] = 0.0
        px = np.full((W, H), -1, dtype=np.int16)
        py = np.full((W, H), -1, dtype=np.int16)
        closed = np.zeros((W, H), dtype=np.uint8)

        def h(x, y): return abs(x-gx) + abs(y-gy)

        heap = [(h(sx,sy), 0.0, sx, sy)]

        inv_T   = 1.0 / max(1e-6, temp_limit)
        inv_R   = 1.0 / max(1e-6, rad_limit)
        soft_t  = soft_frac * temp_limit
        soft_r  = soft_frac * rad_limit
        has_land   = bool(caps_mask & CAP_LAND)
        has_stairs = bool(caps_mask & CAP_STAIRS)
        has_water  = bool(caps_mask & CAP_WATER)
        has_air    = bool(caps_mask & CAP_AIR)
        a_eff = ALPHA * alpha_mult
        b_eff = BETA  * beta_mult

        while heap:
            f, g, x, y = heapq.heappop(heap)
            if closed[x, y]: continue
            closed[x, y] = 1

            if (x, y) == (gx, gy):
                path = []
                cx, cy = gx, gy
                while (cx, cy) != (sx, sy):
                    path.append((cx, cy))
                    ncx, ncy = int(px[cx,cy]), int(py[cx,cy])
                    if ncx < 0: return []
                    cx, cy = ncx, ncy
                path.reverse()
                return path

            for dx, dy in NBR4:
                nx, ny = x+dx, y+dy
                if not (0 <= nx < W and 0 <= ny < H): continue
                if closed[nx, ny]: continue

                tb = int(terrain_u8[nx, ny])

                # ── terrain feasibility ──
                if tb == T_OBS:                                              continue
                if tb == T_STAIRS and not has_stairs and not has_air:        continue
                if tb == T_WATER  and not has_water  and not has_air:        continue
                if tb == T_BRIDGE and not (has_land or has_water or has_air or has_stairs): continue
                if tb == T_FREE   and not has_land   and not has_air:        continue

                # ── unknown: halo safety check ──
                if tb == T_UNKNOWN:
                    # Land robots: treat unknown cells adjacent to known water as
                    # impassable — water is contiguous, the unknown is probably water too.
                    if has_land and not has_water and not has_air:
                        water_adjacent = False
                        for ddx, ddy in NBR4:
                            ax, ay = nx+ddx, ny+ddy
                            if 0<=ax<W and 0<=ay<H:
                                if int(terrain_u8[ax,ay]) == T_WATER:
                                    water_adjacent = True; break
                        if water_adjacent: continue
                    # Hazard halo: skip unknown near known-high-hazard cells
                    if temp_limit < 9000.0 or rad_limit < 9000.0:
                        danger = False
                        for ddx, ddy in NBR4:
                            ax, ay = nx+ddx, ny+ddy
                            if 0<=ax<W and 0<=ay<H and terrain_u8[ax,ay] != T_UNKNOWN:
                                if temp_f32[ax,ay] > soft_t or rad_f32[ax,ay] > soft_r:
                                    danger = True; break
                        if danger: continue
                else:
                    t_ = temp_f32[nx, ny]; r_ = rad_f32[nx, ny]
                    if t_ > temp_limit or r_ > rad_limit: continue

                # ── radio shadow gate ──
                if radio_shadow[nx, ny]:
                    z = cell_to_zone_fn(nx, ny)
                    if not relay_ok_fn(z): continue

                # ── costs ──
                base = 1.0
                cx_c, cy_c = nx//CHUNK_SIZE, ny//CHUNK_SIZE
                lam_T = float(chunked_risk[0, cx_c, cy_c])
                lam_R = float(chunked_risk[1, cx_c, cy_c])
                e_c   = max(lam_T*inv_T, lam_R*inv_R)

                if tb != T_UNKNOWN:
                    t_  = 0.0 if np.isnan(temp_f32[nx,ny]) else float(temp_f32[nx,ny])
                    r_  = 0.0 if np.isnan(rad_f32[nx,ny])  else float(rad_f32[nx,ny])
                    e   = max(t_*inv_T, r_*inv_R)
                    unk = 0.0
                else:
                    e   = unk_prior * e_c
                    unk = unk_pen if global_cov < 0.95 else min(unk_pen, 0.3)

                # info-gain reward: stepping here will reveal unknown neighbours
                ig = 0.0
                if info_w > 1e-9:
                    cnt = sum(1 for ddx,ddy in NBR4
                              if 0<=nx+ddx<W and 0<=ny+ddy<H
                              and terrain_u8[nx+ddx,ny+ddy]==T_UNKNOWN)
                    ig = info_w * cnt

                step_cost = (base + unk
                             + a_eff * (e_c**P)
                             + b_eff * (e**P)
                             - ig
                             + traffic_w * float(traffic_u16[nx,ny]))
                step_cost = max(MIN_STEP_COST, step_cost)

                ng = g + step_cost
                if ng < gscore[nx, ny]:
                    gscore[nx, ny] = ng
                    px[nx, ny] = x; py[nx, ny] = y
                    heapq.heappush(heap, (ng + h(nx,ny), ng, nx, ny))

        return []

# ─────────────────────────────────────────────────────────────────────────────
# Per-tick zone frontier cache (shared across robots, rebuilt once per tick)
# ─────────────────────────────────────────────────────────────────────────────
class ZoneFrontierCache:
    """
    Stores zone frontiers keyed by (zone, robot_caps_mask) so different locomotion
    types get the right result without re-scanning the zone multiple times.
    """
    def __init__(self):
        self._cache: dict = {}

    def clear(self):
        self._cache.clear()

    def get(self, zone, caps_mask, union, reachable_fn, world, zone_cells_fn,
            confidence=None):
        key = (zone, caps_mask)
        if key in self._cache:
            return self._cache[key]
        result = self._compute(zone, caps_mask, union, reachable_fn, world,
                               zone_cells_fn, confidence)
        self._cache[key] = result
        return result

    def _compute(self, zone, caps_mask, union, reachable_fn, world, zone_cells_fn,
                 confidence=None):
        xs, ys = zone_cells_fn(zone)
        reachable = reachable_fn(caps_mask)
        # Accept either a bool numpy array (fast) or set of tuples (legacy)
        if isinstance(reachable, np.ndarray):
            reach_check = lambda x, y: bool(reachable[x, y])
        else:
            reach_check = lambda x, y: (x, y) in reachable
        out = []
        is_boat = bool(caps_mask & CAP_WATER) and not bool(caps_mask & CAP_AIR)
        for x in xs:
            for y in ys:
                if not reach_check(x, y): continue
                is_uncertain = (union[x, y] == T_UNKNOWN or
                                (confidence is not None and
                                 confidence[x, y] < CONF_UNCERTAIN and
                                 union[x, y] != T_UNKNOWN))
                if is_boat:
                    if union[x,y] in (T_WATER, T_BRIDGE):
                        if (any(union[nx,ny]==T_UNKNOWN
                                for (nx,ny) in world.neighbours((x,y))) or
                            is_uncertain):
                            out.append((x,y))
                else:
                    if is_uncertain:
                        out.append((x,y))
                    elif any(union[nx,ny]==T_UNKNOWN
                             for (nx,ny) in world.neighbours((x,y))):
                        out.append((x,y))
        return out

# ─────────────────────────────────────────────────────────────────────────────
# Robot
# ─────────────────────────────────────────────────────────────────────────────
class Robot:
    def __init__(self, name, x, y, caps, world, sim,
                 weights, temp_limit, rad_limit):
        self.name        = name
        self.pos         = (x, y)
        self.caps        = caps
        self.caps_mask   = caps_to_mask(caps)
        self.world       = world
        self.sim         = sim
        self.weights     = weights
        self.temp_limit  = temp_limit
        self.rad_limit   = rad_limit
        self.soft_temp   = 0.85 * temp_limit
        self.soft_rad    = 0.85 * rad_limit

        # belief
        self.terrain_belief = np.full((GRID_W, GRID_H), T_UNKNOWN, dtype=np.uint8)
        self.temp_belief    = np.full((GRID_W, GRID_H), np.nan,    dtype=np.float32)
        self.rad_belief     = np.full((GRID_W, GRID_H), np.nan,    dtype=np.float32)
        self.known_mask     = np.zeros((GRID_W, GRID_H), dtype=bool)
        self.chunked        = np.zeros((2, GRID_W//CHUNK_SIZE, GRID_H//CHUNK_SIZE), dtype=np.float32)

        # Scan confidence — age-based decay.
        # scan_age[x,y] = ticks since this robot last directly scanned cell (x,y).
        # Cells never scanned have age=INF (represented as int16 max = 32767).
        # confidence = exp(-age / CONF_TAU), used by local planner and CBBA.
        self.scan_age       = np.full((GRID_W, GRID_H), 32767, dtype=np.int16)
        self.confidence     = np.zeros((GRID_W, GRID_H), dtype=np.float32)

        # Comms — hop-by-hop message passing.
        # outbox: list of dicts {x,y,terrain,temp,rad,ts,origin,hops_left}
        # Each tick: broadcast to range-neighbours, they merge + forward.
        self.outbox:  list  = []
        self.inbox:   list  = []   # filled by FleetSim.step() comms pass
        # personally_scanned: cells this robot has directly sensed (not received)
        self.personally_scanned = np.zeros((GRID_W, GRID_H), dtype=bool)

        # state
        self.active       = True
        self.battery      = MAX_BATTERY
        self.death_reason = None
        self.role         = Role.SCAN
        self.role_locked_until = 0

        # navigation
        self.goal         = None
        self.path         = []
        self.goal_commit  = 0
        self.stuck_steps  = 0
        self.failed_goals = {}   # {(x,y): retry_after_timestep}

        # zone management
        self.bundle          = []
        self.assigned_zones  = []
        self.task_zone       = None
        self.zone_lease_until= 0
        self.task_no_progress= 0
        self.task_last_known = 0
        self.blacklist       = {}   # {zone: until_timestep}

        # relay
        self.relay_anchor       = None
        self.relay_anchor_zone  = None
        self.relay_hold_until   = 0
        self.relay_last_occupied= 0   # last tick a non-relay robot was inside the shadow bubble

        # hazard dose
        self.dose_T = 0.0; self.dose_R = 0.0; self.survival_p = 1.0

        # caches
        self._reachable_cache: set  = None
        self._reachable_arr:   object = None
        self._reachable_tick:  int  = -999
        self.zone_frontier_signal   = 0.0
        self.zone_frontier_count    = 0
        self.reveal_R               = 2

        self._reveal_all(); self._recompute_chunked()

    def _reveal_all(self):
        """Sensor scan: reveal cells within reveal_R, stamp age=0, queue for broadcast."""
        x0, y0 = self.pos; R = self.reveal_R
        now = self.sim.timestep
        newly_scanned = []
        for dx in range(-R, R+1):
            for dy in range(-R, R+1):
                if dx*dx + dy*dy <= R*R:
                    nx, ny = x0+dx, y0+dy
                    if 0 <= nx < self.world.w and 0 <= ny < self.world.h:
                        self.personally_scanned[nx, ny] = True
                        self.scan_age[nx, ny] = 0
                        self.confidence[nx, ny] = 1.0
                        if not self.known_mask[nx, ny]:
                            self.known_mask[nx, ny] = True
                            self.terrain_belief[nx,ny] = self.world.grid[nx][ny]["t"]
                            self.temp_belief[nx,ny]    = self.world.grid[nx][ny]["temp"]
                            self.rad_belief[nx,ny]     = self.world.grid[nx][ny]["rad"]
                            newly_scanned.append((nx, ny))
                        else:
                            # Re-scan: refresh belief in case nothing changed but
                            # confidence was decayed — mark for rebroadcast
                            newly_scanned.append((nx, ny))

        # Queue all freshly scanned cells into outbox for comms propagation.
        for (nx, ny) in newly_scanned:
            self.outbox.append({
                'x': nx, 'y': ny,
                'terrain': int(self.terrain_belief[nx, ny]),
                'temp':    float(self.temp_belief[nx, ny]),
                'rad':     float(self.rad_belief[nx, ny]),
                'ts':      now,
                'origin':  self.name,
            })

    def _age_decay_tick(self):
        """
        Increment scan_age for all known cells by 1 tick (capped at int16 max).
        Recompute confidence = exp(-age / CONF_TAU).
        Cells that decay below CONF_UNCERTAIN are treated as uncertain by the
        local planner — they regain exploration value and may be revisited.
        """
        known = self.known_mask
        # Increment age only for cells we know about (unknown cells stay at 32767)
        age = self.scan_age
        age[known] = np.minimum(age[known].astype(np.int32) + 1, 32767).astype(np.int16)
        # Recompute confidence only for known cells
        self.confidence[known] = np.exp(-age[known].astype(np.float32) / CONF_TAU)

    def _process_inbox(self):
        """
        Merge received messages into local belief.
        Only accept data that is fresher than what we already have (by timestamp).
        In the centralised model there is no forwarding — the fleet planner
        delivers messages directly.
        """
        if not self.inbox: return
        for msg in self.inbox:
            x, y = msg['x'], msg['y']
            our_age = int(self.scan_age[x, y])
            msg_age = self.sim.timestep - msg['ts']
            if msg_age < our_age:
                self.terrain_belief[x, y] = msg['terrain']
                self.known_mask[x, y]     = True
                t_val = msg['temp']; r_val = msg['rad']
                if not math.isnan(t_val): self.temp_belief[x, y] = t_val
                if not math.isnan(r_val): self.rad_belief[x, y]  = r_val
                self.scan_age[x, y]   = np.int16(min(msg_age, 32767))
                self.confidence[x, y] = math.exp(-msg_age / CONF_TAU)
                # Note: reachable cache NOT cleared here — terrain changes don't affect
                # shadow routing, and the per-tick cache guard handles stale data.
        self.inbox.clear()
        self._recompute_chunked()

    def _recompute_chunked(self):
        nW = GRID_W//CHUNK_SIZE; nH = GRID_H//CHUNK_SIZE
        mT = np.zeros((GRID_W, GRID_H), dtype=np.float32)
        mR = np.zeros((GRID_W, GRID_H), dtype=np.float32)
        mT[self.known_mask] = np.nan_to_num(self.temp_belief[self.known_mask], nan=0.0)
        mR[self.known_mask] = np.nan_to_num(self.rad_belief [self.known_mask], nan=0.0)
        self.chunked[0] = mT.reshape(nW,CHUNK_SIZE,nH,CHUNK_SIZE).max(axis=(1,3))
        self.chunked[1] = mR.reshape(nW,CHUNK_SIZE,nH,CHUNK_SIZE).max(axis=(1,3))

    # ── reachable BFS (cached per tick) ──────────────────────────────────────
    def reachable(self) -> np.ndarray:
        """Compute reachable cells from current position. Returns bool numpy array.
        Uses scipy connected-components for ~100x speedup over pure BFS.
        Cached per tick — repeated calls in the same tick are free."""
        from scipy import ndimage as _ndi
        t = self.sim.timestep
        if self._reachable_tick == t and self._reachable_arr is not None:
            return self._reachable_arr

        W, H    = self.world.w, self.world.h
        tb_arr  = self.terrain_belief          # uint8 numpy array
        shadow  = self.sim.radio_shadow        # bool numpy array
        mask    = self.caps_mask
        trav    = _TRAV_LUT                    # pre-built lookup
        is_boat = bool(mask & CAP_WATER) and not bool(mask & CAP_AIR)
        is_land = bool(mask & CAP_LAND)  and not bool(mask & (CAP_AIR|CAP_WATER))
        mask4   = mask & 0xF

        # Build passable mask: vectorised terrain check
        passable = np.zeros((W, H), dtype=bool)
        for terrain_code in range(6):
            if trav[terrain_code][mask4]:
                passable |= (tb_arr == terrain_code)

        # Unknown-cell refinements for land and boat
        if is_land:
            # Unknown cells adjacent to known water are likely water — block them
            unknown_mask = (tb_arr == T_UNKNOWN)
            water_known  = (tb_arr == T_WATER)
            # Dilate water mask by 1 cell to find "near water" cells
            water_nbr = _ndi.binary_dilation(water_known, structure=np.ones((3,3),dtype=bool))
            # Remove unknown cells that are next to water
            passable &= ~(unknown_mask & water_nbr)
        elif is_boat:
            # Boat unknown cells only passable if adjacent to known water
            unknown_mask = (tb_arr == T_UNKNOWN)
            water_known  = (tb_arr == T_WATER)
            water_nbr    = _ndi.binary_dilation(water_known, structure=np.ones((3,3),dtype=bool))
            passable &= ~(unknown_mask & ~water_nbr)

        # Shadow gate: block shadow cells unless relay_ok
        relay_ok_flood = self.sim._relay_ok_flood
        if np.any(shadow):
            shadow_blocked = shadow.copy()
            # Unblock shadow cells whose zone has relay coverage
            zw, zh = self.sim.zone_w_cells, self.sim.zone_h_cells
            for (zx, zy), ok in relay_ok_flood.items():
                if ok:
                    x0 = zx*zw; x1 = min(x0+zw, W)
                    y0 = zy*zh; y1 = min(y0+zh, H)
                    shadow_blocked[x0:x1, y0:y1] = False
            passable &= ~shadow_blocked

        # scipy connected components: find the component containing start cell
        sx, sy = self.pos
        if not passable[sx, sy]:
            # Robot is on an impassable cell — return just the current cell
            visited_arr = np.zeros((W, H), dtype=bool)
            visited_arr[sx, sy] = True
        else:
            labeled, _ = _ndi.label(passable)
            seed_label  = labeled[sx, sy]
            visited_arr = (labeled == seed_label)

        # Store bool array — set form built lazily only when explicitly requested
        self._reachable_arr   = visited_arr
        self._reachable_cache = None        # set form; built lazily by _reachable_set()
        self._reachable_tick  = t
        return visited_arr

    def _reachable_set(self) -> set:
        """Return reachable as a set of (x,y) tuples. Built lazily from the bool array."""
        if self._reachable_cache is None and self._reachable_arr is not None:
            coords = np.argwhere(self._reachable_arr)
            self._reachable_cache = set(map(tuple, coords.tolist()))
        return self._reachable_cache or set()

    # ── goal setting ──────────────────────────────────────────────────────────
    def set_goal(self, tgt) -> bool:
        """Plan to tgt. Returns True if a valid path was found."""
        if tgt == self.pos:
            self.goal = None; self.path = []; return False

        unk_pen, info_w, a_mult, b_mult = self._planner_params()
        path = AStar.search(
            start=self.pos, goal=tgt,
            caps_mask=self.caps_mask,
            terrain_u8=self.terrain_belief,
            temp_f32=self.temp_belief,
            rad_f32=self.rad_belief,
            chunked_risk=self.chunked,
            temp_limit=self.temp_limit,
            rad_limit=self.rad_limit,
            radio_shadow=self.sim.radio_shadow,
            relay_ok_fn=self.sim.relay_ok_extended,
            cell_to_zone_fn=self.sim.cell_to_zone,
            global_cov=self.sim.global_cov,
            unk_pen=unk_pen, info_w=info_w,
            unk_prior=UNK_PRIOR,
            alpha_mult=a_mult, beta_mult=b_mult,
            soft_frac=SOFT_FRAC,
            traffic_u16=self.sim.traffic_u16,
            traffic_w=TRAFFIC_W,
        )
        if not path:
            self.failed_goals[tgt] = self.sim.timestep + 80
            return False

        # reject paths with known-lethal cells in first K steps
        for cell in path[:12]:
            x, y = cell
            tb = self.terrain_belief[x, y]
            if tb != T_UNKNOWN:
                if self.temp_belief[x,y] > self.temp_limit or self.rad_belief[x,y] > self.rad_limit:
                    self.failed_goals[tgt] = self.sim.timestep + 80
                    return False

        self.goal = tgt; self.path = path; self.goal_commit = 20
        self.stuck_steps = 0
        return True

    def _planner_params(self):
        if self.role == Role.SCOUT:
            return UNK_PEN_SCOUT, INFO_W_SCOUT, 1.0, 1.0
        return UNK_PEN_OTHER, INFO_W_OTHER, 1.0, 1.0

    # ── move one step ─────────────────────────────────────────────────────────
    def move_step(self, occupied: set) -> bool:
        """
        Execute one movement step.
        Returns True if the robot actually moved.
        Handles: battery, path validity, collision reservation, illegal terrain.
        """
        if not self.active or self.battery <= 0:
            self.active = False; return False

        if not self.goal or not self.path:
            return False

        # replan if path became invalid under current belief
        if self._path_invalid():
            unk_pen, info_w, a_mult, b_mult = self._planner_params()
            new_path = AStar.search(
                start=self.pos, goal=self.goal,
                caps_mask=self.caps_mask,
                terrain_u8=self.terrain_belief,
                temp_f32=self.temp_belief, rad_f32=self.rad_belief,
                chunked_risk=self.chunked,
                temp_limit=self.temp_limit, rad_limit=self.rad_limit,
                radio_shadow=self.sim.radio_shadow,
                relay_ok_fn=self.sim.relay_ok_extended,
                cell_to_zone_fn=self.sim.cell_to_zone,
                global_cov=self.sim.global_cov,
                unk_pen=unk_pen, info_w=info_w, unk_prior=UNK_PRIOR,
                alpha_mult=a_mult, beta_mult=b_mult, soft_frac=SOFT_FRAC,
                traffic_u16=self.sim.traffic_u16, traffic_w=TRAFFIC_W,
            )
            if not new_path:
                self.failed_goals[self.goal] = self.sim.timestep + 80
                self.goal = None; self.path = []; self.goal_commit = 0
                return False
            self.path = new_path

        next_cell = self.path[0]

        # ── collision: cell already reserved this tick ──
        if next_cell in occupied:
            # Try to wait one tick — don't thrash the goal
            self.stuck_steps += 1
            if self.stuck_steps > 3:
                # Force replan next tick by clearing path (goal kept)
                self.path = []; self.goal_commit = 0
            return False

        # ── execute move ──
        prev = self.pos
        self.pos = self.path.pop(0)
        occupied.discard(prev); occupied.add(self.pos)

        true_t = self.world.grid[self.pos[0]][self.pos[1]]["t"]

        # ── verify true terrain (hard safety gate) ──
        illegal = False
        if true_t == T_OBS: illegal = True
        elif true_t == T_WATER  and not bool(self.caps_mask & (CAP_WATER|CAP_AIR)): illegal = True
        elif true_t == T_STAIRS and not bool(self.caps_mask & (CAP_STAIRS|CAP_AIR)): illegal = True
        elif true_t == T_FREE   and not bool(self.caps_mask & (CAP_LAND|CAP_AIR)):   illegal = True
        # boat must stay on water/bridge
        if bool(self.caps_mask & CAP_WATER) and not bool(self.caps_mask & CAP_AIR):
            if true_t not in (T_WATER, T_BRIDGE): illegal = True

        if illegal:
            self.terrain_belief[self.pos[0],self.pos[1]] = true_t
            self.known_mask[self.pos[0],self.pos[1]] = True
            self.failed_goals[self.pos] = self.sim.timestep + 120
            occupied.discard(self.pos); occupied.add(prev)
            self.pos = prev; self.path = []; self.goal_commit = 0
            return False

        # ── radio shadow gate — bounce back instead of kill ──
        # If a robot steps into shadow without relay coverage, push it back to
        # where it came from and clear its path so it replans away from shadow.
        # This preserves the robot (no deaths from relay churn) while still
        # enforcing the comms constraint — the robot simply can't go deeper.
        if self.sim.radio_shadow[self.pos[0], self.pos[1]]:
            z = self.sim.cell_to_zone(self.pos[0], self.pos[1])
            if not self.sim.relay_ok_extended(z) and self.role != Role.RELAY:
                occupied.discard(self.pos); occupied.add(prev)
                self.pos = prev; self.path = []; self.goal_commit = 0
                # Blacklist the shadow zone briefly so the planner avoids it
                self.blacklist[z] = self.sim.timestep + 20
                return False

        # ── reveal & update ──
        self._reveal_all()
        self._recompute_chunked()

        # ── battery drain ──
        role_mult = {Role.SCOUT:1.4, Role.SCAN:1.0, Role.RELAY:1.1, Role.LOITER:0.6}[self.role]
        drain = {"Legged": 1.0, "Drone": 2.0, "Boat": 2.0, "Rover": 0.4}
        rt = robot_type(self.name)
        self.battery -= drain.get(rt, 1.0) * role_mult

        # ── hazard exposure ──
        c = self.world.grid[self.pos[0]][self.pos[1]]
        self.dose_T += max(0.0, c["temp"]) * 0.01
        self.dose_R += max(0.0, c["rad"])  * 0.01
        dose = self.dose_T + self.dose_R
        self.survival_p = math.exp(-((dose/300.0)**2))

        if c["temp"] > self.temp_limit or c["rad"] > self.rad_limit:
            reasons = []
            if c["temp"] > self.temp_limit: reasons.append("high temperature")
            if c["rad"]  > self.rad_limit:  reasons.append("high radiation")
            self.active = False
            self.death_reason = " & ".join(reasons)

        if self.goal_commit > 0: self.goal_commit -= 1
        return True

    def _path_invalid(self) -> bool:
        if not self.path: return True
        for cell in self.path[:8]:
            x, y = cell
            tb = self.terrain_belief[x, y]
            if not traversable_code(int(tb), self.caps_mask): return True
            if tb != T_UNKNOWN:
                if self.temp_belief[x,y] > self.temp_limit: return True
                if self.rad_belief [x,y] > self.rad_limit:  return True
            if self.sim.radio_shadow[x, y]:
                z = self.sim.cell_to_zone(x, y)
                if not self.sim.relay_ok_extended(z): return True
        return False

    # ── comms merge (absorb union belief) ────────────────────────────────────
    def merge_union(self, union_t, union_temp, union_rad):
        """Legacy stub — comms now handled via hop-by-hop inbox/outbox. No-op."""
        pass

# ─────────────────────────────────────────────────────────────────────────────
# Simulation Controller
# ─────────────────────────────────────────────────────────────────────────────
class FleetSim:
    def __init__(self):
        self.world    = GridWorld(GRID_W, GRID_H)
        self.timestep = 0
        self.global_cov = 0.0

        # zone grid
        self.zone_w_cells = ZONE_CHUNKS * CHUNK_SIZE
        self.zone_h_cells = ZONE_CHUNKS * CHUNK_SIZE
        self.zone_nx = GRID_W // self.zone_w_cells
        self.zone_ny = GRID_H // self.zone_h_cells
        self.zone_tasks = {
            (zx,zy): ZoneTask((zx,zy))
            for zx in range(self.zone_nx) for zy in range(self.zone_ny)
        }

        # shared arrays
        self.radio_shadow = np.zeros((GRID_W, GRID_H), dtype=bool)
        self.traffic_u16  = np.zeros((GRID_W, GRID_H), dtype=np.uint16)
        self.relay_ok     = {}
        self._shadow_border_mask_cache = None  # cached per simulation build

        self.found       = set()
        self.dead_robots = []

        self._zone_stats_cache     = {}
        self._zone_stats_cache_tick= -1

        self._frontier_cache = ZoneFrontierCache()
        self._reachable_by_mask: dict = {}   # caps_mask -> set (rebuilt each tick)
        self._pending_msgs:      list = []   # fleet comms queue: msgs awaiting delivery

        self._build_robots()
        self._build_survivors()
        self._build_radio_shadow()

        # Pre-compute zone ID per cell: zx * zone_ny + zy  (int16)
        self._zone_id_arr = np.zeros((GRID_W, GRID_H), dtype=np.int16)
        for zx in range(self.zone_nx):
            x0 = zx * self.zone_w_cells; x1 = x0 + self.zone_w_cells
            for zy in range(self.zone_ny):
                y0 = zy * self.zone_h_cells; y1 = y0 + self.zone_h_cells
                self._zone_id_arr[x0:x1, y0:y1] = zx * self.zone_ny + zy

        # Pre-compute per-zone shadow fraction (shadow is static — never changes)
        self._zone_shadow_frac = {}
        for zx in range(self.zone_nx):
            x0 = zx * self.zone_w_cells; x1 = min(x0 + self.zone_w_cells, GRID_W)
            for zy in range(self.zone_ny):
                y0 = zy * self.zone_h_cells; y1 = min(y0 + self.zone_h_cells, GRID_H)
                self._zone_shadow_frac[(zx, zy)] = float(np.mean(self.radio_shadow[x0:x1, y0:y1]))

        # Precompute world water mask (static — grid never changes)
        self._world_water_arr = np.zeros((GRID_W, GRID_H), dtype=bool)
        for x in range(GRID_W):
            for y in range(GRID_H):
                self._world_water_arr[x, y] = self.world.grid[x][y]["t"] in (T_WATER, T_BRIDGE)

        # initialise relay_ok — False for shadow zones (relay needed), True for open zones
        self.relay_ok = {(zx,zy): False
                         for zx in range(self.zone_nx) for zy in range(self.zone_ny)}
        self._relay_ok_flood = dict(self.relay_ok)
        self._relay_ok_prev  = dict(self.relay_ok)  # detect changes to avoid redundant cache clears

        self.union_belief = self._union_terrain()
        self.union_T      = self._union_temp()
        self.union_R      = self._union_rad()

        self._assign_zones_cbba()

    # ── robot factory ─────────────────────────────────────────────────────────
    def _build_robots(self):
        templates = [
            ("Legged", {Capability.LAND,Capability.STAIRS}, np.array([10.,10.]), (TEMP_LIMIT,RAD_LIMIT)),
            ("Drone",  {Capability.AIR},                   np.array([10.,10.]), (TEMP_LIMIT,RAD_LIMIT)),
            ("Boat",   {Capability.WATER},                 np.array([0.,0.]),   (9999.,9999.)),
            ("Rover",  {Capability.LAND},                  np.array([-2.,-2.]),(9999.,9999.)),
        ]
        tpl = {n:(n,c,w,l) for n,c,w,l in templates}
        desired = {"Legged":3,"Drone":4,"Boat":2,"Rover":3}
        spawn = []
        for t,n in desired.items(): spawn += [t]*n
        random.shuffle(spawn)

        clusters = [(6,6),(GRID_W-7,6),(6,GRID_H-7),(GRID_W-7,GRID_H-7)]
        water_cells = [(x,y) for x in range(GRID_W) for y in range(GRID_H)
                       if self.world.grid[x][y]["t"]==T_WATER]
        water_by_quad = {(qx,qy): [(x,y) for x,y in water_cells
                                    if (x<GRID_W//2)==(qx==0) and (y<GRID_H//2)==(qy==0)]
                         for qx in (0,1) for qy in (0,1)}

        self.robots = []
        for i, tname in enumerate(spawn):
            _, caps, weights, (tlim,rlim) = tpl[tname]
            name = f"{tname}{i}"
            center = clusters[i % 4]
            qx = 0 if center[0] < GRID_W//2 else 1
            qy = 0 if center[1] < GRID_H//2 else 1

            if tname == "Boat" and water_cells:
                pool = water_by_quad.get((qx,qy),[]) or water_cells
                # Prefer non-shadow water cells for spawn
                non_shadow_pool = [c for c in pool if not self.radio_shadow[c[0],c[1]]]
                sx,sy = random.choice(non_shadow_pool if non_shadow_pool else pool)
            else:
                sx,sy = center
                for _ in range(30):
                    cx = max(1,min(GRID_W-2, center[0]+random.randint(-8,8)))
                    cy = max(1,min(GRID_H-2, center[1]+random.randint(-8,8)))
                    tt = self.world.grid[cx][cy]["t"]
                    if (tt==T_FREE or (tt==T_STAIRS and Capability.STAIRS in caps)) and not self.radio_shadow[cx,cy]:
                        sx,sy = cx,cy; break

            self.robots.append(Robot(name,sx,sy,caps,self.world,self,weights,tlim,rlim))

    def _build_survivors(self):
        free = [(x,y) for x in range(GRID_W) for y in range(GRID_H)
                if self.world.grid[x][y]["t"]==T_FREE]
        def near(cx,cy,r=10):
            return [(x,y) for x,y in free if abs(x-cx)<=r and abs(y-cy)<=r]
        critical = []
        for pool in (near(int(GRID_W*.75),GRID_H//2),
                     near(int(GRID_W*.55),int(GRID_H*.75)),
                     near(GRID_W//6+2,GRID_H//2+6)):
            if pool: critical.append(random.choice(pool))
        rest = [c for c in free if c not in critical]
        self.survivors = critical + random.sample(rest, max(0,18-len(critical)))

    def _build_radio_shadow(self):
        rs = np.zeros((GRID_W,GRID_H),dtype=bool)
        for x in range(GRID_W):
            for y in range(GRID_H):
                if self.world.grid[x][y]["t"] == T_STAIRS:
                    rs[x,y] = True
        cx = random.randint(GRID_W//5, GRID_W-GRID_W//5)
        cy = random.randint(GRID_H//5, GRID_H-GRID_H//5)
        rad = random.randint(14,26)
        for x in range(max(0,cx-rad),min(GRID_W,cx+rad+1)):
            for y in range(max(0,cy-rad),min(GRID_H,cy+rad+1)):
                if (x-cx)**2+(y-cy)**2 <= rad*rad: rs[x,y]=True
        if not np.any(rs):
            mx,my = GRID_W//2,GRID_H//2
            rs[mx-3:mx+4,my-3:my+4] = True
        self.radio_shadow = rs
        # Cache shadow cell positions (static — never changes)
        self._shadow_cells_arr = np.argwhere(rs)  # shape (N, 2)
        # Cache shadow border mask (outside shadow, touching shadow)
        rs_i = rs.astype(np.uint8)
        nbr = (np.roll(rs_i,1,0)|np.roll(rs_i,-1,0)|np.roll(rs_i,1,1)|np.roll(rs_i,-1,1)).astype(bool)
        nbr[0,:] &= rs[1,:]; nbr[-1,:] &= rs[-2,:]
        nbr[:,0] &= rs[:,1]; nbr[:,-1] &= rs[:,-2]
        self._shadow_border_mask_cache = (~rs) & nbr
        self._shadow_border_cells_arr  = np.argwhere(self._shadow_border_mask_cache)

    # ── union belief ──────────────────────────────────────────────────────────
    def _has_los(self, x0, y0, x1, y1, robot_inside_building: bool) -> bool:
        """
        Bresenham line-of-sight check from (x0,y0) to (x1,y1).
        Blocked by T_OBS walls.
        If the robot is outside a building, T_STAIRS cells also block LOS
        (can't see into a building from outside).
        If the robot is inside a building, T_STAIRS cells are transparent
        (can see other survivors in the same building interior).
        Does NOT check the start or end cell — only intermediate cells.
        """
        dx = abs(x1-x0); dy = abs(y1-y0)
        sx = 1 if x1 > x0 else -1
        sy = 1 if y1 > y0 else -1
        x, y = x0, y0
        err = dx - dy
        steps = dx + dy
        for _ in range(steps - 1):   # skip start and end
            e2 = 2 * err
            if e2 > -dy: err -= dy; x += sx
            if e2 <  dx: err += dx; y += sy
            if x == x1 and y == y1: break
            t = self.world.grid[x][y]["t"]
            if t == T_OBS: return False
            if t == T_STAIRS and not robot_inside_building: return False
        return True

    def _union_terrain(self):
        u = np.zeros((GRID_W, GRID_H), dtype=np.uint8)  # T_UNKNOWN = 0
        for r in self.robots:
            # Only write where we have new info and current is still unknown
            np.maximum(u, r.terrain_belief, out=u)
        return u

    def _union_temp(self):
        u = np.full((GRID_W, GRID_H), np.nan, dtype=np.float32)
        for r in self.robots:
            mask = r.known_mask & np.isnan(u)
            u[mask] = r.temp_belief[mask]
        return u

    def _union_rad(self):
        u = np.full((GRID_W, GRID_H), np.nan, dtype=np.float32)
        for r in self.robots:
            mask = r.known_mask & np.isnan(u)
            u[mask] = r.rad_belief[mask]
        return u

    # ── zone helpers ──────────────────────────────────────────────────────────
    def cell_to_zone(self, x, y):
        zx = x // self.zone_w_cells; zy = y // self.zone_h_cells
        if 0 <= zx < self.zone_nx and 0 <= zy < self.zone_ny:
            return (zx, zy)
        return None

    def zone_cells(self, zone):
        zx, zy = zone
        return (range(zx*self.zone_w_cells, min((zx+1)*self.zone_w_cells, GRID_W)),
                range(zy*self.zone_h_cells, min((zy+1)*self.zone_h_cells, GRID_H)))

    def zone_coverage(self, union, zone):
        zx, zy = zone
        x0 = zx*self.zone_w_cells; x1 = min(x0+self.zone_w_cells, GRID_W)
        y0 = zy*self.zone_h_cells; y1 = min(y0+self.zone_h_cells, GRID_H)
        slc = union[x0:x1, y0:y1]
        total = slc.size
        if total == 0: return 1.0
        known = int(np.count_nonzero(slc))  # T_UNKNOWN==0, so nonzero = known
        return known / total

    def zone_neighbors4(self, z):
        zx, zy = z
        return [(zx+dx, zy+dy) for dx,dy in NBR4
                if 0<=zx+dx<self.zone_nx and 0<=zy+dy<self.zone_ny]

    def relay_ok_extended(self, z):
        """
        A zone is comms-ok if it has a relay OR is reachable (through adjacent zones)
        from a zone that has a relay.  Uses the pre-computed flood-fill set.
        """
        if z is None: return False
        return self._relay_ok_flood.get(z, False)

    def _compute_relay_flood(self):
        """
        Flood relay_ok into adjacent shadow zones only.
        A relay outside shadow seeds its zone; coverage flows into neighbouring
        shadow zones (and chains through connected shadow zones).
        Open-air zones never propagate coverage — otherwise one relay anywhere
        floods all 64 zones via the open zone graph.
        """
        seeds = {z for z, ok in self.relay_ok.items() if ok}
        flooded = set(seeds)
        queue = deque(seeds)
        while queue:
            z = queue.popleft()
            for nz in self.zone_neighbors4(z):
                if nz not in flooded and self._shadow_frac_for_zone(nz) > 0.05:
                    flooded.add(nz)
                    queue.append(nz)
        self._relay_ok_flood = {z: (z in flooded) for z in self.relay_ok}

    def zone_has_outside_relay(self, zone):
        """
        True if any relay robot is positioned outside shadow in `zone` OR
        in any zone adjacent to `zone` (4-connectivity), AND that relay's
        task_zone is in the same shadow cluster as `zone`.
        """
        # Find the shadow cluster containing `zone`
        sf = self._shadow_frac_for_zone(zone)
        if sf <= 0.05:
            return False  # not a shadow zone, relay not needed
        # BFS to find all zones in same shadow cluster
        cluster = set()
        q = deque([zone]); cluster.add(zone)
        while q:
            z2 = q.popleft()
            for nz in self.zone_neighbors4(z2):
                if nz not in cluster and self._shadow_frac_for_zone(nz) > 0.05:
                    cluster.add(nz); q.append(nz)

        check_zones = {zone} | set(self.zone_neighbors4(zone))
        for r in self.robots:
            if not r.active or r.role != Role.RELAY: continue
            rz = self.cell_to_zone(r.pos[0], r.pos[1])
            if rz not in check_zones: continue
            if self.radio_shadow[r.pos[0], r.pos[1]]: continue
            # Relay must be serving this cluster (its task_zone must be in cluster)
            if r.task_zone is not None and r.task_zone in cluster:
                return True
        return False

    # ── zone stats (cached per tick) ──────────────────────────────────────────
    def zone_stats(self, zone):
        t = self.timestep
        if self._zone_stats_cache_tick != t:
            self._zone_stats_cache_tick = t
            self._zone_stats_cache.clear()
        if zone in self._zone_stats_cache:
            return self._zone_stats_cache[zone]

        zx, zy = zone
        xs, ys = self.zone_cells(zone)
        total=unknown=known=sumT=sumR=0
        tc = {T_FREE:0, T_WATER:0, T_STAIRS:0, T_OBS:0}
        union = self.union_belief; uT = self.union_T; uR = self.union_R

        for x in xs:
            for y in ys:
                total += 1; tb = union[x,y]
                if tb == T_UNKNOWN: unknown += 1
                else:
                    known += 1
                    if tb in tc: tc[tb] += 1
                    t_ = uT[x,y]; r_ = uR[x,y]
                    if not np.isnan(t_): sumT += float(t_)
                    if not np.isnan(r_): sumR += float(r_)

        uf = unknown/total if total>0 else 0.0
        avgT = sumT/known if known>0 else 0.0
        avgR = sumR/known if known>0 else 0.0
        fw   = tc[T_WATER]/known  if known>0 else 0.0
        fs   = tc[T_STAIRS]/known if known>0 else 0.0
        ff   = tc[T_FREE]/known   if known>0 else 0.0
        # shadow_frac counts ALL cells (including unknown) because the radio shadow
        # is a physical property of the environment, not a belief.  Using only
        # known cells gives sf=0 for unexplored shadow zones, breaking the relay bonus.
        shadow_all = sum(1 for x in xs for y in ys if self.radio_shadow[x,y])
        sf   = shadow_all/total if total>0 else 0.0
        cx   = zx*self.zone_w_cells + self.zone_w_cells//2
        cy   = zy*self.zone_h_cells + self.zone_h_cells//2
        stats = dict(unknown_frac=uf, avgT=avgT, avgR=avgR,
                     f_water=fw, f_stairs=fs, f_free=ff,
                     shadow_frac=sf, center=(cx,cy),
                     known=known, total=total)
        self._zone_stats_cache[zone] = stats
        return stats

    def zone_feasible(self, r, stats, zone=None):
        """
        Whether robot r can meaningfully work this zone.
        Note: we deliberately do NOT block on relay_ok=False here.
        A robot bidding on a shadow zone may become the relay for it.
        """
        if not r.active or r.battery <= 0: return False
        uf = stats["unknown_frac"]; fw = stats["f_water"]; fs = stats["f_stairs"]

        if r.name.startswith("Boat"):
            # Boat needs actual reachable water in this zone — not just water fraction
            # or high unknown fraction. A zone with uf=0.98 but no water is useless.
            if zone is None: return fw > 0.05
            r.reachable()  # ensure cache is warm
            if r._reachable_arr is None: return False
            zx, zy = zone
            x0 = zx*self.zone_w_cells; x1 = min(x0+self.zone_w_cells, GRID_W)
            y0 = zy*self.zone_h_cells; y1 = min(y0+self.zone_h_cells, GRID_H)
            reach_slice = r._reachable_arr[x0:x1, y0:y1]
            water_slice = self._world_water_arr[x0:x1, y0:y1]
            return bool(np.any(reach_slice & water_slice))

        if r.name.startswith("Legged"):
            return not (fw > 0.30 and fs < 0.05 and uf < 0.60)
        return True

    # ── zone frontiers (shared cache) ─────────────────────────────────────────
    def zone_frontiers_for(self, robot, zone) -> list:
        """Return frontiers in zone for robot, using robot's local belief + confidence."""
        def reachable_fn(mask):
            if mask not in self._reachable_by_mask:
                robot.reachable()  # ensure BFS is run and _reachable_arr is populated
                # Pass the bool array directly — O(1) membership vs O(1) set lookup
                # but avoids the expensive set construction for large maps
                self._reachable_by_mask[mask] = robot._reachable_arr
            return self._reachable_by_mask[mask]

        return self._frontier_cache.get(
            zone, robot.caps_mask,
            robot.terrain_belief,
            reachable_fn,
            self.world,
            self.zone_cells,
            confidence=robot.confidence,
        )

    # ── CBBA ──────────────────────────────────────────────────────────────────
    def _assign_zones_cbba(self):
        """
        CBBA zone assignment.
        Key fix: consensus loop correctly operates per-zone.
        Key fix: only reset bundles for robots NOT currently mid-task.
        """
        # Only fully reset robots that are idle or done with their zone
        for r in self.robots:
            if not r.active or r.battery <= 0: continue
            zone_done = (r.task_zone is None or
                         self.zone_coverage(self.union_belief, r.task_zone) >= ZONE_DONE)
            if zone_done:
                r.bundle = []; r.assigned_zones = []
            # Robots mid-task keep their bundle; CBBA may add/replace other zones

        zones = [(zx,zy) for zx in range(self.zone_nx) for zy in range(self.zone_ny)]

        # refresh zone task progress
        for z, task in self.zone_tasks.items():
            task.progress = self.zone_coverage(self.union_belief, z)
            if task.progress >= ZONE_DONE and task.status != "blacklisted":
                task.owners = []; task.status = "released"; task.expires_at = 0

        bundle_counts = {r.name: len(r.bundle) for r in self.robots}

        for _it in range(CBBA_ITERS):
            # 1) each robot bids and greedily fills bundle
            for r in self.robots:
                if not r.active or r.battery <= 0: continue
                bundle_counts[r.name] = len(r.bundle)
                bids = []
                for z in zones:
                    u = self._zone_utility(r, z, bundle_counts)
                    if u is not None:
                        bids.append((u, z))
                bids.sort(reverse=True)
                for u, z in bids:
                    if len(r.bundle) >= MAX_BUNDLE: break
                    if z not in r.bundle:
                        r.bundle.append(z)

            # 2) consensus: per-zone conflict resolution (FIXED loop scope)
            zone_claims: dict = {}
            for r in self.robots:
                if not r.active: continue
                for z in r.bundle:
                    u = self._zone_utility(r, z, bundle_counts)
                    if u is not None:
                        zone_claims.setdefault(z, []).append((r.name, u))

            # resolve each zone independently
            for z, claims in zone_claims.items():
                if len(claims) <= 1: continue
                claims.sort(key=lambda t: t[1], reverse=True)

                # enforce heterogeneity: one robot per locomotion type per zone
                winners = []; used_types = set()
                for nm, u in claims:
                    rr = next((x for x in self.robots if x.name==nm), None)
                    if rr is None: continue
                    rt = robot_type(nm)
                    if rt in used_types: continue
                    winners.append(nm); used_types.add(rt)
                    if len(winners) >= ZONE_CAPACITY: break

                winner_set = set(winners)
                for r in self.robots:
                    if r.name not in winner_set and z in r.bundle:
                        r.bundle = [zz for zz in r.bundle if zz != z]

        # 3) finalise: bundle -> assigned_zones, update zone_tasks
        for z, t in self.zone_tasks.items():
            if t.status != "blacklisted": t.owners = []

        for r in self.robots:
            r.assigned_zones = list(r.bundle)
            for z in r.assigned_zones:
                t = self.zone_tasks[z]
                if t.status == "blacklisted": continue
                if r.name not in t.owners and len(t.owners) < ZONE_CAPACITY:
                    t.owners.append(r.name)
                t.status = "held"; t.expires_at = self.timestep + LEASE_T

        # fallback: every active robot gets at least one zone
        for r in self.robots:
            if not r.active or r.battery <= 0: continue
            if not r.bundle:
                z = self._fallback_zone(r)
                if z: r.bundle.append(z); r.assigned_zones.append(z)

    def _zone_utility(self, r, zone, bundle_counts):
        """
        Hybrid CBBA utility — sim-to-real information model.

        If the robot has personally scanned >= ZONE_PERSONAL_THRESH of the zone,
        it bids using full zone_stats (terrain, hazard, shadow fraction) from the
        shared comms-gated belief — it has ground truth for this area.

        If the robot has NOT personally explored the zone, it can only bid on:
          - Distance (travel cost)
          - Estimated information gain (fraction of zone still uncertain in its
            local belief, including confidence-decayed cells)
          - Whether the zone is in shadow (relay needed bonus)
        This prevents robots from beelining to shadow zones they've never seen.
        """
        if self.zone_coverage(self.union_belief, zone) >= ZONE_DONE: return None
        if r.blacklist.get(zone, -1) > self.timestep: return None
        t = self.zone_tasks.get(zone)

        # How much has THIS robot personally scanned in this zone?
        zx, zy = zone
        x0 = zx*self.zone_w_cells; x1 = min(x0+self.zone_w_cells, GRID_W)
        y0 = zy*self.zone_h_cells; y1 = min(y0+self.zone_h_cells, GRID_H)
        zone_cells_total = (x1-x0) * (y1-y0)
        personal_scanned = int(np.count_nonzero(r.personally_scanned[x0:x1, y0:y1]))
        personal_frac    = personal_scanned / max(1, zone_cells_total)

        # Cells that are known but confidence-decayed — treated as uncertain
        # for exploration value (they need revisiting)
        known_slice = r.known_mask[x0:x1, y0:y1]
        conf_slice  = r.confidence[x0:x1, y0:y1]
        uncertain_known = int(np.count_nonzero(known_slice & (conf_slice < CONF_UNCERTAIN)))
        unknown_slice   = ~known_slice
        local_unknown_frac = (np.count_nonzero(unknown_slice) + uncertain_known) / zone_cells_total

        # Zone centre for distance calculation
        cx = x0 + (x1-x0)//2; cy = y0 + (y1-y0)//2
        dist   = abs(r.pos[0]-cx) + abs(r.pos[1]-cy)
        travel = dist / float(GRID_W + GRID_H)
        load   = 0.25 * bundle_counts.get(r.name, 0)

        # Shadow status — always observable (shadow is a radio property, not a terrain secret)
        shadow_frac = self._shadow_frac_for_zone(zone)
        relay_needed = shadow_frac > 0.05 and not self._relay_ok_flood.get(zone, False)
        relay_active = shadow_frac > 0.05 and self._relay_ok_flood.get(zone, False)

        # Don't count relay robots toward zone capacity — they're parked outside
        # serving the zone, not consuming exploration bandwidth inside it.
        if t and t.status == "held" and self.timestep < t.expires_at:
            relay_owners = [nm for nm in t.owners
                            if any(rr.name == nm and rr.role == Role.RELAY
                                   for rr in self.robots)]
            effective_owners = len(t.owners) - len(relay_owners)
            if r.name not in t.owners and effective_owners >= ZONE_CAPACITY:
                return None

        if personal_frac >= ZONE_PERSONAL_THRESH:
            # ── KNOWN ZONE: bid on full stats from shared belief ──────────────
            stats = self.zone_stats(zone)
            if not self.zone_feasible(r, stats, zone=zone): return None
            if r.name.startswith("Boat") and stats["f_water"]<=0.0 and stats["unknown_frac"]<0.40:
                return None

            uf = stats["unknown_frac"]; avgT = stats["avgT"]; avgR = stats["avgR"]
            fw = stats["f_water"];      fs   = stats["f_stairs"]
            sf = stats["shadow_frac"]

            eT = avgT / max(1e-6, r.temp_limit)
            eR = avgR / max(1e-6, r.rad_limit)
            lam = float(r.weights[0])*eT + float(r.weights[1])*eR
            risk_term = (-(abs(lam)**P) if r.name.startswith(("Legged","Drone"))
                         else +(abs(lam)**P) if r.name.startswith("Rover") else 0.0)

            terrain_term = 0.0
            if r.name.startswith("Boat"):    terrain_term += 2.0*fw - 0.5*(1-fw)
            elif r.name.startswith("Legged"):terrain_term += 2.0*fs - 1.0*fw
            elif r.name.startswith("Drone"): terrain_term += 1.0*fw + 0.5*fs
            elif r.name.startswith("Rover"): terrain_term -= 0.5*fw

            critical = 0.0
            if r.name.startswith(("Legged","Drone")): critical += 1.5*fs
            if r.name.startswith("Boat"):              critical += 1.5*fw

            shadow_bonus       = 0.8 * sf * uf
            relay_needed_bonus = (3.5 * sf * max(uf, 0.3)) if relay_needed else 0.0
            # Strong bonus when relay is active — pull explorers INTO the bubble
            relay_explorer_bonus = (4.0 * sf * max(uf, 0.5)) if relay_active else 0.0

            u = (1.0*uf + risk_term + terrain_term - 0.05*travel
                 + critical + shadow_bonus + relay_needed_bonus + relay_explorer_bonus - load)

        else:
            # ── UNKNOWN ZONE: bid on information-gain potential only ──────────
            # Robot has not explored this zone — it cannot know terrain or hazards.
            # It bids based purely on how much it doesn't know + distance cost.
            # This is the key sim-to-real fix: no oracle beelining.
            if local_unknown_frac < 0.05: return None   # nothing to learn here

            # Flat terrain feasibility prior (can't know specifics)
            # Boats avoid zones unlikely to have water (far from map edge/centre water)
            # — approximated by: if robot has never seen water anywhere nearby, skip
            if r.name.startswith("Boat"):
                nearby_water = int(np.count_nonzero(
                    r.terrain_belief[max(0,cx-20):cx+20, max(0,cy-20):cy+20] == T_WATER))
                if nearby_water == 0: return None

            # Information gain: unknown fraction in robot's local belief
            info_gain = local_unknown_frac

            # Shadow relay bonus still applies — robot knows the radio topology
            relay_needed_bonus = (2.0 * shadow_frac) if relay_needed else 0.0
            # Pull explorers into bubble when relay is active
            relay_explorer_bonus = (3.0 * shadow_frac * local_unknown_frac) if relay_active else 0.0

            # Slight diversity bonus: prefer zones far from current task
            diversity = 0.1 * travel

            u = (1.5 * info_gain - 0.08*travel + relay_needed_bonus + relay_explorer_bonus - diversity - load)

        return float(u)

    def _shadow_frac_for_zone(self, zone):
        """O(1) lookup into precomputed table (shadow never changes)."""
        return self._zone_shadow_frac.get(zone, 0.0)

    def _local_zone_unknown_frac(self, robot, zone) -> float:
        """
        Fraction of zone cells that are either unknown OR confidence-decayed
        in this robot's local belief.  Used for local planner coverage decisions.
        """
        zx, zy = zone
        x0 = zx*self.zone_w_cells; x1 = min(x0+self.zone_w_cells, GRID_W)
        y0 = zy*self.zone_h_cells; y1 = min(y0+self.zone_h_cells, GRID_H)
        total = (x1-x0)*(y1-y0)
        if total == 0: return 0.0
        tb_slc   = robot.terrain_belief[x0:x1, y0:y1]
        conf_slc = robot.confidence[x0:x1, y0:y1]
        unknown  = int(np.count_nonzero(tb_slc == T_UNKNOWN))
        stale    = int(np.count_nonzero((tb_slc != T_UNKNOWN) & (conf_slc < CONF_UNCERTAIN)))
        return (unknown + stale) / total

    def _fallback_zone(self, r):
        best_z = None; best_u = -1e18
        counts = {rr.name: len(rr.bundle) for rr in self.robots}
        for z in [(zx,zy) for zx in range(self.zone_nx) for zy in range(self.zone_ny)]:
            u = self._zone_utility(r, z, counts)
            if u is not None and u > best_u:
                best_u = u; best_z = z
        return best_z

    # ── role assignment ───────────────────────────────────────────────────────
    def _decide_roles(self):
        """
        Assign relay roles globally.

        For each disconnected shadow cluster that has unexplored cells and no relay:
          - Find the closest non-shadow, non-locked active robot
          - Elect it as relay; it will navigate to the shadow border

        Demote relays whose shadow cluster is fully explored or has had no
        explorers inside for RELAY_IDLE_TICKS.

        All other role assignment (SCOUT/SCAN/LOITER) happens implicitly through
        the CBBA task auction — robots bid on shadow zones once relay_ok=True.
        """
        t = self.timestep
        active = [r for r in self.robots if r.active and r.battery > 0]
        shadow = self.radio_shadow

        # ── 1. Find all shadow clusters (connected components of shadow zones) ──
        shadow_zones = [(zx, zy)
                        for zx in range(self.zone_nx)
                        for zy in range(self.zone_ny)
                        if self._shadow_frac_for_zone((zx, zy)) > 0.05]

        visited_z = set(); clusters = []
        for sz in shadow_zones:
            if sz in visited_z: continue
            cluster = []; q = deque([sz]); visited_z.add(sz)
            while q:
                z2 = q.popleft(); cluster.append(z2)
                for nz in self.zone_neighbors4(z2):
                    if nz not in visited_z and self._shadow_frac_for_zone(nz) > 0.05:
                        visited_z.add(nz); q.append(nz)
            clusters.append(cluster)

        # ── 2. Demote existing relays that are no longer needed ──
        current_relays = [r for r in active if r.role == Role.RELAY]
        for r in current_relays:
            # Safety: relay must be outside shadow
            if shadow[r.pos[0], r.pos[1]]:
                r.role = Role.SCAN; r.relay_hold_until = 0
                r.role_locked_until = t + 10; continue

            # Find which cluster this relay is serving
            serving_cluster = None
            for cl in clusters:
                if any(self.zone_has_outside_relay(z) for z in cl):
                    serving_cluster = cl; break

            if serving_cluster is None:
                # No cluster needs this relay
                if t >= r.relay_hold_until:
                    r.role = Role.SCAN; r.role_locked_until = t + 20
                continue

            cluster_cov = max(
                self.zone_coverage(self.union_belief, z)
                for z in serving_cluster)
            if cluster_cov >= ZONE_DONE:
                r.role = Role.SCAN; r.relay_hold_until = 0
                r.role_locked_until = t + 20; continue

            # Never demote if explorers are currently inside this shadow cluster
            cluster_cells = set()
            for z in serving_cluster:
                zx2, zy2 = z
                x0c = zx2*self.zone_w_cells; x1c = min(x0c+self.zone_w_cells, self.world.w)
                y0c = zy2*self.zone_h_cells; y1c = min(y0c+self.zone_h_cells, self.world.h)
                # Check if any non-relay robot is inside shadow cells of this zone
            robots_in_cluster_shadow = any(
                ro is not r and ro.active and ro.role != Role.RELAY
                and shadow[ro.pos[0], ro.pos[1]]
                and self.cell_to_zone(ro.pos[0], ro.pos[1]) in set(serving_cluster)
                for ro in self.robots
            )
            if robots_in_cluster_shadow:
                r.relay_last_occupied = t
                r.role_locked_until = t + 5
                continue

            # Idle: no explorer in bubble for too long
            if (r.relay_last_occupied > 0
                    and (t - r.relay_last_occupied) > RELAY_IDLE_TICKS
                    and t >= r.relay_hold_until):
                r.role = Role.SCAN; r.relay_hold_until = 0
                r.role_locked_until = t + 20; continue

            # Update idle tracker
            for ro in active:
                if ro is not r and shadow[ro.pos[0], ro.pos[1]]:
                    r.relay_last_occupied = t

            # Still needed — refresh hold
            r.role_locked_until = t + 5

        # ── 3. Elect new relays for uncovered clusters ──
        # A cluster is "covered" if a relay is already en-route OR parked at border
        # Key: check task_zone assignment, not just physical position
        for cluster in clusters:
            cluster_cov = max(
                (self.zone_coverage(self.union_belief, z) for z in cluster),
                default=0.0)
            if cluster_cov >= ZONE_DONE:
                continue

            # Already covered: relay physically at border OR relay assigned to this cluster
            already_covered = any(self.zone_has_outside_relay(z) for z in cluster)
            if not already_covered:
                cluster_set = set(cluster)
                # Also covered if a relay robot's task_zone is in this cluster
                already_covered = any(
                    r.role == Role.RELAY and r.task_zone in cluster_set
                    for r in active
                )
            if already_covered:
                continue

            # Cluster centre (for distance scoring)
            all_z = cluster
            cx_cells = np.mean([z[0]*self.zone_w_cells + self.zone_w_cells//2 for z in all_z])
            cy_cells = np.mean([z[1]*self.zone_h_cells + self.zone_h_cells//2 for z in all_z])

            # Candidates: outside shadow, not locked, not already relay
            # (Skip expensive reachability check here — relay will self-demote
            #  if it can't find a path to the border in _move_relay)
            candidates = [
                r for r in active
                if r.role != Role.RELAY
                and t >= r.role_locked_until
                and not shadow[r.pos[0], r.pos[1]]
            ]

            if not candidates:
                continue

            # Pick nearest to cluster centre
            def relay_score(r):
                return abs(r.pos[0] - cx_cells) + abs(r.pos[1] - cy_cells)
            best = min(candidates, key=relay_score)

            # Assign relay — set task_zone to the most uncovered shadow zone
            best_zone = min(cluster,
                            key=lambda z: self.zone_coverage(self.union_belief, z))
            best.role = Role.RELAY
            best.task_zone = best_zone
            best.relay_hold_until   = t + RELAY_MIN_HOLD
            best.role_locked_until  = t + RELAY_MIN_HOLD
            best.relay_last_occupied = t
            best.relay_anchor = None  # reset anchor so _move_relay recomputes

        # ── 4. Global fleet cap ──
        max_relays = max(1, int(len(active) * RELAY_MAX_FLEET_FRAC))
        all_relays = [r for r in active if r.role == Role.RELAY]
        if len(all_relays) > max_relays:
            # Keep relays serving the most unexplored clusters
            def relay_priority(rr):
                if rr.task_zone is None: return 0.0
                cov = self.zone_coverage(self.union_belief, rr.task_zone)
                sf = self._shadow_frac_for_zone(rr.task_zone)
                return sf * (1.0 - cov)
            all_relays.sort(key=relay_priority)  # lowest priority first
            for rr in all_relays[:len(all_relays) - max_relays]:
                if t >= rr.relay_hold_until:
                    rr.role = Role.SCAN; rr.role_locked_until = t + 20

    def _global_relay_count(self):
        """Count active relays across the entire fleet."""
        return sum(1 for rr in self.robots if rr.active and rr.role == Role.RELAY)

    def _can_reach_shadow_border(self, r, zone) -> bool:
        """True if r can reach any shadow-border cell near `zone`."""
        r.reachable()
        reach_arr = r._reachable_arr
        if reach_arr is None:
            return False
        W, H = self.world.w, self.world.h
        zx, zy = zone
        x0 = max(0, (zx-1)*self.zone_w_cells)
        x1 = min(W,  (zx+2)*self.zone_w_cells)
        y0 = max(0, (zy-1)*self.zone_h_cells)
        y1 = min(H,  (zy+2)*self.zone_h_cells)
        # Use precomputed border mask
        candidates = reach_arr[x0:x1,y0:y1] & self._shadow_border_mask_cache[x0:x1,y0:y1]
        return bool(np.any(candidates))

    def _role_utility(self, r, role, zone, role_counts, stats):
        """Utility for SCOUT/SCAN/LOITER roles. RELAY is handled by _best_response_roles."""
        uf = stats["unknown_frac"]
        sf = stats["shadow_frac"]
        eT = stats["avgT"] / max(1e-6, r.temp_limit)
        eR = stats["avgR"] / max(1e-6, r.rad_limit)
        risk = max(eT, eR)
        m = r.zone_frontier_signal

        if role == Role.RELAY:
            return -1e9   # relay election handled separately — never elected here

        U = 0.0
        U += {Role.SCOUT:1.6, Role.SCAN:1.2, Role.LOITER:0.1}[role] * uf
        U += (-0.2 if r.name.startswith("Rover") else -1.0) * risk**2
        U -= 0.7 * role_counts.get(role, 0)

        if role == Role.LOITER:
            U -= 4.0*m + 1.5*uf

        if role == Role.SCOUT and r.name.startswith("Drone"):
            U += 0.5

        return U

    def _best_response_roles(self, zone, robots, stats):
        """
        Relay semantics (see design intent):
          - A relay is needed when ANY shadow zone adjacent to this zone (or this
            zone itself) has unexplored cells with no relay coverage.
          - Elected relay moves to the shadow border (outside shadow, touching it).
          - Relay stays until: zone cluster fully explored, OR no robot has been
            inside bubble for RELAY_IDLE_TICKS, OR minimum hold expires.
          - All other robots in zone explore freely while relay is active.
        """
        t = self.timestep
        sf  = stats["shadow_frac"]
        zone_cov = self.zone_coverage(self.union_belief, zone)

        # Reset unlocked robots
        for r in robots:
            if t >= r.role_locked_until:
                r.role = Role.SCAN

        # Track robots inside shadow (for idle timer)
        for r in robots:
            if r.role == Role.RELAY:
                for ro in robots:
                    if ro is not r and ro.active and self.radio_shadow[ro.pos[0], ro.pos[1]]:
                        r.relay_last_occupied = t

        # Demote relays that are no longer needed
        for r in list(robots):
            if r.role != Role.RELAY: continue
            if self.radio_shadow[r.pos[0], r.pos[1]]:   # relay ended up in shadow
                r.role = Role.SCAN; r.relay_hold_until = 0
                r.role_locked_until = t + 10; continue
            if zone_cov >= ZONE_DONE:
                r.role = Role.SCAN; r.relay_hold_until = 0
                r.role_locked_until = t + 20; continue
            if (r.relay_last_occupied > 0
                    and (t - r.relay_last_occupied) > RELAY_IDLE_TICKS
                    and t >= r.relay_hold_until):
                r.role = Role.SCAN; r.relay_hold_until = 0
                r.role_locked_until = t + 20; continue
            r.role_locked_until = t + 5   # keep locked while serving

        # Does this zone's shadow cluster need a relay?
        # Check zone itself + all neighbours for uncovered shadow
        def cluster_needs_relay():
            zones_to_check = [zone] + list(self.zone_neighbors4(zone))
            for z in zones_to_check:
                if (self._shadow_frac_for_zone(z) > 0.05
                        and not self.relay_ok.get(z, False)
                        and self.zone_coverage(self.union_belief, z) < ZONE_DONE):
                    return True
            return False

        current_relays = [r for r in robots if r.role == Role.RELAY]

        if not current_relays and cluster_needs_relay():
            candidates = [r for r in robots
                          if t >= r.role_locked_until
                          and not self.radio_shadow[r.pos[0], r.pos[1]]
                          and self._can_reach_shadow_border(r, zone)]
            if candidates:
                shadow_cells = self._shadow_cells_arr
                def relay_score(r):
                    if len(shadow_cells) == 0: return 9999
                    d = np.abs(shadow_cells[:,0]-r.pos[0]) + np.abs(shadow_cells[:,1]-r.pos[1])
                    return int(np.min(d))
                best = min(candidates, key=relay_score)
                best.role = Role.RELAY
                best.relay_hold_until   = t + RELAY_MIN_HOLD
                best.role_locked_until  = t + RELAY_MIN_HOLD
                best.relay_last_occupied = t

        # Hard cap
        relays = [r for r in robots if r.role == Role.RELAY]
        if len(relays) > RELAY_MAX_PER_ZONE:
            relays.sort(key=lambda rr: rr.relay_hold_until, reverse=True)
            for rr in relays[RELAY_MAX_PER_ZONE:]:
                if t >= rr.relay_hold_until:
                    rr.role = Role.SCAN; rr.role_locked_until = t + 10

    # ── relay anchor logic ────────────────────────────────────────────────────
    def _move_relay(self, r, occupied):
        """
        Move relay to the shadow border cell nearest to its task_zone centre,
        reachable without entering shadow.  Once there, hold position.
        """
        if r.task_zone is None:
            return

        W, H   = self.world.w, self.world.h
        shadow = self.radio_shadow
        tb_arr = r.terrain_belief
        mask4  = r.caps_mask & 0xF
        trav   = _TRAV_LUT

        # Shadow-border mask: outside shadow AND has a shadow-cell neighbour (cached)
        border_mask = self._shadow_border_mask_cache

        # If already on a border cell — hold
        if border_mask[r.pos[0], r.pos[1]]:
            r._reveal_all(); r._recompute_chunked()
            return

        # Task zone centre (target area)
        zx, zy = r.task_zone
        cx = zx * self.zone_w_cells + self.zone_w_cells // 2
        cy = zy * self.zone_h_cells + self.zone_h_cells // 2

        # Recompute anchor only if needed (zone changed or lost anchor)
        if r.relay_anchor_zone != r.task_zone or r.relay_anchor is None:
            # BFS from task_zone centre outward through all traversable (including shadow)
            # to find the closest non-shadow border cell to the zone centre.
            # We search from the ZONE CENTRE, not from the robot position.
            # This ensures the relay always targets the correct shadow region.
            best_border = None; best_d = 1e9
            for pt in self._shadow_border_cells_arr:
                bx, by = int(pt[0]), int(pt[1])
                d = abs(bx - cx) + abs(by - cy)
                if d < best_d:
                    best_d = d; best_border = (bx, by)

            if best_border is None:
                r.role = Role.SCAN; r.relay_hold_until = 0; return

            r.relay_anchor = best_border
            r.relay_anchor_zone = r.task_zone

        target = r.relay_anchor

        # Plan shadow-free path to anchor
        need_replan = (r.goal != target or not r.path)
        if not need_replan:
            for cell in r.path[:4]:
                if shadow[cell[0], cell[1]]: need_replan = True; break

        if need_replan:
            unk_pen, info_w, a_mult, b_mult = r._planner_params()
            path = AStar.search(
                start=r.pos, goal=target,
                caps_mask=r.caps_mask, terrain_u8=tb_arr,
                temp_f32=r.temp_belief, rad_f32=r.rad_belief,
                chunked_risk=r.chunked,
                temp_limit=r.temp_limit, rad_limit=r.rad_limit,
                radio_shadow=shadow, relay_ok_fn=lambda z: False,
                cell_to_zone_fn=self.cell_to_zone,
                global_cov=self.global_cov,
                unk_pen=unk_pen, info_w=info_w, unk_prior=UNK_PRIOR,
                alpha_mult=a_mult, beta_mult=b_mult, soft_frac=SOFT_FRAC,
                traffic_u16=self.traffic_u16, traffic_w=TRAFFIC_W,
            )
            if not path:
                # Can't reach — try a different border cell next tick
                r.relay_anchor = None; r.relay_anchor_zone = None
                return
            r.goal = target; r.path = path; r.goal_commit = 20

        if r.path and r.path[0] not in occupied:
            next_cell = r.path[0]
            if shadow[next_cell[0], next_cell[1]]:
                r.relay_anchor = None; return  # anchor crept into shadow, replan
            # Hard terrain safety: relay must not walk into water/OBS either
            true_t = self.world.grid[next_cell[0]][next_cell[1]]["t"]
            if true_t == T_OBS:
                r.relay_anchor = None; r.path = []; return
            if true_t == T_WATER and not bool(r.caps_mask & (CAP_WATER|CAP_AIR)):
                # Update terrain belief and force anchor reselection
                r.terrain_belief[next_cell[0],next_cell[1]] = true_t
                r.known_mask[next_cell[0],next_cell[1]] = True
                r.relay_anchor = None; r.path = []; return
            occupied.discard(r.pos); occupied.add(next_cell)
            r.pos = r.path.pop(0)
            r._reveal_all(); r._recompute_chunked()
            drain = {"Legged": 1.0, "Drone": 2.0, "Boat": 2.0, "Rover": 0.4}
            r.battery -= drain.get(robot_type(r.name), 1.0) * 1.1

    # ── choose exploration goal ───────────────────────────────────────────────
    def _choose_goal(self, r) -> tuple | None:
        """
        Pick the best frontier for robot r given its task_zone.
        Scores by: info-gain (unknown neighbours revealed) + distance + chunk risk.
        """
        union = self.union_belief
        r.reachable()  # ensure BFS run; _reachable_arr populated
        reach_arr = r._reachable_arr  # bool array for fast membership

        # frontiers in assigned zone (preferred), then global
        if r.task_zone is not None:
            candidates = self.zone_frontiers_for(r, r.task_zone)
        else:
            candidates = []

        if not candidates and reach_arr is not None:
            # global frontiers from reachable array
            is_boat = bool(r.caps_mask & CAP_WATER) and not bool(r.caps_mask & CAP_AIR)
            candidates = []
            reach_coords = np.argwhere(reach_arr)
            for pos in reach_coords:
                x, y = int(pos[0]), int(pos[1])
                if is_boat:
                    if union[x,y] in (T_WATER,T_BRIDGE) and \
                       any(union[nx,ny]==T_UNKNOWN for nx,ny in self.world.neighbours((x,y))):
                        candidates.append((x,y))
                else:
                    if union[x,y] == T_UNKNOWN:
                        candidates.append((x,y))

        if not candidates and reach_arr is not None:
            # last resort: reveal-frontiers (known cells adjacent to unknown)
            reach_coords = np.argwhere(reach_arr)
            candidates = [(int(p[0]),int(p[1])) for p in reach_coords
                          if union[int(p[0]),int(p[1])] != T_UNKNOWN
                          and any(union[nx,ny]==T_UNKNOWN
                                  for nx,ny in self.world.neighbours((int(p[0]),int(p[1]))))]

        if not candidates: return None

        # filter failed goals
        candidates = [c for c in candidates if self.timestep >= r.failed_goals.get(c, 0)]
        if not candidates: return None

        # score: lower = better goal
        def score(c):
            x, y = c
            dist = abs(r.pos[0]-x) + abs(r.pos[1]-y)

            # info gain: how many unknown neighbours will stepping here reveal?
            info = sum(1 for nx,ny in self.world.neighbours((x,y))
                       if union[nx,ny]==T_UNKNOWN)
            if union[x,y]==T_UNKNOWN:
                info += 4  # stepping onto unknown cell itself is high gain

            cx2, cy2 = x//CHUNK_SIZE, y//CHUNK_SIZE
            lT = float(r.chunked[0,cx2,cy2]); lR = float(r.chunked[1,cx2,cy2])
            eT = lT/max(1e-6,r.temp_limit);   eR = lR/max(1e-6,r.rad_limit)
            e  = max(eT, eR)

            # rover seeks risk, others avoid
            risk_term = -(ALPHA*(e**P)) if r.name.startswith("Rover") else (ALPHA*(e**P))

            # crowd penalty: other robots already targeting nearby
            # Uses current r.goal values — earlier robots in the tick have already
            # registered their intent, so this naturally produces dispersion.
            crowd = sum(1.0 for rr in self.robots
                        if rr is not r and rr.active and rr.goal is not None
                        and abs(rr.goal[0]-x)+abs(rr.goal[1]-y) < 5)

            return dist - 2.5*info + risk_term + 5.0*crowd

        return min(candidates, key=score)

    # ── release / blacklist ───────────────────────────────────────────────────
    def _release_zone(self, r, reason=""):
        z = r.task_zone
        if z is None: return
        t = self.zone_tasks.get(z)
        if t:
            if r.name in t.owners: t.owners.remove(r.name)
            t.last_owner = r.name; t.last_release_reason = reason
            t.status = "held" if t.owners else "released"
            if not t.owners: t.expires_at = 0
        if z in r.bundle: r.bundle = [zz for zz in r.bundle if zz!=z]
        if z in r.assigned_zones: r.assigned_zones.remove(z)
        r.task_zone=None; r.zone_lease_until=0
        r.task_no_progress=0; r.task_last_known=0
        r.goal=None; r.path=[]; r.goal_commit=0

    def _blacklist_zone(self, r, z, reason=""):
        if z is None: return
        self._release_zone(r, reason)
        r.blacklist[z] = self.timestep + COOLDOWN_T
        t = self.zone_tasks.get(z)
        if t and r.name in t.owners: t.owners.remove(r.name)

    # ── main step ─────────────────────────────────────────────────────────────
    def step(self) -> bool:
        self.timestep += 1

        # ── CBBA reassignment every 50 ticks ──
        if self.timestep % 50 == 0:
            self._assign_zones_cbba()

        # ── rebuild per-tick caches ──
        self._frontier_cache.clear()
        self._reachable_by_mask.clear()
        self._zone_stats_cache_tick = -1  # force zone stats refresh

        # ── traffic map ──
        self.traffic_u16.fill(0)
        for r in self.robots:
            if r.active and r.path:
                x0,y0 = r.pos
                self.traffic_u16[x0,y0] = min(65535, int(self.traffic_u16[x0,y0])+2)
                for px,py in r.path[:TRAFFIC_LOOKAHEAD]:
                    self.traffic_u16[px,py] = min(65535, int(self.traffic_u16[px,py])+1)

        # ── global coverage ──
        union = self.union_belief
        self.global_cov = float(np.mean(union != T_UNKNOWN))

        # ── comms: centralised planner model ────────────────────────────────────
        # Architecture: robots broadcast sensor data to all reachable peers
        # instantly (radio propagation << 1 tick). The constraint is NOT bandwidth
        # or mesh hops — it's the radio shadow blackout.
        #
        # Rules:
        #   1. Robot in open (not shadow): shares with all other open robots +
        #      any relay-bridged shadow robots, instantly this tick.
        #   2. Robot in shadow WITH relay coverage: data reaches base via relay
        #      chain. We model the chain latency as RELAY_COMMS_DELAY ticks.
        #   3. Robot in shadow WITHOUT relay: data stays local until it exits
        #      or a relay is established.
        #
        # Each robot's outbox holds pending messages with a `deliver_at` tick.
        # FleetSim delivers them when timestep >= deliver_at AND the robot is
        # comms-capable at that point.

        active_robots = [r for r in self.robots if r.active]

        # Phase 1: age decay
        for r in active_robots:
            r._age_decay_tick()

        # Phase 2: each robot scans and queues messages this tick
        # (_reveal_all already ran in PHASE 1 movement, outbox already populated)

        # Phase 3: determine comms status for each robot
        def comms_ok(r):
            """True if robot can communicate with base/fleet this tick."""
            if not self.radio_shadow[r.pos[0], r.pos[1]]:
                return True   # open air — direct comms
            z = self.cell_to_zone(r.pos[0], r.pos[1])
            return self.relay_ok_extended(z)  # shadow but relay-bridged

        # Phase 4: build the global shared map from all comms-capable robots.
        # Robots that can communicate share everything they know instantly
        # (centralised planner aggregates all incoming data each tick).
        # Messages from relay-bridged shadow robots arrive with RELAY_COMMS_DELAY.
        t = self.timestep
        for r in active_robots:
            if not r.outbox: continue
            delay = 0 if not self.radio_shadow[r.pos[0], r.pos[1]] else RELAY_COMMS_DELAY
            for msg in r.outbox:
                msg['deliver_at'] = t + delay
            # Move outbox to the fleet pending queue
            self._pending_msgs.extend(r.outbox)
            r.outbox.clear()

        # Phase 5: deliver matured messages to all comms-capable robots
        still_pending = []
        deliverable = []
        for msg in self._pending_msgs:
            if t >= msg['deliver_at']:
                deliverable.append(msg)
            else:
                still_pending.append(msg)
        self._pending_msgs = still_pending

        if deliverable:
            for r in active_robots:
                if comms_ok(r):
                    r.inbox.extend(deliverable)

        # Phase 6: each robot processes inbox
        for r in active_robots:
            r._process_inbox()

        # ── rebuild union belief ──────────────────────────────────────────────
        self.union_belief = self._union_terrain()
        self.union_T      = self._union_temp()
        self.union_R      = self._union_rad()

        # ── survivor detection ──────────────────────────────────────────────────
        # Detection radius matches the sensor scan disc exactly (reveal_R).
        # Line-of-sight is checked using Bresenham's ray from robot to survivor —
        # walls (T_OBS) and building interiors (T_STAIRS from outside) block detection.
        # A robot inside a building CAN detect survivors inside the same building.
        for r in self.robots:
            if not r.active: continue
            R = r.reveal_R
            rx, ry = r.pos
            r_inside_building = self.world.grid[rx][ry]["t"] == T_STAIRS
            for s in self.survivors:
                if s in self.found: continue
                sx, sy = s
                # Euclidean disc check (matches _reveal_all exactly)
                if (rx-sx)**2 + (ry-sy)**2 > R*R: continue
                # LOS check via Bresenham's line
                if self._has_los(rx, ry, sx, sy, r_inside_building):
                    self.found.add(s)

        # ── role decisions ──
        self._decide_roles()

        # reveal radius by role
        for r in self.robots:
            if r.active:
                r.reveal_R = 3 if r.role == Role.SCOUT else 2

        # ── occupation set (collision reservation) ──
        occupied = {r.pos for r in self.robots if r.active and r.battery>0}

        # ── PHASE 1: move relays first, then update relay_ok ──
        for r in self.robots:
            if not r.active: continue
            if r.role == Role.RELAY:
                self._move_relay(r, occupied)

        # update relay_ok AFTER relays have settled
        # Only mark zones that directly have an outside relay — the flood handles propagation
        self.relay_ok = {(zx,zy): self.zone_has_outside_relay((zx,zy))
                         for zx in range(self.zone_nx) for zy in range(self.zone_ny)}
        self._compute_relay_flood()  # propagate through connected zone cluster

        # invalidate reachable caches only when relay_ok actually changed
        if self._relay_ok_flood != self._relay_ok_prev:
            for r in self.robots:
                r._reachable_cache = None
            self._relay_ok_prev = dict(self._relay_ok_flood)

        # ── PHASE 2: task management + movement for non-relay robots ──
        for r in self.robots:
            if not r.active: continue
            if r.role == Role.RELAY: continue
            if r.battery <= 0:
                if r.active:
                    r.active = False; r.death_reason = "battery depleted"
                    self.dead_robots.append((r.name, r.death_reason))
                continue

            # ── Shadow evacuation: robot stranded inside shadow with no relay ──
            # This happens when relay is demoted while explorers are inside.
            # BFS to nearest non-shadow cell and step one cell toward it.
            if (self.radio_shadow[r.pos[0], r.pos[1]]
                    and not self.relay_ok_extended(self.cell_to_zone(r.pos[0], r.pos[1]))):
                W2, H2 = self.world.w, self.world.h
                shd = self.radio_shadow
                tb2 = r.terrain_belief; mask2 = r.caps_mask & 0xF
                # BFS: find nearest non-shadow reachable cell + shortest path back
                evac_parent = {r.pos: None}
                evac_q2 = deque([r.pos]); evac_target2 = None
                while evac_q2 and evac_target2 is None:
                    x2, y2 = evac_q2.popleft()
                    for dx2, dy2 in NBR4:
                        nx3, ny3 = x2+dx2, y2+dy2
                        if not (0<=nx3<W2 and 0<=ny3<H2): continue
                        if (nx3,ny3) in evac_parent: continue
                        if not _TRAV_LUT[int(tb2[nx3,ny3])][mask2]: continue
                        evac_parent[(nx3,ny3)] = (x2, y2)
                        if not shd[nx3, ny3]:
                            evac_target2 = (nx3, ny3); break
                        evac_q2.append((nx3, ny3))
                if evac_target2:
                    # Trace path back to find first step from current pos
                    evac_path = []
                    cur_e = evac_target2
                    while evac_parent[cur_e] is not None:
                        evac_path.append(cur_e)
                        cur_e = evac_parent[cur_e]
                    evac_path.reverse()  # now: [first_step, ..., evac_target]
                    if evac_path and evac_path[0] not in occupied:
                        occupied.discard(r.pos); occupied.add(evac_path[0])
                        r.pos = evac_path[0]
                        r.path = evac_path[1:]
                        r.goal = evac_target2
                r.goal_commit = 0  # force replan next tick
                continue  # skip normal move logic this tick

            self._manage_task_zone(r)
            self._move_robot(r, occupied)

        if len(self.found) == len(self.survivors):
            return False

        alive = any(r.active and r.battery>0 for r in self.robots)
        return alive

    def _manage_task_zone(self, r):
        """Select and maintain a task zone for robot r."""
        # Use robot's local belief for coverage — not oracle union.
        # This means a robot only thinks it's "done" with a zone when it has
        # personally seen (or received via comms) enough of it.
        lease_active = (r.task_zone is not None and self.timestep < r.zone_lease_until)

        # try to select a zone if none held
        if r.task_zone is None:
            def zone_priority(z):
                stats = self.zone_stats(z)
                sf = stats["shadow_frac"]
                needs_relay = sf > 0.2 and not self._relay_ok_flood.get(z, False)
                # Local unknown fraction — what this robot sees as unexplored
                local_uf = self._local_zone_unknown_frac(r, z)
                return (0 if needs_relay else 1, -local_uf)

            for z in sorted(r.assigned_zones, key=zone_priority):
                if r.blacklist.get(z,-1) > self.timestep: continue
                stats = self.zone_stats(z)
                if not self.zone_feasible(r, stats, zone=z): continue

                shadow_needs_relay = (stats["shadow_frac"] > 0.2
                                      and not self._relay_ok_flood.get(z, False)
                                      and self._local_zone_unknown_frac(r, z) > 1.0 - ZONE_DONE)
                if not shadow_needs_relay:
                    fronts = self.zone_frontiers_for(r, z)
                    if not fronts: continue

                r.task_zone = z
                r.zone_lease_until = self.timestep + LEASE_T
                r.task_no_progress = 0; r.task_last_known = 0
                break

        if r.task_zone is None: return

        # Coverage from robot's local belief (comms-gated, age-decayed)
        local_cov = 1.0 - self._local_zone_unknown_frac(r, r.task_zone)
        # Also check union (fleet-wide) coverage — if fleet says done, trust it
        union_cov = self.zone_coverage(self.union_belief, r.task_zone)
        cov = max(local_cov, union_cov)

        stats = self.zone_stats(r.task_zone)

        if cov >= ZONE_DONE:
            self._release_zone(r, "complete"); return
        if not self.zone_feasible(r, stats, zone=r.task_zone):
            self._release_zone(r, "unsuitable"); return
        if r.role != Role.RELAY and not lease_active and r.task_no_progress >= NO_PROGRESS_K:
            self._blacklist_zone(r, r.task_zone, "no_progress"); return

        # update progress counter using local coverage
        known_now = int(local_cov * 10000)
        if r.task_last_known == 0: r.task_last_known = known_now
        if known_now > r.task_last_known:
            r.task_no_progress = 0; r.task_last_known = known_now
        else:
            r.task_no_progress += 1

        # update frontier signal
        fronts = self.zone_frontiers_for(r, r.task_zone)
        r.zone_frontier_count  = len(fronts)
        r.zone_frontier_signal = min(1.0, len(fronts)/25.0)

    def _move_robot(self, r, occupied):
        """Goal selection and movement for non-relay active robots."""
        # pick new goal if needed
        need_new = (r.goal is None or r.pos == r.goal or
                    (r.goal_commit == 0 and r.stuck_steps > 5))
        if need_new:
            tgt = self._choose_goal(r)
            if tgt is None:
                # Robot has nothing to do — increment idle counter.
                # After IDLE_RESCUE_K ticks with no goal, force a full CBBA
                # rebid so stale blacklists / exhausted bundles get cleared.
                r.stuck_steps += 1
                if r.stuck_steps >= IDLE_RESCUE_K:
                    r.stuck_steps    = 0
                    r.bundle         = []
                    r.assigned_zones = []
                    r.task_zone      = None
                    r.blacklist      = {}   # clear all blacklists — fresh start
                r.goal = None; r.path = []; return
            # Register intent BEFORE path-planning so that subsequent robots
            # calling _choose_goal() this same tick will see this robot's
            # intended goal in the crowd penalty and diverge from it.
            r.goal = tgt
            if not r.set_goal(tgt):
                r.goal = None
                return  # path failed, try next tick

        r.move_step(occupied)

# ─────────────────────────────────────────────────────────────────────────────
# Rendering helpers
# ─────────────────────────────────────────────────────────────────────────────
def build_grid_surface(sim, show_map, show_survivors, show_risk,
                       union_belief, union_T, union_R):
    world = sim.world
    surf = pygame.Surface((world.w*CELL_SIZE, world.h*CELL_SIZE))

    if show_risk:
        nW = (world.w+CHUNK_SIZE-1)//CHUNK_SIZE; nH = (world.h+CHUNK_SIZE-1)//CHUNK_SIZE
        risk_map = np.zeros((nW,nH),dtype=float); known_c = np.zeros((nW,nH),dtype=bool)
        for cx in range(nW):
            for cy in range(nH):
                xs = range(cx*CHUNK_SIZE,min((cx+1)*CHUNK_SIZE,world.w))
                ys = range(cy*CHUNK_SIZE,min((cy+1)*CHUNK_SIZE,world.h))
                vT=[]; vR=[]
                for x in xs:
                    for y in ys:
                        if show_map or union_belief[x,y]!=T_UNKNOWN:
                            vT.append(float(world.grid[x][y]["temp"] if show_map else (union_T[x,y] if not np.isnan(union_T[x,y]) else 0)))
                            vR.append(float(world.grid[x][y]["rad"]  if show_map else (union_R[x,y] if not np.isnan(union_R[x,y]) else 0)))
                if vT or vR:
                    risk_map[cx,cy] = max(vT+vR); known_c[cx,cy] = True
        max_risk = max(float(np.max(risk_map)),1e-9)

    for x in range(world.w):
        for y in range(world.h):
            if show_map:
                tb = world.grid[x][y]["t"]; clr = TERRAIN_COLOUR_CODE[tb]; visible=True
            else:
                tb = int(union_belief[x,y]); clr = TERRAIN_COLOUR_CODE[tb]
                visible = (tb != T_UNKNOWN)

            if show_risk:
                if show_map or visible:
                    cx2,cy2 = x//CHUNK_SIZE, y//CHUNK_SIZE
                    if known_c[cx2,cy2]:
                        n = min(max(risk_map[cx2,cy2]/max_risk,0),1)
                        clr = (int(255*n), int(255*(1-n)), 0)
                    else:
                        clr = (180,180,180)

            if (x,y) in sim.found or (show_survivors and (x,y) in sim.survivors):
                clr = (255,0,0)

            surf.fill(clr, (x*CELL_SIZE, y*CELL_SIZE, CELL_SIZE, CELL_SIZE))

    # faint grid lines
    for px in range(0, world.w*CELL_SIZE, 4*CELL_SIZE):
        pygame.draw.line(surf,(170,170,170),(px,0),(px,world.h*CELL_SIZE),1)
    for py in range(0, world.h*CELL_SIZE, 4*CELL_SIZE):
        pygame.draw.line(surf,(170,170,170),(0,py),(world.w*CELL_SIZE,py),1)

    return surf


def build_shadow_surface(sim):
    surf = pygame.Surface((GRID_W*CELL_SIZE,GRID_H*CELL_SIZE), pygame.SRCALPHA)
    col = (40,40,40,110)
    for x in range(GRID_W):
        for y in range(GRID_H):
            if sim.radio_shadow[x,y]:
                surf.fill(col,(x*CELL_SIZE,y*CELL_SIZE,CELL_SIZE,CELL_SIZE))
    return surf


def draw_zones(screen, sim):
    for zx in range(sim.zone_nx):
        for zy in range(sim.zone_ny):
            z = (zx,zy); t = sim.zone_tasks.get(z)
            owners = [next((r for r in sim.robots if r.name==nm),None)
                      for nm in (t.owners if t else [])]
            owners = [r for r in owners if r is not None]
            rect = pygame.Rect(zx*sim.zone_w_cells*CELL_SIZE, zy*sim.zone_h_cells*CELL_SIZE,
                               sim.zone_w_cells*CELL_SIZE, sim.zone_h_cells*CELL_SIZE)
            if not owners:
                pygame.draw.rect(screen,(120,120,120),rect,1)
            elif len(owners)==1:
                pygame.draw.rect(screen,ROBOT_COLOUR[robot_type(owners[0].name)],rect,2)
            else:
                pygame.draw.rect(screen,ROBOT_COLOUR[robot_type(owners[0].name)],rect,3)
                inner = rect.inflate(-6,-6)
                if inner.width>0: pygame.draw.rect(screen,ROBOT_COLOUR[robot_type(owners[1].name)],inner,3)


def draw_robots(screen, robots, show_plans):
    if show_plans:
        for r in robots:
            clr = tuple(max(0,c//2) for c in ROBOT_COLOUR[robot_type(r.name)])
            for (px,py) in r.path:
                screen.fill(clr,(px*CELL_SIZE,py*CELL_SIZE,CELL_SIZE,CELL_SIZE))
            if r.goal:
                gx,gy = r.goal
                pygame.draw.rect(screen,ROBOT_COLOUR[robot_type(r.name)],
                                 (gx*CELL_SIZE,gy*CELL_SIZE,CELL_SIZE,CELL_SIZE),2)

    for r in robots:
        clr = ROBOT_COLOUR[robot_type(r.name)]
        x,y = r.pos
        screen.fill(clr,(x*CELL_SIZE,y*CELL_SIZE,CELL_SIZE,CELL_SIZE))
        if r.role == Role.RELAY:
            pygame.draw.circle(screen,(255,255,0),(x*CELL_SIZE+CELL_SIZE//2,y*CELL_SIZE+CELL_SIZE//2),8,2)


# ─────────────────────────────────────────────────────────────────────────────
# GUI loop
# ─────────────────────────────────────────────────────────────────────────────
def gui_loop():
    pygame.init()
    screen = pygame.display.set_mode((GRID_W*CELL_SIZE+SIDEBAR_WIDTH, GRID_H*CELL_SIZE))
    pygame.display.set_caption("Heterogeneous Robot Fleet Simulator")
    font   = pygame.font.SysFont(None, 24)
    sim    = FleetSim()
    clock  = pygame.time.Clock()

    running=False; show_map=False; show_surv=False
    show_risk=False; show_plans=False; show_zones=False; show_shadow=False

    shadow_surf = build_shadow_surface(sim)
    grid_surf   = None; last_vk=None; last_union_tick=-1

    BX = GRID_W*CELL_SIZE+10
    buttons = [
        ("start",  pygame.Rect(BX,10, 80,20)),
        ("map",    pygame.Rect(BX,35,120,20)),
        ("surv",   pygame.Rect(BX,60,120,20)),
        ("risk",   pygame.Rect(BX,85,120,20)),
        ("plans",  pygame.Rect(BX,110,120,20)),
        ("zones",  pygame.Rect(BX,135,120,20)),
        ("shadow", pygame.Rect(BX,160,120,20)),
    ]

    while True:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT: pygame.quit(); sys.exit()
            if ev.type == pygame.MOUSEBUTTONDOWN and ev.button==1:
                for name, rect in buttons:
                    if rect.collidepoint(ev.pos):
                        if name=="start":  running = not running
                        elif name=="map":  show_map  = not show_map;  grid_surf=None
                        elif name=="surv": show_surv = not show_surv; grid_surf=None
                        elif name=="risk": show_risk = not show_risk; grid_surf=None
                        elif name=="plans":show_plans= not show_plans
                        elif name=="zones":show_zones= not show_zones
                        elif name=="shadow":show_shadow=not show_shadow

        if running:
            if not sim.step():
                running = False
                print(f"[DONE] t={sim.timestep}  found={len(sim.found)}/{len(sim.survivors)}")

        union = sim.union_belief; uT = sim.union_T; uR = sim.union_R
        vk = (show_map, show_surv, show_risk)
        union_changed = sim.timestep != last_union_tick
        if grid_surf is None or vk != last_vk or (union_changed and not show_map):
            grid_surf = build_grid_surface(sim, show_map, show_surv, show_risk, union, uT, uR)
            last_vk = vk; last_union_tick = sim.timestep

        screen.blit(grid_surf,(0,0))
        if show_shadow: screen.blit(shadow_surf,(0,0))
        if show_zones:  draw_zones(screen, sim)
        draw_robots(screen, sim.robots, show_plans)

        # sidebar
        pygame.draw.rect(screen,(255,255,255),(GRID_W*CELL_SIZE,0,SIDEBAR_WIDTH,GRID_H*CELL_SIZE))
        labels = {"start":"Pause" if running else "Start","map":"Hide Map" if show_map else "Show Map",
                  "surv":"Hide Surv" if show_surv else "Show Surv","risk":"Hide Risk" if show_risk else "Show Risk",
                  "plans":"Hide Plans" if show_plans else "Show Plans","zones":"Hide Zones" if show_zones else "Show Zones",
                  "shadow":"Hide Shadow" if show_shadow else "Show Shadow"}
        for name, rect in buttons:
            pygame.draw.rect(screen,(200,200,200),rect)
            screen.blit(font.render(labels[name],True,(0,0,0)),(rect.x+6,rect.y+4))

        # stats
        y_sb = 200
        disc = int(np.sum(union != T_UNKNOWN))
        pct  = disc/(GRID_W*GRID_H)*100
        for line in [f"Step: {sim.timestep}", f"Coverage: {pct:.1f}%",
                     f"Survivors: {len(sim.found)}/{len(sim.survivors)}"]:
            screen.blit(font.render(line,True,(0,0,0)),(BX,y_sb)); y_sb+=22

        y_sb += 8
        screen.blit(font.render("Battery:",True,(0,0,0)),(BX,y_sb)); y_sb+=20
        groups={"Legged":[],"Drone":[],"Boat":[],"Rover":[]}
        alive_c={"Legged":0,"Drone":0,"Boat":0,"Rover":0}
        for r in sim.robots:
            t = robot_type(r.name); groups[t].append(r.battery)
            if r.active and r.battery>0: alive_c[t]+=1
        for t in ("Legged","Drone","Boat","Rover"):
            v=groups[t]; avg=sum(v)/len(v) if v else 0
            screen.blit(font.render(f"{t}: {avg:.0f} ({alive_c[t]}/{len(v)})",True,(0,0,0)),(BX,y_sb)); y_sb+=18

        y_sb+=8
        screen.blit(font.render("Robots:",True,(0,0,0)),(BX,y_sb)); y_sb+=20
        for nm, clr in ROBOT_COLOUR.items():
            pygame.draw.rect(screen,clr,(BX,y_sb,16,16))
            screen.blit(font.render(nm,True,(0,0,0)),(BX+22,y_sb)); y_sb+=20

        y_sb+=8
        screen.blit(font.render("Survivors:",True,(0,0,0)),(BX,y_sb)); y_sb+=20
        for i,(pos) in enumerate(sim.survivors,1):
            found = pos in sim.found
            screen.blit(font.render(f"{'✔' if found else '✖'} S{i} {pos}",True,
                                    (0,180,0) if found else (180,0,0)),(BX,y_sb)); y_sb+=20
            if y_sb > GRID_H*CELL_SIZE-20: break

        pygame.display.flip()
        clock.tick(FPS)


if __name__ == "__main__":
    gui_loop()