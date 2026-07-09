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

# Numba is not used — pure-Python A* is fast enough for this sim and avoids
# JIT compilation overhead, cache corruption, and dependency issues.
_NUMBA_AVAILABLE = False


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
ZONE_DONE        = 0.98   # open-terrain completion threshold
SHADOW_ZONE_DONE = 0.95   # shadow zone (zone-level) exit threshold
STAIR_CELL_DONE  = 0.95   # 95% stair coverage counts as done — last few LOS-blocked
                          # corner cells can keep a building perpetually incomplete at 1.00

# Hazard field
N_HOTSPOTS_TEMP  = 5;   TEMP_AMP_N = 35; TEMP_AMP_P = 0.45; TEMP_AMP_SCALE = 14.0
N_HOTSPOTS_RAD   = 4;   RAD_AMP_N  = 33; RAD_AMP_P  = 0.40; RAD_AMP_SCALE  = 16.0
SIGMA_MIN = 3.0;  SIGMA_MAX = 10.0
TEMP_LIMIT = 120.0;  RAD_LIMIT = 150.0

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
CBBA_ITERS     = 2
MAX_BUNDLE     = 8
ZONE_CAPACITY  = 2
LEASE_T        = 50
COOLDOWN_T     = 40   # reduced from 80 — shorter blacklist so zones get reassigned faster
NO_PROGRESS_K  = 30
IDLE_RESCUE_K  = 10   # reduced from 15 — faster rescue of idle robots

# Relay
RELAY_MIN_HOLD     = 150       # ticks a relay must stay before being demoted
                               # Must exceed max explorer travel time to building + clearance
                               # Must exceed max explorer travel time to building + clearance
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
        # River always starts on the left edge, near vertical centre
        margin = self.h // 6
        return (0, random.randint(self.h//2 - margin, self.h//2 + margin))

    def _pick_other_edge(self, p0):
        # River always exits on the right edge, near vertical centre
        margin = self.h // 6
        return (self.w-1, random.randint(self.h//2 - margin, self.h//2 + margin))

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
        # ── Doors: two wide (3-cell) entrances on OPPOSITE walls (S + N). ──
        # A single localised heat/radiation hotspot (Gaussian, lethal radius a
        # few cells) can seal one approach; an opposite-wall door gives a
        # survivable alternative so a stair-capable robot can still reach the
        # interior even when one side sits in a furnace.  Deterministic — adds
        # no RNG draw, so existing seeded worlds keep their hazard/survivor
        # layout and simply gain a second door.
        for wall_y in (hy + hh - 1, hy):          # south wall, then north wall
            inset = wall_y - 1 if wall_y == hy + hh - 1 else wall_y + 1
            door_x = hx + hw//2
            for dx in range(-(3//2), 3//2+1):
                x = door_x + dx
                if hx <= x < hx+hw:
                    self.grid[x][wall_y]["t"] = T_STAIRS
                    if hy < inset < hy+hh-1:
                        self.grid[x][inset]["t"] = T_STAIRS

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

        # ── river system: exactly one river splitting the map horizontally ──
        self._carve_river(base_width=random.randint(5,7), n_ctrl=4)
        self._fill_river_islands()

        # ── place bridges: exactly one guaranteed crossing ──
        self._place_bridges(n_bridges=1, bridge_w=5, min_spacing=40)
        self._fix_land_pinches(min_corridor=4)

        # ── buildings: exactly 4 staircase buildings, one per quadrant ──
        quadrants = [
            (W//8,    H//8,    W//2-10, H//2-10),   # top-left
            (W//2+5,  H//8,    W-W//8,  H//2-10),   # top-right
            (W//8,    H//2+5,  W//2-10, H-H//8),    # bottom-left
            (W//2+5,  H//2+5,  W-W//8,  H-H//8),    # bottom-right
        ]
        placed_houses = []
        for qx0, qy0, qx1, qy1 in quadrants:
            hw = random.randint(12, 18); hh = random.randint(12, 18)
            placed = False
            for _ in range(120):
                hx = random.randint(qx0, max(qx0, qx1-hw))
                hy = random.randint(qy0, max(qy0, qy1-hh))
                if self._rect_clear(hx, hy, hw, hh, pad=4):
                    self._stamp_house(hx, hy, hw, hh)
                    placed_houses.append((hx, hy, hw, hh))
                    placed = True; break
            if not placed:
                hx = qx0 + 2; hy = qy0 + 2
                hw2 = min(hw, qx1-hx-2); hh2 = min(hh, qy1-hy-2)
                if hw2 >= 8 and hh2 >= 8 and self._rect_clear(hx, hy, hw2, hh2, pad=2):
                    self._stamp_house(hx, hy, hw2, hh2)
                    placed_houses.append((hx, hy, hw2, hh2))

        # ── final connectivity pass ──
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
    def _build_cost_field(terrain_u8, temp_f32, rad_f32, chunked_risk,
                          temp_limit, rad_limit, caps_mask,
                          shadow_interior, relay_ok_cell,
                          global_cov, unk_pen, info_w, unk_prior,
                          alpha_mult, beta_mult, soft_frac,
                          traffic_u16, traffic_w, union_T, union_R):
        """Vectorised per-cell step-cost + blocked mask.

        The cost to ENTER a cell depends only on that cell's properties, never on
        the path taken to reach it, so the whole field can be computed once with
        numpy and then looked up in the A* inner loop.  This reproduces the exact
        semantics of the original per-node cost in _search_ref.
        """
        W, H = terrain_u8.shape
        has_land   = bool(caps_mask & CAP_LAND)
        has_stairs = bool(caps_mask & CAP_STAIRS)
        has_water  = bool(caps_mask & CAP_WATER)
        has_air    = bool(caps_mask & CAP_AIR)
        inv_T = 1.0 / max(1e-6, temp_limit); inv_R = 1.0 / max(1e-6, rad_limit)
        soft_t = soft_frac * temp_limit;     soft_r = soft_frac * rad_limit
        a_eff = ALPHA * alpha_mult;          b_eff = BETA * beta_mult

        unk = (terrain_u8 == T_UNKNOWN)

        def has_nbr(mask):
            out = np.zeros((W, H), dtype=bool)
            out[1:, :]  |= mask[:-1, :]; out[:-1, :] |= mask[1:, :]
            out[:, 1:]  |= mask[:, :-1]; out[:, :-1] |= mask[:, 1:]
            return out

        # ── blocked mask ──────────────────────────────────────────────────────
        blocked = (terrain_u8 == T_OBS)
        if not (has_stairs or has_air): blocked = blocked | (terrain_u8 == T_STAIRS)
        if not (has_water or has_air):  blocked = blocked | (terrain_u8 == T_WATER)
        if not (has_land or has_water or has_air or has_stairs):
            blocked = blocked | (terrain_u8 == T_BRIDGE)
        if not (has_land or has_air):   blocked = blocked | (terrain_u8 == T_FREE)

        if shadow_interior is not None:
            if relay_ok_cell is not None:
                blocked = blocked | (shadow_interior & ~relay_ok_cell)
            else:
                blocked = blocked | shadow_interior

        is_water = (terrain_u8 == T_WATER); is_bridge = (terrain_u8 == T_BRIDGE)
        if has_land and not has_water and not has_air:
            wa = has_nbr(is_water); fb = has_nbr(is_bridge)
            blocked = blocked | (unk & wa & ~fb)
        if has_water and not has_air:
            wba = has_nbr(is_water | is_bridge)
            blocked = blocked | (unk & ~wba)
        if temp_limit < 9000.0 or rad_limit < 9000.0:
            known = ~unk
            hot = known & (np.nan_to_num(temp_f32) > soft_t)
            hot |= known & (np.nan_to_num(rad_f32) > soft_r)
            blocked = blocked | (unk & has_nbr(hot))

        # known-cell hard hazard block (uses union_* fallback, nan-safe)
        if union_T is not None:
            tcomb = np.where(~np.isnan(union_T), union_T, temp_f32)
        else:
            tcomb = temp_f32
        if union_R is not None:
            rcomb = np.where(~np.isnan(union_R), union_R, rad_f32)
        else:
            rcomb = rad_f32
        known = ~unk
        blocked = blocked | (known & (np.nan_to_num(tcomb) > temp_limit))
        blocked = blocked | (known & (np.nan_to_num(rcomb) > rad_limit))

        # ── step cost ───────────────────────────────────────────────────────────
        nW, nH = chunked_risk.shape[1], chunked_risk.shape[2]
        lamT = np.repeat(np.repeat(chunked_risk[0], CHUNK_SIZE, axis=0), CHUNK_SIZE, axis=1)
        lamR = np.repeat(np.repeat(chunked_risk[1], CHUNK_SIZE, axis=0), CHUNK_SIZE, axis=1)
        lamT = lamT[:W, :H]; lamR = lamR[:W, :H]
        e_c = np.maximum(lamT * inv_T, lamR * inv_R)

        tn = np.nan_to_num(temp_f32); rn = np.nan_to_num(rad_f32)
        e_known = np.maximum(tn * inv_T, rn * inv_R)
        e = np.where(unk, unk_prior * e_c, e_known)

        unk_pen_eff = unk_pen if global_cov < 0.95 else min(unk_pen, 0.3)
        unk_cost = np.where(unk, np.float32(unk_pen_eff), np.float32(0.0))

        sc = (1.0 + unk_cost
              + a_eff * (e_c ** P)
              + b_eff * (e ** P)
              + traffic_w * traffic_u16.astype(np.float32))
        if info_w > 1e-9:
            cnt = np.zeros((W, H), dtype=np.float32)
            cnt[1:, :]  += unk[:-1, :]; cnt[:-1, :] += unk[1:, :]
            cnt[:, 1:]  += unk[:, :-1]; cnt[:, :-1] += unk[:, 1:]
            sc = sc - info_w * cnt
        sc = np.maximum(MIN_STEP_COST, sc).astype(np.float32)
        return blocked, sc

    @staticmethod
    def search(start, goal, caps_mask,
               terrain_u8, temp_f32, rad_f32,
               chunked_risk, temp_limit, rad_limit,
               radio_shadow, relay_ok_fn, cell_to_zone_fn,
               global_cov, unk_pen, info_w, unk_prior,
               alpha_mult, beta_mult, soft_frac,
               traffic_u16, traffic_w,
               union_T=None, union_R=None,
               shadow_border=None):
        W, H = terrain_u8.shape
        if start == goal: return []
        sx, sy = start; gx, gy = goal

        if radio_shadow is not None and np.any(radio_shadow):
            shadow_interior = (radio_shadow & ~shadow_border) if shadow_border is not None else radio_shadow
        else:
            shadow_interior = None

        # Precompute relay-ok per shadow-interior cell (one eval per unique zone)
        relay_ok_cell = None
        if shadow_interior is not None and relay_ok_fn is not None:
            relay_ok_cell = np.zeros((W, H), dtype=bool)
            zcache = {}
            for p in np.argwhere(shadow_interior):
                nx, ny = int(p[0]), int(p[1])
                z = cell_to_zone_fn(nx, ny)
                ok = zcache.get(z)
                if ok is None:
                    ok = bool(relay_ok_fn(z)); zcache[z] = ok
                if ok: relay_ok_cell[nx, ny] = True

        blocked, sc = AStar._build_cost_field(
            terrain_u8, temp_f32, rad_f32, chunked_risk,
            temp_limit, rad_limit, caps_mask,
            shadow_interior, relay_ok_cell,
            global_cov, unk_pen, info_w, unk_prior,
            alpha_mult, beta_mult, soft_frac,
            traffic_u16, traffic_w, union_T, union_R)

        # Start cell must be enterable for reconstruction; never block start/goal
        # purely by cost — but keep blocked semantics: original skips blocked
        # neighbours, start is always expandable.
        cost = sc.tolist()           # nested Python lists — fast scalar reads
        blk  = blocked.tolist()
        INF = 1e30
        gscore = [INF] * (W * H); gscore[sx * H + sy] = 0.0
        par = [-1] * (W * H)
        closed = bytearray(W * H)
        heap = [(abs(sx - gx) + abs(sy - gy), 0.0, sx, sy)]
        push = heapq.heappush; pop = heapq.heappop

        while heap:
            f, g, x, y = pop(heap)
            idx = x * H + y
            if closed[idx]: continue
            closed[idx] = 1
            if x == gx and y == gy:
                path = []; cur = idx
                while cur != sx * H + sy:
                    cx, cy = divmod(cur, H)
                    path.append((cx, cy))
                    cur = par[cur]
                    if cur < 0: return []
                path.reverse(); return path
            # inline 4-neighbour expansion
            if x > 0:
                nx = x - 1
                if not blk[nx][y]:
                    nidx = nx * H + y
                    if not closed[nidx]:
                        ng = g + cost[nx][y]
                        if ng < gscore[nidx]:
                            gscore[nidx] = ng; par[nidx] = idx
                            push(heap, (ng + abs(nx - gx) + abs(y - gy), ng, nx, y))
            if x < W - 1:
                nx = x + 1
                if not blk[nx][y]:
                    nidx = nx * H + y
                    if not closed[nidx]:
                        ng = g + cost[nx][y]
                        if ng < gscore[nidx]:
                            gscore[nidx] = ng; par[nidx] = idx
                            push(heap, (ng + abs(nx - gx) + abs(y - gy), ng, nx, y))
            if y > 0:
                ny = y - 1
                if not blk[x][ny]:
                    nidx = x * H + ny
                    if not closed[nidx]:
                        ng = g + cost[x][ny]
                        if ng < gscore[nidx]:
                            gscore[nidx] = ng; par[nidx] = idx
                            push(heap, (ng + abs(x - gx) + abs(ny - gy), ng, x, ny))
            if y < H - 1:
                ny = y + 1
                if not blk[x][ny]:
                    nidx = x * H + ny
                    if not closed[nidx]:
                        ng = g + cost[x][ny]
                        if ng < gscore[nidx]:
                            gscore[nidx] = ng; par[nidx] = idx
                            push(heap, (ng + abs(x - gx) + abs(ny - gy), ng, x, ny))
        return []

    @staticmethod
    def _search_ref(start, goal, caps_mask,
               terrain_u8, temp_f32, rad_f32,
               chunked_risk, temp_limit, rad_limit,
               radio_shadow, relay_ok_fn, cell_to_zone_fn,
               global_cov, unk_pen, info_w, unk_prior,
               alpha_mult, beta_mult, soft_frac,
               traffic_u16, traffic_w,
               union_T=None, union_R=None,
               shadow_border=None):

        W, H = terrain_u8.shape
        if start == goal: return []
        sx, sy = start; gx, gy = goal

        has_land   = bool(caps_mask & CAP_LAND)
        has_stairs = bool(caps_mask & CAP_STAIRS)
        has_water  = bool(caps_mask & CAP_WATER)
        has_air    = bool(caps_mask & CAP_AIR)
        inv_T   = 1.0 / max(1e-6, temp_limit)
        inv_R   = 1.0 / max(1e-6, rad_limit)
        soft_t  = soft_frac * temp_limit
        soft_r  = soft_frac * rad_limit
        a_eff   = ALPHA * alpha_mult
        b_eff   = BETA  * beta_mult

        # Pure-Python A* — callbacks resolved per-cell, only for visited cells
        if radio_shadow is not None and np.any(radio_shadow):
            shadow_interior = radio_shadow & ~shadow_border if shadow_border is not None else radio_shadow.copy()
        else:
            shadow_interior = None

        INF = 1e30
        gscore = np.full((W, H), INF, dtype=np.float32); gscore[sx, sy] = 0.0
        px = np.full((W, H), -1, dtype=np.int16); py = np.full((W, H), -1, dtype=np.int16)
        closed = np.zeros((W, H), dtype=np.uint8)
        heap = [(abs(sx-gx)+abs(sy-gy), 0.0, sx, sy)]

        while heap:
            f, g, x, y = heapq.heappop(heap)
            if closed[x, y]: continue
            closed[x, y] = 1
            if x==gx and y==gy:
                path=[]; cx,cy=gx,gy
                while (cx,cy)!=(sx,sy):
                    path.append((cx,cy))
                    ncx,ncy=int(px[cx,cy]),int(py[cx,cy])
                    if ncx<0: return []
                    cx,cy=ncx,ncy
                path.reverse(); return path
            for dx,dy in NBR4:
                nx,ny=x+dx,y+dy
                if not (0<=nx<W and 0<=ny<H): continue
                if closed[nx,ny]: continue
                tb=int(terrain_u8[nx,ny])
                if tb==T_OBS: continue
                if tb==T_STAIRS and not has_stairs and not has_air: continue
                if tb==T_WATER  and not has_water  and not has_air: continue
                if tb==T_BRIDGE and not (has_land or has_water or has_air or has_stairs): continue
                if tb==T_FREE   and not has_land   and not has_air: continue
                if shadow_interior is not None and shadow_interior[nx,ny]:
                    if not relay_ok_fn(cell_to_zone_fn(nx,ny)): continue
                if tb==T_UNKNOWN:
                    if has_land and not has_water and not has_air:
                        wa=fb=False
                        for ddx,ddy in NBR4:
                            ax,ay=nx+ddx,ny+ddy
                            if 0<=ax<W and 0<=ay<H:
                                nt=int(terrain_u8[ax,ay])
                                if nt==T_WATER: wa=True
                                if nt==T_BRIDGE: fb=True
                        if wa and not fb: continue
                    if has_water and not has_air:
                        wba=False
                        for ddx,ddy in NBR4:
                            ax,ay=nx+ddx,ny+ddy
                            if 0<=ax<W and 0<=ay<H and int(terrain_u8[ax,ay]) in (T_WATER,T_BRIDGE):
                                wba=True; break
                        if not wba: continue
                    if temp_limit<9000.0 or rad_limit<9000.0:
                        danger=False
                        for ddx,ddy in NBR4:
                            ax,ay=nx+ddx,ny+ddy
                            if 0<=ax<W and 0<=ay<H and terrain_u8[ax,ay]!=T_UNKNOWN:
                                if temp_f32[ax,ay]>soft_t or rad_f32[ax,ay]>soft_r:
                                    danger=True; break
                        if danger: continue
                else:
                    t_=(float(union_T[nx,ny]) if (union_T is not None and not np.isnan(union_T[nx,ny]))
                        else float(temp_f32[nx,ny]))
                    r_=(float(union_R[nx,ny]) if (union_R is not None and not np.isnan(union_R[nx,ny]))
                        else float(rad_f32[nx,ny]))
                    if t_>temp_limit or r_>rad_limit: continue
                cx_c,cy_c=nx//CHUNK_SIZE,ny//CHUNK_SIZE
                lam_T=float(chunked_risk[0,cx_c,cy_c]); lam_R=float(chunked_risk[1,cx_c,cy_c])
                e_c=max(lam_T*inv_T,lam_R*inv_R)
                if tb!=T_UNKNOWN:
                    t_=0.0 if np.isnan(temp_f32[nx,ny]) else float(temp_f32[nx,ny])
                    r_=0.0 if np.isnan(rad_f32[nx,ny])  else float(rad_f32[nx,ny])
                    e=max(t_*inv_T,r_*inv_R); unk=0.0
                else:
                    e=unk_prior*e_c
                    unk=unk_pen if global_cov<0.95 else min(unk_pen,0.3)
                ig=0.0
                if info_w>1e-9:
                    cnt=sum(1 for ddx,ddy in NBR4
                            if 0<=nx+ddx<W and 0<=ny+ddy<H and terrain_u8[nx+ddx,ny+ddy]==T_UNKNOWN)
                    ig=info_w*cnt
                sc=1.0+unk+a_eff*(e_c**P)+b_eff*(e**P)-ig+traffic_w*float(traffic_u16[nx,ny])
                sc=max(MIN_STEP_COST,sc)
                ng=g+sc
                if ng<gscore[nx,ny]:
                    gscore[nx,ny]=ng; px[nx,ny]=x; py[nx,ny]=y
                    heapq.heappush(heap,(ng+abs(nx-gx)+abs(ny-gy),ng,nx,ny))
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
            confidence=None, union_T=None, union_R=None):
        key = (zone, caps_mask)
        if key in self._cache:
            return self._cache[key]
        result = self._compute(zone, caps_mask, union, reachable_fn, world,
                               zone_cells_fn, confidence, union_T, union_R)
        self._cache[key] = result
        return result

    def _compute(self, zone, caps_mask, union, reachable_fn, world, zone_cells_fn,
                 confidence=None, union_T=None, union_R=None):
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
                # A cell is "uncertain" if terrain is unknown, OR if it has
                # hazard readings that have decayed below confidence threshold.
                # Terrain type is permanent — do NOT re-visit just because terrain
                # knowledge aged.  Only re-visit for stale hazard (temp/rad) readings.
                has_stale_hazard = (confidence is not None and
                                    confidence[x, y] < CONF_UNCERTAIN and
                                    union[x, y] != T_UNKNOWN and
                                    union[x, y] not in (T_OBS,) and
                                    # Only revisit for hazard if the zone actually has hazard
                                    not ((union_T is None or np.isnan(union_T[x, y])) and
                                         (union_R is None or np.isnan(union_R[x, y]))))
                is_uncertain = union[x, y] == T_UNKNOWN or has_stale_hazard
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
        # Preallocated full-grid buffers for _recompute_chunked — avoids per-call allocation
        self._chunked_buf   = (np.zeros((GRID_W, GRID_H), dtype=np.float32),
                               np.zeros((GRID_W, GRID_H), dtype=np.float32))

        # Scan confidence — age-based decay.
        # scan_age[x,y] = ticks since this robot last directly scanned cell (x,y).
        # Cells never scanned have age=INF (represented as int16 max = 32767).
        # confidence = exp(-age / CONF_TAU), used by local planner and CBBA.
        self.scan_age       = np.full((GRID_W, GRID_H), 32767, dtype=np.int16)
        self.confidence     = np.zeros((GRID_W, GRID_H), dtype=np.float32)

        # Comms — per-cell dict messages (fast for typical 50-cell scan batches).
        # Each message: {x, y, terrain, temp, rad, ts}
        self.outbox:  list  = []
        self.inbox:   list  = []   # filled by FleetSim.step()
        self._inbox_dirty = False  # set when inbox has new data
        self._scan_dirty  = False  # set when _reveal_all discovers new cells
        # personally_scanned: cells this robot has directly sensed (not received)
        self.personally_scanned = np.zeros((GRID_W, GRID_H), dtype=bool)

        # state
        self.active       = True
        self.battery      = MAX_BATTERY
        self.death_reason = None
        self.hazard_killed = False   # True if destroyed by temp/rad — cannot be revived
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
        self.relay_failed_path_count = 0   # consecutive ticks _move_relay couldn't find a path
        self._relay_anchor_blacklist = set()  # border cells that A* couldn't reach
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
        self.terrain_R              = 3

        if self._reveal_all(): self._recompute_chunked()

    def _reveal_all(self):
        """
        Sensor scan: reveal terrain within terrain_R using ray casting.
        Disc offsets are precomputed once per (R, inside_building) combination.
        Only queues new or stale cells into the outbox — skips stable known cells.
        """
        x0, y0 = self.pos
        R   = self.terrain_R
        now = self.sim.timestep
        W, H = self.world.w, self.world.h
        robot_inside_building = (self.world.grid[x0][y0]["t"] == T_STAIRS)

        # Build and cache LOS-passed offsets for this position.
        # Cache key: (x0, y0, inside_building) — changes only on movement.
        # This avoids 169 _has_los calls per stationary robot per tick.
        los_cache_key = (x0, y0, robot_inside_building)
        _lc = getattr(self, '_los_cache', None)
        if _lc is None or getattr(self, '_los_cache_key', None) != los_cache_key:
            offsets = []
            for dx in range(-R, R + 1):
                for dy in range(-R, R + 1):
                    if dx*dx + dy*dy > R*R: continue
                    nx, ny = x0 + dx, y0 + dy
                    if not (0 <= nx < W and 0 <= ny < H): continue
                    if (dx != 0 or dy != 0) and not self.sim._has_los(x0, y0, nx, ny, robot_inside_building):
                        continue
                    offsets.append((nx, ny))
            self._los_cache     = offsets
            self._los_cache_key = los_cache_key
        visible = self._los_cache

        newly_scanned = []
        new_data = False
        for nx, ny in visible:
            self.personally_scanned[nx, ny] = True
            was_fresh = self.scan_age[nx, ny] == 0
            new_cell  = not self.known_mask[nx, ny]
            self.scan_age[nx, ny] = 0
            self.confidence[nx, ny] = 1.0
            if new_cell:
                self.known_mask[nx, ny] = True
                self.terrain_belief[nx, ny] = self.world.grid[nx][ny]["t"]
                self.temp_belief[nx, ny]    = self.world.grid[nx][ny]["temp"]
                self.rad_belief[nx, ny]     = self.world.grid[nx][ny]["rad"]
                new_data = True
            if new_cell or not was_fresh:
                newly_scanned.append((nx, ny))

        for (nx, ny) in newly_scanned:
            # Tuple format: (x, y, terrain, temp, rad, ts) — faster than dict
            self.outbox.append((nx, ny,
                int(self.terrain_belief[nx, ny]),
                float(self.temp_belief[nx, ny]),
                float(self.rad_belief[nx, ny]),
                now))
        return new_data

    def _age_decay_tick(self):
        """
        Increment scan_age and recompute confidence for known cells.
        Throttled to every 5 ticks — CONF_TAU=200 means a 5-tick delay causes
        <2.5% confidence error, well within the CONF_UNCERTAIN=0.25 threshold.
        """
        # Only run every 5 ticks to reduce numpy overhead
        if self.sim.timestep % 5 != 0:
            return
        known = self.known_mask
        age = self.scan_age
        # Add 5 (not 1) since we run every 5 ticks; clip at int16 max
        np.add(age, 5, out=age, where=known, casting='unsafe')
        np.clip(age, 0, 32767, out=age)
        # Recompute confidence for known cells only
        self.confidence[known] = np.exp(-age[known].astype(np.float32) / CONF_TAU)

    def _merge_fleet_update(self, fleet_update, now):
        """Merge shared fleet update arrays directly — avoids per-robot inbox copy.
        fleet_update = (xs, ys, terr, temp, rad, ts) numpy arrays."""
        xs, ys, terr, temp, rad, ts = fleet_update
        ages     = np.clip(now - ts, 0, 32767).astype(np.int16)
        cur_ages = self.scan_age[xs, ys]
        fresh    = ages < cur_ages
        if not np.any(fresh):
            return
        fx, fy, fa = xs[fresh], ys[fresh], ages[fresh]
        self.terrain_belief[fx, fy] = terr[fresh]
        self.known_mask[fx, fy]     = True
        ft = temp[fresh]; fr = rad[fresh]
        valid_t = ~np.isnan(ft); valid_r = ~np.isnan(fr)
        self.temp_belief[fx[valid_t], fy[valid_t]] = ft[valid_t]
        self.rad_belief [fx[valid_r], fy[valid_r]] = fr[valid_r]
        self.scan_age[fx, fy] = fa
        self.confidence[fx, fy] = np.exp(-fa.astype(np.float32) / CONF_TAU)
        self._inbox_dirty = True

    def _process_inbox(self):
        """Legacy inbox processing — only used for delayed relay-chain messages."""
        if not self.inbox: return
        now = self.sim.timestep
        msgs = self.inbox
        n = len(msgs)
        xs   = np.empty(n, dtype=np.int16); ys = np.empty(n, dtype=np.int16)
        terr = np.empty(n, dtype=np.uint8);  ts = np.empty(n, dtype=np.int32)
        temp = np.empty(n, dtype=np.float32); rad = np.empty(n, dtype=np.float32)
        for i, m in enumerate(msgs):
            xs[i]=m[0]; ys[i]=m[1]; terr[i]=m[2]; temp[i]=m[3]; rad[i]=m[4]; ts[i]=m[5]

        ages     = np.clip(now - ts, 0, 32767).astype(np.int16)
        cur_ages = self.scan_age[xs, ys]
        fresh    = ages < cur_ages   # only update if message is fresher

        if not np.any(fresh):
            self.inbox.clear(); return

        fx, fy, fa = xs[fresh], ys[fresh], ages[fresh]
        self.terrain_belief[fx, fy] = terr[fresh]
        self.known_mask[fx, fy]     = True
        # NaN-safe temp/rad update
        ft = temp[fresh]; fr = rad[fresh]
        valid_t = ~np.isnan(ft); valid_r = ~np.isnan(fr)
        self.temp_belief[fx[valid_t], fy[valid_t]] = ft[valid_t]
        self.rad_belief [fx[valid_r], fy[valid_r]] = fr[valid_r]
        self.scan_age[fx, fy] = fa
        # Vectorised confidence — one np.exp call instead of N math.exp calls
        self.confidence[fx, fy] = np.exp(-fa.astype(np.float32) / CONF_TAU)

        self.inbox.clear()
        self._inbox_dirty = True

    def _recompute_chunked(self):
        nW = GRID_W//CHUNK_SIZE; nH = GRID_H//CHUNK_SIZE
        mT = self._chunked_buf[0]; mR = self._chunked_buf[1]
        mT.fill(0.0); mR.fill(0.0)
        km = self.known_mask
        mT[km] = self.temp_belief[km]
        mR[km] = self.rad_belief[km]
        np.nan_to_num(mT, copy=False); np.nan_to_num(mR, copy=False)
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
            # Unknown cells adjacent to known water are likely water — block them.
            # Exception: unknown cells adjacent to a known bridge are the approach to a
            # crossing and must stay passable, otherwise land rovers get pushed back.
            unknown_mask  = (tb_arr == T_UNKNOWN)
            water_known   = (tb_arr == T_WATER)
            bridge_known  = (tb_arr == T_BRIDGE)
            # Dilate masks by 1 cell
            water_nbr  = _ndi.binary_dilation(water_known,  structure=np.ones((3,3),dtype=bool))
            bridge_nbr = _ndi.binary_dilation(bridge_known, structure=np.ones((3,3),dtype=bool))
            # Block unknown-near-water, but restore unknown-near-bridge
            passable &= ~(unknown_mask & water_nbr & ~bridge_nbr)

            # Inside-building fix: when a stair-capable robot is standing on
            # T_STAIRS, unknown cells adjacent to known stair cells are almost
            # certainly more stair interior — the BFS should pass through them so
            # the far side of the building appears reachable.
            # Without this, the BFS stops at the known/unknown boundary and the
            # robot's candidate list has no far-side cells, causing premature exit.
            # This is a belief model correction, not a candidate override — the
            # robot still has to actually navigate there via A*.
            if bool(mask & (CAP_STAIRS | CAP_AIR)):
                robot_on_stair = (self.world.grid[self.pos[0]][self.pos[1]]["t"] == T_STAIRS)
                if robot_on_stair:
                    stair_known = (tb_arr == T_STAIRS)
                    stair_nbr   = _ndi.binary_dilation(stair_known,
                                                        structure=np.ones((3,3), dtype=bool))
                    # Unknown cells neighbouring known stair are passable for BFS
                    passable |= (unknown_mask & stair_nbr)
        elif is_boat:
            # Boat unknown cells only passable if adjacent to known water or bridge.
            # This mirrors the A* boat halo: unknown cells away from water are
            # treated as probable land and blocked.
            # IMPORTANT: always keep the start cell passable even if it is unknown
            # (e.g. the boat spawned on a cell whose terrain hasn't been revealed yet).
            unknown_mask  = (tb_arr == T_UNKNOWN)
            water_known   = (tb_arr == T_WATER)
            bridge_known  = (tb_arr == T_BRIDGE)
            wb_nbr = _ndi.binary_dilation(
                water_known | bridge_known, structure=np.ones((3,3), dtype=bool))
            # Block unknown not near water/bridge, but never block start cell
            block = unknown_mask & ~wb_nbr
            block[self.pos[0], self.pos[1]] = False   # always keep own cell passable
            passable &= ~block

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
        # Overlay space-time reservations onto traffic map for this robot's plan.
        # Cells reserved at t_offset=1 (next step) are the most critical — mark
        # them as high-traffic so A* routes around robots that have already planned.
        traffic = self.sim.traffic_u16
        res = getattr(self.sim, '_reservations', {})
        reservation_bump = {}
        if res:  # skip dict build when table is empty (most ticks early-game)
            for (rx2, ry2, t_off), _ in res.items():
                if t_off == 1:  # immediate next-step conflicts matter most
                    key = (rx2, ry2)
                    reservation_bump[key] = reservation_bump.get(key, 0) + 150
        if reservation_bump:
            # Only copy when we actually have reservations to apply
            traffic = traffic.copy()
            for (rx2, ry2), bump in reservation_bump.items():
                traffic[rx2, ry2] = min(65535, int(traffic[rx2, ry2]) + bump)
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
            relay_ok_fn=lambda z: self.sim._relay_ok_flood.get(z, False),
            cell_to_zone_fn=self.sim.cell_to_zone,
            global_cov=self.sim.global_cov,
            unk_pen=unk_pen, info_w=info_w,
            unk_prior=UNK_PRIOR,
            alpha_mult=a_mult, beta_mult=b_mult,
            soft_frac=SOFT_FRAC,
            traffic_u16=traffic,
            traffic_w=TRAFFIC_W,
            union_T=self.sim.union_T,
            union_R=self.sim.union_R,
            shadow_border=self.sim._shadow_border_mask_cache,
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
        # Register this path in the space-time reservation table so subsequent
        # robots planning this tick see our intended positions and route around us.
        if hasattr(self.sim, '_reservations'):
            res = self.sim._reservations
            win = self.sim._reservation_window
            for t_off, (px2, py2) in enumerate(path[:win], start=1):
                res[(px2, py2, t_off)] = True
        return True

    def _planner_params(self):
        if self.role == Role.SCOUT:
            # SCOUT = sacrifice role: survivors outweigh robot safety.
            # Near-zero hazard multipliers mean A* paths through dangerous zones
            # that SCAN robots would route around. High info_w maximises
            # information gain per step — greedy frontier pursuit.
            return UNK_PEN_SCOUT, INFO_W_SCOUT, 0.05, 0.05
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
                relay_ok_fn=lambda z: self.sim._relay_ok_flood.get(z, False),
                cell_to_zone_fn=self.sim.cell_to_zone,
                global_cov=self.sim.global_cov,
                unk_pen=unk_pen, info_w=info_w, unk_prior=UNK_PRIOR,
                alpha_mult=a_mult, beta_mult=b_mult, soft_frac=SOFT_FRAC,
                traffic_u16=self.sim.traffic_u16, traffic_w=TRAFFIC_W,
                union_T=self.sim.union_T, union_R=self.sim.union_R,
                shadow_border=self.sim._shadow_border_mask_cache,
            )
            if not new_path:
                self.failed_goals[self.goal] = self.sim.timestep + 80
                self.goal = None; self.path = []; self.goal_commit = 0
                return False
            self.path = new_path

        next_cell = self.path[0]

        # ── collision: robots pass through each other ──
        # Heterogeneous fleet — a Drone flying over a Legged robot at a corner
        # is physically plausible. Removing hard collision blocking eliminates
        # the head-on deadlock at narrow corridors without needing WHCA* replanning.
        # The reservation table in set_goal() still softly discourages co-routing.

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
        # ── reveal & update ──
        if self._reveal_all():
            self._scan_dirty = True   # chunked recomputed once per tick in step()

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
            if c["temp"] > self.temp_limit: reasons.append(f"high temperature ({c['temp']:.0f}>{self.temp_limit:.0f})")
            if c["rad"]  > self.rad_limit:  reasons.append(f"high radiation ({c['rad']:.0f}>{self.rad_limit:.0f})")
            self.active = False
            self.hazard_killed = True
            self.death_reason = " & ".join(reasons)
            print(f"[HAZARD DEATH] t={self.sim.timestep}  {self.name} @ {self.pos}  — {self.death_reason}")

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

    # merge_union removed — comms handled via hop-by-hop inbox/outbox

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
        self._zone_uf_cache        = {}    # per-tick zone unknown-frac: (zx,zy)->float
        self._zone_uf_cache_tick   = -1

        self._frontier_cache = ZoneFrontierCache()
        self._reachable_by_mask: dict = {}   # caps_mask -> set (rebuilt each tick)
        self._pending_msgs:      list = []   # fleet comms queue: msgs awaiting delivery
        self._last_clusters:     list = []   # shadow zone clusters, updated each _decide_roles

        self._build_radio_shadow()
        self._build_robots()
        self._build_survivors()

        # Pre-compute zone ID per cell: zx * zone_ny + zy  (int16)
        self._zone_id_arr = np.zeros((GRID_W, GRID_H), dtype=np.int16)
        for zx in range(self.zone_nx):
            x0 = zx * self.zone_w_cells; x1 = x0 + self.zone_w_cells
            for zy in range(self.zone_ny):
                y0 = zy * self.zone_h_cells; y1 = y0 + self.zone_h_cells
                self._zone_id_arr[x0:x1, y0:y1] = zx * self.zone_ny + zy

        # Pre-compute per-zone shadow fraction (shadow is static — never changes)
        self._zone_shadow_frac = {}
        self._zone_shadow_count = {}  # precomputed for zone_stats
        for zx in range(self.zone_nx):
            x0 = zx * self.zone_w_cells; x1 = min(x0 + self.zone_w_cells, GRID_W)
            for zy in range(self.zone_ny):
                y0 = zy * self.zone_h_cells; y1 = min(y0 + self.zone_h_cells, GRID_H)
                slc = self.radio_shadow[x0:x1, y0:y1]
                self._zone_shadow_frac [(zx, zy)] = float(np.mean(slc))
                self._zone_shadow_count[(zx, zy)] = int(np.sum(slc))

        # Precompute cluster IDs — must be after _zone_shadow_frac is built
        self._build_shadow_cluster_ids()

                # Precompute world water and stair masks (static — grid never changes)
        self._world_water_arr = np.zeros((GRID_W, GRID_H), dtype=bool)
        self._world_stair_arr = np.zeros((GRID_W, GRID_H), dtype=bool)
        for x in range(GRID_W):
            for y in range(GRID_H):
                self._world_water_arr[x, y] = self.world.grid[x][y]["t"] in (T_WATER, T_BRIDGE)
                self._world_stair_arr[x, y] = self.world.grid[x][y]["t"] == T_STAIRS

        # Precompute per-zone stair fraction from world truth (never zero due to unknown cells)
        self._zone_stair_frac = {}
        for zx in range(self.zone_nx):
            x0 = zx*self.zone_w_cells; x1 = min(x0+self.zone_w_cells, GRID_W)
            for zy in range(self.zone_ny):
                y0 = zy*self.zone_h_cells; y1 = min(y0+self.zone_h_cells, GRID_H)
                slc = self._world_stair_arr[x0:x1, y0:y1]
                total = slc.size
                self._zone_stair_frac[(zx,zy)] = float(np.sum(slc)) / total if total > 0 else 0.0

        # Populate disc_cluster_border_has_water — needs _world_water_arr and
        # _shadow_border_cells_arr both built above.
        # For each disc cluster, check whether any border cell (non-shadow cell
        # adjacent to the disc) is a water/bridge cell. A boat relay can only
        # serve a disc cluster if it can physically reach the border via water.
        disc_cluster_ids = set(
            self._shadow_cluster_id[z]
            for z, t in self._shadow_zone_type.items()
            if t == 'disc' and z in self._shadow_cluster_id
        )
        for cid in disc_cluster_ids:
            self._disc_cluster_border_has_water[cid] = False
        for pt in self._shadow_border_cells_arr:
            bx, by = int(pt[0]), int(pt[1])
            if not self._world_water_arr[bx, by]: continue
            # This border cell is water — check which disc cluster it touches
            for nx2, ny2 in self.world.neighbours((bx, by)):
                if not self.radio_shadow[nx2, ny2]: continue
                nz2 = self.cell_to_zone(nx2, ny2)
                if self._shadow_zone_type.get(nz2) == 'disc':
                    cid2 = self._shadow_cluster_id.get(nz2)
                    if cid2 is not None:
                        self._disc_cluster_border_has_water[cid2] = True

        # initialise relay_ok — False for shadow zones (relay needed), True for open zones
        self.relay_ok = {(zx,zy): False
                         for zx in range(self.zone_nx) for zy in range(self.zone_ny)}
        self._relay_ok_flood = dict(self.relay_ok)
        self._relay_ok_prev  = dict(self.relay_ok)  # detect changes to avoid redundant cache clears

        self.union_belief = self._union_terrain()
        self.union_T      = self._union_temp()
        self.union_R      = self._union_rad()

        # Space-time reservation table for cooperative pathfinding (WHCA*-lite).
        # Maps (x, y, t_offset) -> robot_name for the next RESERVATION_WINDOW ticks.
        # Robots planning this tick see earlier robots' reservations as blocked cells
        # at each time step, eliminating head-ons and corridor deadlocks.
        self._reservations: dict = {}   # (x, y, t_offset) -> True
        self._reservation_window = 8    # look-ahead depth

        self._decide_roles()          # build _last_clusters before first CBBA
        self._assign_zones_cbba()     # now relay utility is correct from t=0

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
                # Prefer non-shadow water in same quadrant, then any non-shadow water,
                # only fall back to shadow water if the entire map has no alternatives
                non_shadow_quad   = [c for c in pool       if not self.radio_shadow[c[0],c[1]]]
                non_shadow_global = [c for c in water_cells if not self.radio_shadow[c[0],c[1]]]
                chosen_pool = non_shadow_quad or non_shadow_global or pool
                sx, sy = random.choice(chosen_pool)
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
        # Survivors can be anywhere habitable — open ground OR inside buildings (T_STAIRS)
        free = [(x,y) for x in range(GRID_W) for y in range(GRID_H)
                if self.world.grid[x][y]["t"] in (T_FREE, T_STAIRS)]
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
        # ── Stair shadow: building interiors + 1-cell dilation ──────────────────
        stair_mask = np.zeros((GRID_W,GRID_H),dtype=bool)
        for x in range(GRID_W):
            for y in range(GRID_H):
                if self.world.grid[x][y]["t"] == T_STAIRS:
                    stair_mask[x,y] = True
        stair_i = stair_mask.astype(np.uint8)
        stair_dilated = (np.roll(stair_i,1,0)|np.roll(stair_i,-1,0)|
                         np.roll(stair_i,1,1)|np.roll(stair_i,-1,1)).astype(bool)
        rs |= stair_mask | stair_dilated

        # ── Disc shadow: exclude only the stair shadow (building + 1-cell dilation) ─
        # `rs` at this point already contains stair_mask | stair_dilated.
        # We only need to exclude the building footprint — not scattered debris OBS.
        # Debris inside the disc is perfectly fine to be in shadow; excluding it
        # was causing the visual cross/hole artefact (one 3×3 hole per debris cell).
        building_excl = rs.copy()   # stair + dilation — the only region to exclude

        # One large disc centred on the map — prominent, unavoidable radio shadow.
        # Radius 28-36 gives a circle covering ~20-30% of the map.
        cx = GRID_W // 2 + random.randint(-12, 12)
        cy = GRID_H // 2 + random.randint(-12, 12)
        rad = random.randint(18, 24)  # ~1/3 smaller than previous 28-36
        x0d=max(0,cx-rad); x1d=min(GRID_W,cx+rad+1)
        y0d=max(0,cy-rad); y1d=min(GRID_H,cy+rad+1)
        xs=np.arange(x0d,x1d); ys=np.arange(y0d,y1d)
        xx,yy=np.meshgrid(xs,ys,indexing='ij')
        disc = np.zeros((GRID_W,GRID_H),dtype=bool)
        disc[x0d:x1d,y0d:y1d] = (xx-cx)**2+(yy-cy)**2 <= rad**2
        disc &= ~building_excl   # punch out all building cells
        rs |= disc

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
        # Tag each shadow zone as 'stair' or 'disc'.
        # Zones with actual T_STAIRS cells are 'stair'.
        # Zones whose shadow is entirely dilation artefacts (T_FREE/T_OBS bled out
        # from a neighbouring stair building) inherit 'stair' from that neighbour —
        # they form the door approach corridor and must be in the same relay cluster.
        # All other shadow zones are 'disc' (open-area signal loss).
        self._shadow_zone_type = {}  # zone -> 'stair' | 'disc' | 'none'

        # Precompute the stair+dilation mask so zone classification can test
        # whether a shadow cell belongs to building shadow vs disc shadow.
        # A shadow cell is "building-origin" if it is T_STAIRS itself OR
        # it is in the 1-cell dilation of a T_STAIRS cell.
        stair_origin = stair_mask | stair_dilated  # same as rs before disc added

        # First pass: classify by actual cell terrain.
        # Only count a shadow cell as "disc" if it is NOT building-origin.
        # This prevents disc shadow cells that land in the same zone as building
        # cells from flipping the zone type to 'disc' via majority vote.
        stair_cell_counts = {}
        for zx in range(self.zone_nx):
            for zy in range(self.zone_ny):
                z = (zx, zy)
                xs, ys = self.zone_cells(z)
                stair_count = sum(1 for x in xs for y in ys if stair_origin[x,y])
                disc_count  = sum(1 for x in xs for y in ys
                                  if rs[x,y] and not stair_origin[x,y])
                stair_cell_counts[z] = sum(1 for x in xs for y in ys
                                           if rs[x,y] and self.world.grid[x][y]["t"] == T_STAIRS)
                if stair_count + disc_count == 0:
                    self._shadow_zone_type[z] = 'none'
                elif stair_count >= 1:
                    self._shadow_zone_type[z] = 'stair'
                else:
                    self._shadow_zone_type[z] = 'disc'

        # Second pass: promote 'disc' zones that are pure dilation artefacts of a
        # stair building to 'stair'. A zone qualifies if:
        #   - it has zero actual T_STAIRS cells (pure dilation from neighbour wall)
        #   - very few shadow cells (< 8% of zone area — only wall-lip bleed)
        #   - it is 4-connected adjacent to a zone that IS 'stair'
        # This ensures the door approach corridor (shadow bled 1 cell outside the
        # building wall by dilation) is included in the same relay cluster.
        # Zones with many shadow cells are genuine disc shadows and must not be promoted.
        zone_total = self.zone_w_cells * self.zone_h_cells
        dilation_threshold = max(1, int(zone_total * 0.08))  # ≤8% shadow cells
        changed = True
        while changed:
            changed = False
            for zx in range(self.zone_nx):
                for zy in range(self.zone_ny):
                    z = (zx, zy)
                    if self._shadow_zone_type[z] != 'disc': continue
                    if stair_cell_counts[z] > 0: continue  # has real stairs, leave as disc
                    zx2, zy2 = z
                    xs2, ys2 = self.zone_cells(z)
                    shadow_count = sum(1 for x in xs2 for y in ys2 if rs[x, y])
                    if shadow_count > dilation_threshold: continue  # too many — real disc
                    # Check 4-neighbours for a stair zone
                    for dz in self.zone_neighbors4(z):
                        if self._shadow_zone_type.get(dz) == 'stair':
                            self._shadow_zone_type[z] = 'stair'
                            changed = True
                            break

        # Zone adjacency: only connect zones of the SAME shadow type
        self._shadow_zone_adj = set()
        for x in range(GRID_W):
            for y in range(GRID_H):
                if not rs[x, y]: continue
                za = self.cell_to_zone(x, y)
                ta = self._shadow_zone_type.get(za, 'none')
                for nx, ny in self.world.neighbours((x, y)):
                    if rs[nx, ny]:
                        zb = self.cell_to_zone(nx, ny)
                        tb = self._shadow_zone_type.get(zb, 'none')
                        if za != zb and ta == tb and ta != 'none':
                            self._shadow_zone_adj.add(frozenset((za, zb)))



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
        # Cache keyed by zone — union_belief doesn't change within a tick
        zc_cache = getattr(self, '_zc_cache', None)
        if zc_cache is None or getattr(self, '_zc_cache_tick', -1) != self.timestep:
            self._zc_cache = {}; self._zc_cache_tick = self.timestep
            zc_cache = self._zc_cache
        if zone in zc_cache: return zc_cache[zone]
        zx, zy = zone
        x0 = zx*self.zone_w_cells; x1 = min(x0+self.zone_w_cells, GRID_W)
        y0 = zy*self.zone_h_cells; y1 = min(y0+self.zone_h_cells, GRID_H)
        slc = union[x0:x1, y0:y1]
        total = slc.size
        if total == 0: zc_cache[zone] = 1.0; return 1.0
        known = int(np.count_nonzero(slc))
        v = known / total
        zc_cache[zone] = v
        return v

    def zone_neighbors4(self, z):
        zx, zy = z
        return [(zx+dx, zy+dy) for dx,dy in NBR4
                if 0<=zx+dx<self.zone_nx and 0<=zy+dy<self.zone_ny]

    def relay_ok_extended(self, z):
        """
        A zone is comms-ok only if a relay robot is PHYSICALLY present at the
        shadow border covering this zone (_relay_ok_flood).

        The old en-route case (relay merely assigned task_zone in this cluster)
        has been removed — it allowed explorers to enter shadow before the relay
        arrived, which is the exact bug: robots walking into shadow with no relay
        at the border.  Now the contract is strict: one relay physically holding
        the border at all times while explorers are inside.
        """
        if z is None: return False
        return self._relay_ok_flood.get(z, False)

    def _build_shadow_cluster_ids(self):
        """
        Precompute a cluster ID for every shadow zone using the same
        cell-level adjacency graph (_shadow_zone_adj) that _compute_relay_flood
        uses.  Two zones are in the same cluster iff they are connected through
        _shadow_zone_adj edges (same type, physically touching shadow cells).
        Stored in self._shadow_cluster_id: zone -> int.
        Called once after _build_radio_shadow and zone-type assignment.
        """
        self._shadow_cluster_id = {}
        next_id = 0
        shadow_zones = [z for z, t in self._shadow_zone_type.items() if t != 'none']
        visited = set()
        for seed in shadow_zones:
            if seed in visited: continue
            q = deque([seed]); visited.add(seed); cluster_id = next_id; next_id += 1
            while q:
                z = q.popleft()
                self._shadow_cluster_id[z] = cluster_id
                for nz in self.zone_neighbors4(z):
                    if nz not in visited and self._shadow_zone_type.get(nz, 'none') != 'none':
                        if frozenset((z, nz)) in self._shadow_zone_adj:
                            visited.add(nz); q.append(nz)

        # Precompute: for each disc cluster_id, does its shadow border have water?
        # Used to exclude boats from landlocked disc relays. O(N_border) once at init.
        # A boat can only serve a disc relay if it can physically reach the border.
        self._disc_cluster_border_has_water = {}  # cluster_id -> bool

    def _same_shadow_cluster(self, z1, z2):
        """True if z1 and z2 are in the same connected shadow cluster.
        Uses precomputed cluster IDs built from cell-level adjacency — the same
        graph as _compute_relay_flood — so disc zones never bridge stair buildings.
        """
        if z1 == z2: return True
        cid1 = self._shadow_cluster_id.get(z1)
        cid2 = self._shadow_cluster_id.get(z2)
        if cid1 is None or cid2 is None: return False
        return cid1 == cid2

    def _compute_relay_flood(self):
        """
        Flood relay coverage only within each relay's own contiguous shadow cluster.
        Two shadow zones are only considered connected if their shadow cells
        physically touch at cell level (precomputed in _shadow_zone_adj).
        This prevents a relay at one building's border from unlocking a
        completely separate building on the other side of the map.
        """
        seeds = {z for z, ok in self.relay_ok.items() if ok}
        flooded = set(seeds)
        queue = deque(seeds)
        while queue:
            z = queue.popleft()
            for nz in self.zone_neighbors4(z):
                if nz not in flooded and self._shadow_zone_type.get(nz, 'none') != 'none':
                    if frozenset((z, nz)) in self._shadow_zone_adj:
                        flooded.add(nz)
                        queue.append(nz)
        self._relay_ok_flood = {z: (z in flooded) for z in self.relay_ok}

    def zone_has_outside_relay(self, zone):
        """
        True if any relay robot is:
          1. Not in shadow
          2. Directly adjacent to a shadow cell in `zone`
          3. Its task_zone is in the same shadow cluster AND same shadow type as `zone`
        Disc-shadow and stair-shadow are always separate clusters.
        """
        zone_type = self._shadow_zone_type.get(zone, 'none')
        if zone_type == 'none':
            return False

        zx, zy = zone
        x0 = zx*self.zone_w_cells; x1 = min(x0+self.zone_w_cells, self.world.w)
        y0 = zy*self.zone_h_cells; y1 = min(y0+self.zone_h_cells, self.world.h)

        for r in self.robots:
            if not r.active or r.role != Role.RELAY: continue
            if r.task_zone is None: continue
            # Relay's task_zone must be same type and same cluster
            if self._shadow_zone_type.get(r.task_zone, 'none') != zone_type: continue
            if not self._same_shadow_cluster(r.task_zone, zone): continue
            rx, ry = r.pos
            if self.radio_shadow[rx, ry]: continue
            for nx, ny in self.world.neighbours((rx, ry)):
                if (x0 <= nx < x1 and y0 <= ny < y1
                        and self.radio_shadow[nx, ny]):
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
        x0 = zx*self.zone_w_cells; x1 = min(x0+self.zone_w_cells, GRID_W)
        y0 = zy*self.zone_h_cells; y1 = min(y0+self.zone_h_cells, GRID_H)

        union  = self.union_belief[x0:x1, y0:y1]
        uT_slc = self.union_T     [x0:x1, y0:y1]
        uR_slc = self.union_R     [x0:x1, y0:y1]

        total   = union.size
        # Use precomputed uf cache when available (rebuilt each tick after union)
        if self._zone_uf_cache_tick == t and zone in self._zone_uf_cache:
            uf_val = self._zone_uf_cache[zone]
            unknown = int(round(uf_val * total))
        else:
            unknown = int(np.sum(union == T_UNKNOWN))
            uf_val  = unknown / total if total > 0 else 0.0
        known = total - unknown

        known_mask = (union != T_UNKNOWN)
        n_water  = int(np.sum(union == T_WATER))
        n_stairs = int(np.sum(union == T_STAIRS))
        n_free   = int(np.sum(union == T_FREE))

        if known > 0:
            valid_t = known_mask & ~np.isnan(uT_slc)
            valid_r = known_mask & ~np.isnan(uR_slc)
            avgT = float(np.mean(uT_slc[valid_t])) if np.any(valid_t) else 0.0
            avgR = float(np.mean(uR_slc[valid_r])) if np.any(valid_r) else 0.0
            fw = n_water  / known
            fs = n_stairs / known
            ff = n_free   / known
        else:
            avgT = avgR = fw = fs = ff = 0.0

        fs_world = self._zone_stair_frac.get(zone, fs)
        if fs_world > fs:
            fs = fs_world

        shadow_count = self._zone_shadow_count.get(zone, 0)
        sf = shadow_count / total if total > 0 else 0.0

        cx = zx*self.zone_w_cells + self.zone_w_cells//2
        cy = zy*self.zone_h_cells + self.zone_h_cells//2
        stats = dict(unknown_frac=uf_val,
                     avgT=avgT, avgR=avgR, f_water=fw, f_stairs=fs, f_free=ff,
                     shadow_frac=sf, center=(cx,cy), known=known, total=total)
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
            # Boat needs water in this zone to be useful.
            # Before the boat has explored the zone, its reachable BFS only sees
            # cells near its spawn — the zone may be far away with 0 reachable cells
            # even though it is full of water.  Fall back to a world-water check
            # (does the zone contain any water at all?) when local knowledge is thin.
            if zone is None: return fw > 0.05
            zx, zy = zone
            x0 = zx*self.zone_w_cells; x1 = min(x0+self.zone_w_cells, GRID_W)
            y0 = zy*self.zone_h_cells; y1 = min(y0+self.zone_h_cells, GRID_H)
            water_slice = self._world_water_arr[x0:x1, y0:y1]
            zone_has_water = bool(np.any(water_slice))
            if not zone_has_water:
                return False   # zone has no water at all — definitely skip
            # If the boat has personally scanned enough of the zone, use the
            # reachability check to confirm connectivity.  Otherwise trust that
            # if the zone has water, the boat can eventually reach it.
            personal_frac = float(np.mean(r.personally_scanned[x0:x1, y0:y1]))
            r.reachable()
            if r._reachable_arr is None:
                return personal_frac < ZONE_PERSONAL_THRESH  # no data yet, assume reachable only if unexplored
            reach_slice = r._reachable_arr[x0:x1, y0:y1]
            return bool(np.any(reach_slice & water_slice))

        if r.name.startswith("Legged"):
            return not (fw > 0.30 and fs < 0.05 and uf < 0.60)
        # Rovers are LAND-only (no STAIRS, no AIR). If the zone is stair-dominant,
        # a Rover can't enter the building interior so it has no useful work there.
        # Let Legged/Drone robots take these slots exclusively.
        if r.name.startswith("Rover"):
            if fs > 0.15 and not bool(r.caps_mask & (CAP_STAIRS | CAP_AIR)):
                return False
        return True

    # ── zone frontiers (shared cache) ─────────────────────────────────────────
    def zone_frontiers_for(self, robot, zone) -> list:
        """Return frontiers in zone for robot, using robot's local belief + confidence."""
        def reachable_fn(mask):
            if mask not in self._reachable_by_mask:
                robot.reachable()
                self._reachable_by_mask[mask] = robot._reachable_arr
            return self._reachable_by_mask[mask]

        return self._frontier_cache.get(
            zone, robot.caps_mask,
            robot.terrain_belief,
            reachable_fn,
            self.world,
            self.zone_cells,
            confidence=robot.confidence,
            union_T=self.union_T,
            union_R=self.union_R,
        )

    def zone_has_frontiers(self, robot, zone) -> bool:
        """Check if zone has any frontiers for this specific robot.
        Cache is per (zone, robot_id) — two robots with identical caps_mask
        have different local beliefs so must not share a cache entry."""
        key = (zone, id(robot))
        hf_cache = getattr(self, '_has_frontiers_cache', None)
        if hf_cache is None or getattr(self, '_hf_cache_tick', -1) != self.timestep:
            self._has_frontiers_cache = {}
            self._hf_cache_tick = self.timestep
            hf_cache = self._has_frontiers_cache
        if key not in hf_cache:
            hf_cache[key] = bool(self.zone_frontiers_for(robot, zone))
        return hf_cache[key]

    # ── CBBA ──────────────────────────────────────────────────────────────────
    def _zone_capacity(self, zone):
        """Always ZONE_CAPACITY. Relays clear their bundle on election so they
        no longer consume an owner slot — the old capacity-3 override for stair
        zones was causing 3 explorers to compete on one building, producing
        flip-flopping goals and wasted travel."""
        return ZONE_CAPACITY

    def _assign_zones_cbba(self):
        """
        Two-Level Clustered CBBA (TLC-CBBA).

        Guard: only runs once per simulation tick regardless of how many call
        sites trigger it in the same step() call. Without this, relay border
        arrivals, relay_ok change events, and the cadence timer can all fire
        CBBA independently on the same tick — multiplying cost by 3-5x.
        The first call in a tick does the full recompute; subsequent calls
        in the same tick are no-ops (state hasn't changed anyway).
        Exception: t=0 initialisation always runs (no previous tick exists).
        """
        if (self.timestep > 0
                and getattr(self, '_cbba_last_tick', -1) == self.timestep):
            return   # already ran this tick
        self._cbba_last_tick = self.timestep

        # ── Shared setup (identical to original) ─────────────────────────────
        self._dead_in_zone_cache = {}
        for r in self.robots:
            if not r.active:
                z = self.cell_to_zone(r.pos[0], r.pos[1])
                if z: self._dead_in_zone_cache[z] = self._dead_in_zone_cache.get(z, 0) + 1

        # Zone age tracking: ticks since a robot last actively scanned this zone.
        # Initialised on first CBBA call, then updated in step().
        if not hasattr(self, '_zone_last_visited'):
            self._zone_last_visited = {}  # zone -> last timestep when any robot was inside

        # Clear relay bundles; reset idle/done explorers
        for r in self.robots:
            if not r.active or r.battery <= 0: continue
            if r.role == Role.RELAY:
                for z in r.bundle:
                    task = self.zone_tasks.get(z)
                    if task and r.name in task.owners:
                        task.owners.remove(r.name)
                        if not task.owners:
                            task.status = 'free'; task.expires_at = 0
                r.bundle = []; r.assigned_zones = []
                continue
            zone_done = (r.task_zone is None or
                         self.zone_coverage(self.union_belief, r.task_zone) >= ZONE_DONE)
            if zone_done:
                r.bundle = []; r.assigned_zones = []

        zones = [(zx,zy) for zx in range(self.zone_nx) for zy in range(self.zone_ny)]

        # Refresh zone task progress
        for z, task in self.zone_tasks.items():
            task.progress = self.zone_coverage(self.union_belief, z)
            is_shz = self._zone_shadow_frac.get(z, 0.0) > 0.2
            done_thr = SHADOW_ZONE_DONE if is_shz else ZONE_DONE
            # For stair zones: also check stair-cell-only completion (uses same cache as _zone_utility)
            if (is_shz and self._shadow_zone_type.get(z) == 'stair'
                    and task.progress < done_thr):
                _sd = getattr(self, '_stair_done_cache', {})
                done = _sd.get(z)
                if done is None:
                    zx_c,zy_c=z
                    x0_c=zx_c*self.zone_w_cells; x1_c=min(x0_c+self.zone_w_cells,GRID_W)
                    y0_c=zy_c*self.zone_h_cells; y1_c=min(y0_c+self.zone_h_cells,GRID_H)
                    stair_m=self._world_stair_arr[x0_c:x1_c,y0_c:y1_c]
                    n_s=int(np.sum(stair_m))
                    done = (n_s>0 and (1.0-int(np.sum((self.union_belief[x0_c:x1_c,y0_c:y1_c]==T_UNKNOWN)&stair_m))/n_s)>=STAIR_CELL_DONE)
                if done:
                    task.progress = done_thr
            if task.progress >= done_thr and task.status != "blacklisted":
                task.owners = []; task.status = "released"; task.expires_at = 0

        # ── Layer 0: spatial clustering of explorer robots ────────────────────
        # Cluster robots by position using a simple grid partition.
        # Number of clusters scales with sqrt(R) so each cluster has ~sqrt(R) robots.
        active_explorers = [r for r in self.robots
                            if r.active and r.battery > 0 and r.role != Role.RELAY]

        if not active_explorers:
            return

        R = len(active_explorers)
        # Target ~3-4 robots per cluster; minimum 1 cluster, max 8
        k = max(1, min(8, round(R ** 0.5)))

        # Partition map into k×k grid cells; assign each robot to nearest cell centre
        # Use k=ceil(sqrt(k_target)) grid on each axis
        import math
        k_axis = max(1, round(math.sqrt(k)))
        cell_w = math.ceil(GRID_W / k_axis)
        cell_h = math.ceil(GRID_H / k_axis)

        clusters: dict[int, list] = {}   # cluster_id -> [robots]
        for r in active_explorers:
            cx = min(k_axis - 1, r.pos[0] // cell_w)
            cy = min(k_axis - 1, r.pos[1] // cell_h)
            cid = cx * k_axis + cy
            clusters.setdefault(cid, []).append(r)

        # ── Layer 1: intra-cluster CBBA ───────────────────────────────────────
        # Each cluster bids on zones within and near its spatial region.
        # Zones are considered "local" if their centre is within 1.5× the cluster
        # cell size — this allows overlap so boundary zones get competed for.
        zone_centres = {
            z: (z[0]*self.zone_w_cells + self.zone_w_cells//2,
                z[1]*self.zone_h_cells + self.zone_h_cells//2)
            for z in zones
        }

        for cid, cluster_robots in clusters.items():
            cx_idx = cid // k_axis
            cy_idx = cid %  k_axis
            # Cluster spatial bounds — overlap margin scales down with fleet size
            # so large fleets don't have every zone appearing in dozens of clusters.
            margin = max(0.5, 1.5 / max(1.0, math.sqrt(R / 12.0)))
            x_lo = (cx_idx - margin) * cell_w
            x_hi = (cx_idx + 1 + margin) * cell_w
            y_lo = (cy_idx - margin) * cell_h
            y_hi = (cy_idx + 1 + margin) * cell_h

            # Local zone subset: zones whose centre falls within expanded cluster bounds
            local_zones = [
                z for z in zones
                if x_lo <= zone_centres[z][0] < x_hi
                and y_lo <= zone_centres[z][1] < y_hi
            ]

            if not local_zones: continue

            bundle_counts = {r.name: len(r.bundle) for r in cluster_robots}

            for _it in range(CBBA_ITERS):
                # 1a) intra-cluster bundle building
                for r in cluster_robots:
                    bundle_counts[r.name] = len(r.bundle)
                    _other_capable = sum(
                        1 for rr in self.robots
                        if rr is not r and rr.active
                        and bool(rr.caps_mask & (CAP_STAIRS | CAP_AIR))
                        and rr.role != Role.RELAY
                    )
                    bids = []
                    for z in local_zones:
                        u = self._zone_utility(r, z, bundle_counts,
                                               other_capable_cache=_other_capable)
                        if u is not None:
                            bids.append((u, z))
                    bids.sort(reverse=True)
                    for u, z in bids:
                        if len(r.bundle) >= MAX_BUNDLE: break
                        if z not in r.bundle:
                            r.bundle.append(z)

                # 1b) intra-cluster consensus
                zone_claims: dict = {}
                for r in cluster_robots:
                    for z in r.bundle:
                        u = self._zone_utility(r, z, bundle_counts)
                        if u is not None:
                            zone_claims.setdefault(z, []).append((r.name, u))

                for z, claims in zone_claims.items():
                    if len(claims) <= 1: continue
                    claims.sort(key=lambda t: t[1], reverse=True)
                    winners = []; used_caps = set()
                    for nm, u in claims:
                        rr = next((x for x in cluster_robots if x.name == nm), None)
                        if rr is None: continue
                        # Enforce capability diversity: two robots with identical
                        # caps_mask explore the same cells — pure redundancy.
                        # Always skip same-caps duplicates regardless of zone type.
                        if rr.caps_mask in used_caps: continue
                        winners.append(nm); used_caps.add(rr.caps_mask)
                        if len(winners) >= self._zone_capacity(z): break
                    winner_set = set(winners)
                    for r in cluster_robots:
                        if r.name not in winner_set and z in r.bundle:
                            r.bundle = [zz for zz in r.bundle if zz != z]

        # ── Layer 2: inter-cluster conflict resolution ────────────────────────
        # A zone in the overlap margin may be claimed by robots from different
        # clusters. Resolve globally: for each contested zone keep the top-k
        # bidders regardless of cluster membership.
        all_bundle_counts = {r.name: len(r.bundle) for r in self.robots}
        zone_claims_global: dict = {}
        for r in active_explorers:
            for z in r.bundle:
                u = self._zone_utility(r, z, all_bundle_counts)
                if u is not None:
                    zone_claims_global.setdefault(z, []).append((r.name, u))

        for z, claims in zone_claims_global.items():
            # Always enforce capability diversity — two robots with identical
            # caps_mask explore identical cells, pure redundancy regardless of
            # whether we are over capacity or not.
            claims.sort(key=lambda t: t[1], reverse=True)
            winners = []; used_caps = set()
            for nm, u in claims:
                rr = next((x for x in self.robots if x.name == nm), None)
                if rr is None: continue
                if rr.caps_mask in used_caps: continue
                winners.append(nm); used_caps.add(rr.caps_mask)
                if len(winners) >= self._zone_capacity(z): break
            # Only prune if the winner set actually changed (avoids churning bundles
            # for zones that were already correctly assigned)
            winner_set = set(winners)
            for r in active_explorers:
                if r.name not in winner_set and z in r.bundle:
                    r.bundle = [zz for zz in r.bundle if zz != z]

        # ── Finalise: bundle -> assigned_zones, update zone_tasks ─────────────
        for z, t in self.zone_tasks.items():
            if t.status != "blacklisted": t.owners = []

        for r in self.robots:
            r.assigned_zones = list(r.bundle)
            for z in r.assigned_zones:
                t = self.zone_tasks[z]
                if t.status == "blacklisted": continue
                if r.name not in t.owners and len(t.owners) < self._zone_capacity(z):
                    t.owners.append(r.name)
                t.status = "held"; t.expires_at = self.timestep + LEASE_T

        # Prune bundles: remove zones where this robot couldn't register as owner
        # (capacity was full). Without this, non-owner robots still task-zone these
        # zones and pile up at the border — the stacking problem at scale.
        for r in self.robots:
            if not r.active: continue
            owned = {z for z in r.assigned_zones
                     if r.name in self.zone_tasks[z].owners}
            if len(owned) < len(r.assigned_zones):
                r.bundle = [z for z in r.bundle if z in owned]
                r.assigned_zones = list(r.bundle)

        # ── Fallback: every active non-relay robot gets at least one zone ─────
        for r in self.robots:
            if not r.active or r.battery <= 0: continue
            if r.role == Role.RELAY: continue
            if not r.bundle:
                z = self._fallback_zone(r)
                if z: r.bundle.append(z); r.assigned_zones.append(z)

        # ── Stair-zone rescue ─────────────────────────────────────────────────
        # Zones at cluster boundaries may be missed — force-assign relay-covered
        # stair zones that still have no owner after both layers.
        for z, zt in self._shadow_zone_type.items():
            if zt != 'stair': continue
            if not self._relay_ok_flood.get(z, False): continue
            if self.zone_stats(z)['unknown_frac'] < 0.20: continue
            if self.zone_tasks[z].owners: continue
            best_r = None; best_d = 1e9
            zx, zy = z
            zcx = zx * self.zone_w_cells + self.zone_w_cells // 2
            zcy = zy * self.zone_h_cells + self.zone_h_cells // 2
            for r in self.robots:
                if not r.active or r.battery <= 0: continue
                if r.role == Role.RELAY: continue
                if len(r.bundle) >= MAX_BUNDLE: continue
                if not self.zone_feasible(r, self.zone_stats(z), zone=z): continue
                d = abs(r.pos[0] - zcx) + abs(r.pos[1] - zcy)
                if d < best_d:
                    best_d = d; best_r = r
            if best_r is not None and z not in best_r.bundle:
                best_r.bundle.append(z)
                best_r.assigned_zones.append(z)
                t2 = self.zone_tasks[z]
                if best_r.name not in t2.owners:
                    t2.owners.append(best_r.name)
                t2.status = "held"
                t2.expires_at = self.timestep + LEASE_T

        # ── Open-terrain zone rescue ──────────────────────────────────────────
        # TLC-CBBA's spatial clustering leaves peripheral corner zones permanently
        # unowned when robots' bundles fill with closer zones first.
        # Force-assign the nearest feasible robot to any orphaned open zone.
        for z in [(zx,zy) for zx in range(self.zone_nx) for zy in range(self.zone_ny)]:
            if self.zone_tasks[z].owners: continue
            zt2 = self._shadow_zone_type.get(z, 'none')
            if zt2 == 'stair': continue           # handled by stair rescue
            if self.zone_stats(z)['unknown_frac'] < 0.40: continue
            best_r = None; best_d = 1e9
            zx, zy = z
            zcx = zx*self.zone_w_cells + self.zone_w_cells//2
            zcy = zy*self.zone_h_cells + self.zone_h_cells//2
            for r in self.robots:
                if not r.active or r.battery <= 0: continue
                if r.role == Role.RELAY: continue
                if len(r.bundle) >= MAX_BUNDLE: continue
                if not self.zone_feasible(r, self.zone_stats(z), zone=z): continue
                d = abs(r.pos[0]-zcx) + abs(r.pos[1]-zcy)
                if d < best_d: best_d = d; best_r = r
            if best_r is not None and z not in best_r.bundle:
                best_r.bundle.append(z); best_r.assigned_zones.append(z)
                t2 = self.zone_tasks[z]
                if best_r.name not in t2.owners: t2.owners.append(best_r.name)
                t2.status = "held"; t2.expires_at = self.timestep + LEASE_T

    def _zone_utility(self, r, zone, bundle_counts, other_capable_cache=None):
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
        # Stair zones: also skip if stair cells are fully explored (cached per zone per tick)
        if self._shadow_zone_type.get(zone) == 'stair':
            _sd = getattr(self, '_stair_done_cache', None)
            if _sd is None or getattr(self, '_stair_done_tick', -1) != self.timestep:
                self._stair_done_cache = {}; self._stair_done_tick = self.timestep
                _sd = self._stair_done_cache
            if zone not in _sd:
                zx_u,zy_u=zone
                x0_u=zx_u*self.zone_w_cells; x1_u=min(x0_u+self.zone_w_cells,GRID_W)
                y0_u=zy_u*self.zone_h_cells; y1_u=min(y0_u+self.zone_h_cells,GRID_H)
                stair_u=self._world_stair_arr[x0_u:x1_u,y0_u:y1_u]
                n_su=int(np.sum(stair_u))
                if n_su>0:
                    unk_su=int(np.sum((self.union_belief[x0_u:x1_u,y0_u:y1_u]==T_UNKNOWN)&stair_u))
                    _sd[zone] = (1.0-unk_su/n_su) >= STAIR_CELL_DONE
                else:
                    _sd[zone] = False
            if _sd[zone]: return None
        if r.blacklist.get(zone, -1) > self.timestep: return None
        t = self.zone_tasks.get(zone)

        # How much has THIS robot personally scanned in this zone?
        zx, zy = zone
        x0 = zx*self.zone_w_cells; x1 = min(x0+self.zone_w_cells, GRID_W)
        y0 = zy*self.zone_h_cells; y1 = min(y0+self.zone_h_cells, GRID_H)
        zone_cells_total = (x1-x0) * (y1-y0)
        personal_scanned = int(np.count_nonzero(r.personally_scanned[x0:x1, y0:y1]))
        personal_frac    = personal_scanned / max(1, zone_cells_total)

        # Cells that are known but confidence-decayed — only count as uncertain
        # if they carry hazard readings (temp/rad) that may have changed.
        # Terrain type is permanent; don't pull robots back just because terrain
        # knowledge aged. This prevents excessive revisiting of cleared zones.
        known_slice = r.known_mask[x0:x1, y0:y1]
        conf_slice  = r.confidence[x0:x1, y0:y1]
        uT_sl = self.union_T[x0:x1, y0:y1]
        uR_sl = self.union_R[x0:x1, y0:y1]
        has_hazard_sl = ~(np.isnan(uT_sl) & np.isnan(uR_sl))
        uncertain_known = int(np.count_nonzero(
            known_slice & (conf_slice < CONF_UNCERTAIN) & has_hazard_sl))
        unknown_slice   = ~known_slice
        local_unknown_frac = (np.count_nonzero(unknown_slice) + uncertain_known) / zone_cells_total

        # Zone centre for distance calculation
        cx = x0 + (x1-x0)//2; cy = y0 + (y1-y0)//2
        dist   = abs(r.pos[0]-cx) + abs(r.pos[1]-cy)
        travel = dist / float(GRID_W + GRID_H)
        load   = 0.25 * bundle_counts.get(r.name, 0)

        # Shadow status — always observable (shadow is a radio property, not a terrain secret)
        shadow_frac = self._zone_shadow_frac.get(zone, 0.0)
        relay_needed = shadow_frac > 0.05 and not self._relay_ok_flood.get(zone, False)
        relay_active = shadow_frac > 0.05 and self._relay_ok_flood.get(zone, False)

        # Boats must not bid on disc shadow zones they can't reach via water.
        # Without this they border-loiter at the disc doing nothing useful.
        is_boat_robot = r.name.startswith("Boat")
        if is_boat_robot and relay_needed:
            zone_type_pre = self._shadow_zone_type.get(zone, 'none')
            if zone_type_pre == 'disc':
                # Only allow if this boat has confirmed water access to the disc border
                cid_pre = self._shadow_cluster_id.get(zone)
                if cid_pre is not None:
                    _brc = getattr(self, '_boat_reach_cache', {})
                    brc_key = (r.name, cid_pre)
                    if not _brc.get(brc_key, True):  # default True = assume reachable if unchecked
                        return None   # confirmed unreachable — skip this zone

        # Don't count relay robots toward zone capacity — they're parked outside
        # serving the zone, not consuming exploration bandwidth inside it.
        # Also don't count stalled owners who have made zero personal progress
        # (personal_frac=0, no_progress >= NO_PROGRESS_K//2) — they've claimed
        # the slot but aren't advancing, so new robots should be allowed in.
        if t and t.status == "held" and self.timestep < t.expires_at:
            relay_owners = [nm for nm in t.owners
                            if any(rr.name == nm and rr.role == Role.RELAY
                                   for rr in self.robots)]
            stalled_owners = [nm for nm in t.owners
                              if nm not in relay_owners
                              and any(rr.name == nm
                                      and float(np.mean(rr.personally_scanned[x0:x1, y0:y1])) < 0.01
                                      and rr.task_no_progress >= NO_PROGRESS_K // 2
                                      for rr in self.robots if rr.active)]
            effective_owners = len(t.owners) - len(relay_owners) - len(stalled_owners)
            if r.name not in t.owners and effective_owners >= self._zone_capacity(zone):
                return None

        if personal_frac >= ZONE_PERSONAL_THRESH:
            # ── KNOWN ZONE: bid on full stats from shared belief ──────────────
            stats = self.zone_stats(zone)
            if not self.zone_feasible(r, stats, zone=zone): return None
            if r.name.startswith("Boat"):
                if stats["f_water"] <= 0.0 and stats["unknown_frac"] < 0.40:
                    return None  # no water in this zone and mostly explored
                # When global water is exhausted, boats should only bid on zones
                # with actual water content — freeing them for relay duty
                global_water_uf = getattr(self, '_type_pref_cache', {}).get('water_uf', 1.0)
                if global_water_uf < 0.15 and stats["f_water"] <= 0.0:
                    return None  # water done globally — don't waste boat on dry terrain

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
            if r.name.startswith(("Legged","Drone")): critical += 2.5*fs
            if r.name.startswith("Boat"):              critical += 1.5*fw

            shadow_bonus       = 0.8 * sf * uf
            dead_bonus = 1.5 * self._dead_in_zone_cache.get(zone, 0)
            zone_type = self._shadow_zone_type.get(zone, 'none')
            can_enter_shadow = bool(r.caps_mask & (CAP_STAIRS | CAP_AIR))
            if relay_active and zone_type == 'stair' and can_enter_shadow:
                relay_explorer_bonus = 12.0 * sf * uf
                u = (1.0*uf + risk_term + terrain_term - 0.02*travel
                     + critical + shadow_bonus + relay_explorer_bonus + dead_bonus - load)
            elif relay_active and zone_type == 'disc':
                relay_explorer_bonus = 6.0 * sf * uf
                u = (1.0*uf + risk_term + terrain_term - 0.05*travel
                     + critical + shadow_bonus + relay_explorer_bonus + dead_bonus - load)
            elif relay_needed and zone_type == 'stair' and can_enter_shadow:
                # Pre-relay stair pull: capable robot heads toward building BEFORE
                # relay exists. This approach triggers relay election — the robot
                # moving close to the border makes a Rover elect itself.
                # Scaled by uf so nearly-explored buildings don't pull indefinitely.
                relay_explorer_bonus = 4.0 * sf * uf
                u = (1.0*uf + risk_term + terrain_term - 0.03*travel
                     + critical + shadow_bonus + relay_explorer_bonus + dead_bonus - load)
            else:
                relay_explorer_bonus = (4.0 * sf * uf) if relay_active else 0.0
                u = (1.0*uf + risk_term + terrain_term - 0.05*travel
                     + critical + shadow_bonus + relay_explorer_bonus + dead_bonus - load)

        else:
            # ── UNKNOWN ZONE: bid on information-gain potential only ──────────
            if local_unknown_frac < 0.05: return None   # nothing to learn here

            # Feasibility check applies even for unknown zones — a boat that
            # already knows it's landlocked should not bid on dry zones it can't reach
            stats_for_feasibility = self.zone_stats(zone)
            if not self.zone_feasible(r, stats_for_feasibility, zone=zone): return None

            # Boats avoid zones with no water within sensor range
            if r.name.startswith("Boat"):
                nearby_water = int(np.count_nonzero(
                    r.terrain_belief[max(0,cx-20):cx+20, max(0,cy-20):cy+20] == T_WATER))
                if nearby_water == 0: return None

            # Information gain: unknown fraction in robot's local belief
            info_gain = local_unknown_frac

            # ── Terrain-affinity bonus (world truth, always known) ────────────
            # A Legged robot should strongly prefer stair zones — they are its
            # exclusive terrain and the only way to reach building survivors.
            # A Drone similarly prefers stair zones (AIR capability).
            # This uses _zone_stair_frac (world truth) so the signal fires even
            # before the robot has personally explored the zone.
            fs_world = self._zone_stair_frac.get(zone, 0.0)
            fw_world = float(np.any(self._world_water_arr[x0:x1, y0:y1]))
            terrain_affinity = 0.0
            if r.name.startswith("Legged"):
                terrain_affinity = 2.8 * fs_world - 0.5 * fw_world
            elif r.name.startswith("Drone"):
                terrain_affinity = 1.5 * fs_world
            elif r.name.startswith("Boat"):
                terrain_affinity = 2.0 * fw_world - 0.5 * fs_world
            # Critical-access bonus: stair zones are only reachable by Legged/Drone
            # so they have asymmetric value — count it even in the unknown branch
            can_enter = bool(r.caps_mask & (CAP_STAIRS | CAP_AIR))
            critical_bonus = 2.5 * fs_world if can_enter else 0.0

            # Pull explorers into bubble only when relay is physically active.
            # Disc zones additionally get a pre-relay urgency pull — they represent
            # large unknown areas that must be explored but have no terrain affinity
            # bonus to drive robots toward them before a relay exists.
            zone_type_unk = self._shadow_zone_type.get(zone, 'none')
            if relay_active:
                relay_explorer_bonus = 3.0 * shadow_frac * local_unknown_frac
            elif relay_needed and zone_type_unk == 'stair' and can_enter:
                # Pre-relay pull: same logic as known-zone branch — approach
                # triggers relay election even before relay is confirmed.
                relay_explorer_bonus = 2.0 * shadow_frac * local_unknown_frac
            elif zone_type_unk == 'disc' and shadow_frac > 0.05:
                relay_explorer_bonus = 1.2 * shadow_frac * local_unknown_frac
            else:
                relay_explorer_bonus = 0.0

            # Slight diversity bonus: prefer zones far from current task
            diversity = 0.1 * travel

            dead_bonus = 1.5 * self._dead_in_zone_cache.get(zone, 0)

            u = (1.5 * info_gain + terrain_affinity + critical_bonus
                 - 0.08*travel + relay_explorer_bonus - diversity + dead_bonus - load)

        # ── Zone age bonus (last-mile fix) ───────────────────────────────────
        # Zones unvisited for a long time get an escalating utility boost.
        # This prevents corners of the map from being permanently neglected
        # when nearby higher-utility zones keep winning every CBBA round.
        last_visit = getattr(self, '_zone_last_visited', {}).get(zone, 0)
        ticks_neglected = self.timestep - last_visit
        if ticks_neglected > 100:
            # Escalates from 0 at 100 ticks to max 2.0 at 500+ ticks
            age_bonus = min(2.0, (ticks_neglected - 100) / 200.0)
            u += age_bonus

        # ── Battery urgency adjustment ───────────────────────────────────────
        # Low-battery robots should prioritise nearby completable zones and
        # avoid bidding on distant zones they'll never finish. The last
        # stair-capable robot gets a strong urgency bonus on uncleared buildings.
        batt_frac = r.battery / max(1.0, MAX_BATTERY)
        if batt_frac < 0.30:
            scan_budget = r.battery - dist
            if scan_budget < 20:
                u -= 3.0   # can't reach — heavy penalty
            else:
                u += min(1.5, scan_budget / 100.0) * \
                     self.zone_stats(zone)['unknown_frac']

            # Last capable robot urgency
            if bool(r.caps_mask & (CAP_STAIRS | CAP_AIR)):
                other_capable = other_capable_cache if other_capable_cache is not None else sum(
                    1 for rr in self.robots
                    if rr is not r and rr.active
                    and bool(rr.caps_mask & (CAP_STAIRS | CAP_AIR))
                    and rr.role != Role.RELAY
                )
                fs_zone = self._zone_stair_frac.get(zone, 0.0)
                zone_uf = self.zone_stats(zone)['unknown_frac']
                if other_capable == 0 and fs_zone > 0.05 and zone_uf > 0.10:
                    u += 5.0 * zone_uf * (1.0 - batt_frac)

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
        # Only treat confidence-decayed cells as "unknown" if they have hazard
        # readings that may have changed — terrain type is permanent and should
        # not drive revisits once scanned.
        uT_slc = self.union_T[x0:x1, y0:y1]
        uR_slc = self.union_R[x0:x1, y0:y1]
        has_hazard = ~(np.isnan(uT_slc) & np.isnan(uR_slc))
        stale    = int(np.count_nonzero(
            (tb_slc != T_UNKNOWN) & (conf_slc < CONF_UNCERTAIN) & has_hazard))
        return (unknown + stale) / total

    def _fallback_zone(self, r):
        """
        Assign a zone to a robot with no bundle.  Prefer unowned zones with
        high unknown fraction — this catches peripheral zones that the main
        CBBA bundle-building missed because robots near the centre filled
        their bundles with closer zones first.
        """
        counts = {rr.name: len(rr.bundle) for rr in self.robots}
        best_z = None; best_u = -1e18

        # First pass: unowned zones with uf > 0.3 — these are the peripheral
        # zones that got skipped entirely.  Give them priority over utility score.
        for zx in range(self.zone_nx):
            for zy in range(self.zone_ny):
                z = (zx, zy)
                if self.zone_stats(z)['unknown_frac'] < 0.30: continue
                if self.zone_tasks[z].owners: continue  # already owned
                u = self._zone_utility(r, z, counts)
                if u is not None and u > best_u:
                    best_u = u; best_z = z

        if best_z is not None:
            return best_z

        # Second pass: any zone by utility (original behaviour)
        for zx in range(self.zone_nx):
            for zy in range(self.zone_ny):
                z = (zx, zy)
                u = self._zone_utility(r, z, counts)
                if u is not None and u > best_u:
                    best_u = u; best_z = z
        return best_z

    # ── role assignment ───────────────────────────────────────────────────────
    def _decide_roles(self):
        """
        Pure potential-game role assignment over {SCOUT, SCAN, LOITER, RELAY}.
        Clusters are static (shadow never changes) — built once on first call.
        """
        t      = self.timestep
        active = [r for r in self.robots if r.active and r.battery > 0]
        shadow = self.radio_shadow

        # ── 1. Clusters are static — build once, reuse every tick ────────────
        if not self._last_clusters:
            shadow_zones = [
                (zx, zy)
                for zx in range(self.zone_nx)
                for zy in range(self.zone_ny)
                if self._shadow_zone_type.get((zx,zy), 'none') != 'none'
            ]
            visited_z = set(); clusters = []
            for sz in shadow_zones:
                if sz in visited_z: continue
                cluster = []; q = deque([sz]); visited_z.add(sz)
                while q:
                    z2 = q.popleft(); cluster.append(z2)
                    for nz in self.zone_neighbors4(z2):
                        if nz not in visited_z and self._shadow_zone_type.get(nz,'none') != 'none':
                            if frozenset((z2, nz)) in self._shadow_zone_adj:
                                visited_z.add(nz); q.append(nz)
                max_sf = max((self._zone_shadow_frac.get(z, 0.0) for z in cluster), default=0.0)
                if max_sf >= 0.10:
                    clusters.append(cluster)
            self._last_clusters = clusters
        clusters = self._last_clusters

        # ── 2. Safety: demote any relay that drifted into shadow ──────────────
        for r in active:
            if r.role == Role.RELAY and shadow[r.pos[0], r.pos[1]]:
                r.role = Role.SCAN
                r.relay_hold_until  = 0
                r.role_locked_until = t + 10

        # ── 3. Potential-game best-response over ALL four roles ───────────────
        self._pg_best_response_roles(active, clusters)



    # _global_relay_count removed — inlined at call sites

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

    # ── Heterogeneous Global Potential Game constants ─────────────────────────
    # Effort weights w(a): marginal exploration value of one robot in role a.
    # SCOUT is a sacrifice role — high value but very high congestion cost so
    # the equilibrium only elects 1-2 SCOUTs per cluster. Sending more than one
    # robot into a dangerous zone is wasteful; SCOUT congestion must be steep.
    _W = {Role.SCOUT: 3.0, Role.SCAN: 1.2, Role.LOITER: 0.1}

    # Congestion coefficients γ(a): cost per additional robot in role a.
    # SCOUT has γ=1.5 so equilibrium stops at 2 SCOUTs (3.0 - 1.5 = 1.5 > SCAN,
    # but 3.0 - 3.0 = 0.0 < SCAN at 1.2 - 0.5 = 0.7, so 3rd SCOUT never elected).
    _GAMMA = {Role.SCOUT: 1.5, Role.SCAN: 0.5, Role.LOITER: 0.4, Role.RELAY: 0.8}

    # Private distance cost weight for relay (Lemma 2.7 — preserves potential).
    _RELAY_TRAVEL_W = 3.0

    def _capability_yield(self, robot, zone) -> float:
        """
        How much does this robot type actually contribute to exploring `zone`?

        This is the key heterogeneity term that upgrades the local potential game
        to a global heterogeneous game.  A Rover in a stair zone can't enter the
        building — its yield is near zero.  A Legged robot is the only ground type
        that can enter — its yield is maximum.

        Returns a value in [0, 1] representing fractional exploration contribution
        relative to the ideal robot for this zone type.  Used to scale
        exploration_gain in effort role utility so robots self-select into zones
        where they contribute most, without explicit capability checks.
        """
        if zone is None:
            return 1.0
        # Cache keyed by (caps_mask, zone) — result is deterministic given these two
        cy_cache = getattr(self, '_cy_cache', None)
        if cy_cache is None or getattr(self, '_cy_cache_tick', -1) != self.timestep:
            self._cy_cache = {}; self._cy_cache_tick = self.timestep
            cy_cache = self._cy_cache
        cy_key = (robot.caps_mask, zone)
        if cy_key in cy_cache:
            return cy_cache[cy_key]
        fs_world = self._zone_stair_frac.get(zone, 0.0)
        zx, zy = zone
        x0 = zx*self.zone_w_cells; x1 = min(x0+self.zone_w_cells, GRID_W)
        y0 = zy*self.zone_h_cells; y1 = min(y0+self.zone_h_cells, GRID_H)
        fw_world = float(np.any(self._world_water_arr[x0:x1, y0:y1]))

        can_stairs = bool(robot.caps_mask & (CAP_STAIRS | CAP_AIR))
        can_water  = bool(robot.caps_mask & CAP_WATER)
        can_land   = bool(robot.caps_mask & (CAP_LAND | CAP_AIR))
        is_boat    = can_water and not bool(robot.caps_mask & CAP_AIR)

        if fs_world > 0.10:
            v = 1.0 if can_stairs else (0.0 if is_boat else 0.15)
        elif fw_world > 0.3:
            v = 1.0 if is_boat else (0.2 if can_land else 0.5)
        elif is_boat:
            v = 1.0 if fw_world else 0.05
        else:
            v = 1.0
        cy_cache[cy_key] = v
        return v

    def _mean_field_signal(self, robot, zone) -> float:
        """Mean field approximation for inter-zone coordination. O(Z) using uf cache."""
        if zone is None:
            return 0.0
        zuf = self._zone_uf_cache
        uf = zuf.get(zone, 0.0)
        if uf < 0.05:
            return 0.0
        zt = self._shadow_zone_type.get(zone, 'none')

        n_capable_assigned = sum(
            1 for r2 in self.robots
            if r2.active and r2.role != Role.RELAY
            and r2.task_zone is not None
            and self._shadow_zone_type.get(r2.task_zone, 'none') == zt
            and self._capability_yield(r2, r2.task_zone) > 0.5
        )
        # Use precomputed uf cache — avoids 64 zone_stats calls per invocation
        n_zones_needing = sum(
            1 for z2, zt2 in self._shadow_zone_type.items()
            if zt2 == zt
            and zuf.get(z2, 0.0) > 0.10
            and (zt2 == 'none' or self._relay_ok_flood.get(z2, False))
        )
        if n_zones_needing == 0:
            return 0.0

        underservice = max(0.0, 1.0 - n_capable_assigned / max(1, n_zones_needing))
        cap_yield = self._capability_yield(robot, zone)
        return underservice * cap_yield * uf * 1.5

    def _relay_val(self, cluster):
        """
        Value unlocked by placing a relay at the border of `cluster`.

        Mechanism design formulation (replaces raw unknown-cell fraction):
        The relay provides a public good — it makes the building accessible to
        all explorers. The value is the expected survivor yield of the cluster
        discounted by how explored it already is, divided by the number of
        robots needed to realise that value (relay + explorers).

        relay_val = (expected_unfound_survivors × SURVIVOR_VALUE × stair_uf)
                    / n_enabling_robots
                  + explorers_inside_safety_bonus

        This fires from t=0 for fully unknown buildings (stair_uf=1.0, high
        expected survivors) rather than waiting for an explorer to approach
        first — solving the chicken-and-egg deadlock.

        Cached per cluster-key per tick.
        """
        cache_key = tuple(sorted(cluster))
        rv_cache = getattr(self, '_relay_val_cache', None)
        if rv_cache is None or getattr(self, '_relay_val_tick', -1) != self.timestep:
            self._relay_val_cache = {}
            self._relay_val_tick  = self.timestep
            rv_cache = self._relay_val_cache
        if cache_key in rv_cache:
            return rv_cache[cache_key]

        cluster_zones = set(cluster)
        cluster_type  = self._shadow_zone_type.get(cluster[0], 'none') if cluster else 'none'
        zuf           = self._zone_uf_cache
        zone_size     = self.zone_w_cells * self.zone_h_cells

        # Bounding box for inside-check
        x0_min = self.world.w; x1_max = 0
        y0_min = self.world.h; y1_max = 0
        for z in cluster:
            zx, zy = z
            x0 = zx*self.zone_w_cells; x1 = min(x0+self.zone_w_cells, self.world.w)
            y0 = zy*self.zone_h_cells; y1 = min(y0+self.zone_h_cells, self.world.h)
            x0_min=min(x0_min,x0); x1_max=max(x1_max,x1)
            y0_min=min(y0_min,y0); y1_max=max(y1_max,y1)

        # Safety override: explorer inside shadow → relay is life-or-death.
        # This dominates everything else — the game must always keep a relay
        # when robots are at risk inside.
        explorers_inside = sum(
            1 for r in self.robots
            if r.active and r.role != Role.RELAY
            and x0_min <= r.pos[0] < x1_max
            and y0_min <= r.pos[1] < y1_max
            and self.radio_shadow[r.pos[0], r.pos[1]]
        )
        if explorers_inside:
            val = 50.0 * explorers_inside
            rv_cache[cache_key] = val
            return val

        if cluster_type == 'stair':
            # ── Mechanism design: expected survivor value ─────────────────────
            # Count unknown stair cells — the resource that relay makes accessible
            unknown_stair = 0
            total_stair   = 0
            for z in cluster:
                zx, zy = z
                x0 = zx*self.zone_w_cells; x1 = min(x0+self.zone_w_cells, self.world.w)
                y0 = zy*self.zone_h_cells; y1 = min(y0+self.zone_h_cells, self.world.h)
                sm  = self._world_stair_arr[x0:x1, y0:y1]
                unk = (self.union_belief[x0:x1, y0:y1] == T_UNKNOWN)
                unknown_stair += int(np.count_nonzero(sm & unk))
                total_stair   += int(np.sum(sm))

            if total_stair == 0 or unknown_stair == 0:
                # Building fully explored — minimal holding value so relay
                # demotes quickly and moves to the next building
                val = 0.2
                rv_cache[cache_key] = val
                return val

            stair_uf = unknown_stair / total_stair  # fraction of stair cells still unknown

            # Expected unfound survivors in this cluster.
            # Use world-truth survivor count (robot can't see inside yet, but
            # the mechanism designer — i.e. us — knows the ground truth).
            # This is the key change: relay value is proportional to survivor
            # density, not just unknown fraction. A building with 3 survivors
            # at 10% explored is worth far more relay duty than open terrain
            # at 100% unknown with 0 survivors.
            # World-truth unfound survivors in this cluster (cached per tick)
            surv_cache = getattr(self, '_cluster_surv_cache', None)
            if surv_cache is None or getattr(self, '_cluster_surv_tick', -1) != self.timestep:
                self._cluster_surv_cache = {}; self._cluster_surv_tick = self.timestep
                surv_cache = self._cluster_surv_cache
            ck = cache_key
            if ck not in surv_cache:
                surv_cache[ck] = sum(
                    1 for s in self.survivors
                    if s not in self.found
                    and self.cell_to_zone(s[0], s[1]) in cluster_zones
                )
            unfound_survivors = surv_cache[ck]

            # Prior: every building has non-zero relay value even before survivors
            # are confirmed. Scaled so a typical 150-stair-cell building yields
            # relay_val ≈ 3.5 at stair_uf=1.0 — enough to beat the worst-case
            # travel penalty (max = RELAY_TRAVEL_W = 3.0) for any building.
            AVG_STAIR_CELLS = 150.0
            survivor_prior  = max(1.0, total_stair / AVG_STAIR_CELLS)
            expected_survivors = max(unfound_survivors, survivor_prior)

            # The relay robot receives the full expected survivor value.
            # No VCG division: the relay must see sufficient private utility
            # to overcome the travel penalty for distant buildings. Setting
            # SURVIVOR_VALUE=3.5 ensures val >= 3.5 at stair_uf=1.0 which
            # strictly dominates the max travel penalty of 3.0.
            SURVIVOR_VALUE = 3.5
            val = expected_survivors * SURVIVOR_VALUE * stair_uf

            # Dead robot bonus
            dead_inside = sum(
                1 for r in self.robots
                if not r.active
                and self.cell_to_zone(r.pos[0], r.pos[1]) in cluster_zones
                and self.radio_shadow[r.pos[0], r.pos[1]]
            )
            val += dead_inside * 1.5

            # Hard floor: relay value must beat worst-case travel penalty
            val = max(val, 3.5)

        elif cluster_type == 'disc':
            # Disc: no stair cells, no survivors, but unknown coverage still matters.
            # Use raw unknown fraction scaled by cluster size — original formulation
            # is correct here since disc value is purely information, not survivor access.
            val = sum(zuf.get(z, 0.0) * self._zone_shadow_frac.get(z, 0.0) * zone_size
                      for z in cluster) / max(1, zone_size)
            val *= 1.5
            val  = min(val, 4.0)

            dead_inside = sum(
                1 for r in self.robots
                if not r.active
                and self.cell_to_zone(r.pos[0], r.pos[1]) in cluster_zones
                and self.radio_shadow[r.pos[0], r.pos[1]]
            )
            val += dead_inside * 1.5
        else:
            val = 0.0

        rv_cache[cache_key] = val
        return val

    def _role_utility_pg(self, robot, role, active, zone, cluster_info):
        """
        Heterogeneous global potential game utility for `robot` choosing `role`.

        Upgraded from local (anonymous count) to global (type-aware) game:
        - Effort utilities are scaled by capability_yield so robots self-select
          into zones where they actually contribute.
        - Congestion is computed over robots with similar capability yields,
          not total robot counts — a Rover and a Legged in a stair zone are
          not interchangeable and should not equally penalise each other.
        - Mean field signal provides fleet-wide rebalancing without O(R²) cost.

        Parameters
        ----------
        robot        : Robot instance (robot i)
        role         : Role being evaluated
        active       : list of all active robots (full joint state — global game)
        zone         : robot's current task_zone (may be None)
        cluster_info : dict {cluster_id -> (cluster_list, covered_bool)}

        Returns
        -------
        float utility
        """
        if role == Role.RELAY:
            # Pre-check: if relay cap is already met, return very low utility
            # so robots don't waste BR iterations competing for unavailable relay slots.
            # This is the key fix: without it robots see high relay_val and keep
            # trying to elect relay even when all clusters are covered, starving
            # SCOUT/SCAN of robots and causing cascade LOITER.
            #
            # CLAIM-BASED cap (replaces the old global head-count).  The previous
            # rule compared the TOTAL number of relays against the number of needy
            # buildings and returned a -5.0 sentinel when total >= needed.  That
            # wrongly counted relays still parked on ALREADY-FINISHED buildings as
            # "supply", so when two relays sat on done buildings the counter refused
            # to let any robot relay the buildings that genuinely had none — and it
            # also blocked a finished relay from rotating after demotion.  This is a
            # non-potential-game override that overrules the per-building utility.
            #
            # Correct rule: a building needs a relay iff it is an uncovered shadow
            # cluster with remaining demand AND no relay is already assigned to it
            # (unclaimed).  Only discourage becoming a relay when EVERY such building
            # already has a relay heading to it.  This keeps the decision per-cluster
            # and inside the game — a robot relays iff some unserved building's relay
            # value beats its own travel + congestion cost — and the congestion term
            # plus this claim check together keep it to one relay per building.
            # ── Model J: public-good value-SHARING replaces the hard relay cap ──
            # The previous code returned a hard -5.0 sentinel whenever every needy
            # building already had a relay assigned.  That sentinel is keyed off a
            # global predicate over other robots' (role, task_zone) and is a hard
            # cliff, so it is NOT an exact-potential term — it was the single in-game
            # override breaking the Monderer-Shapley property.
            #
            # Replacement: treat each cluster's coverage value as a shared public
            # good (Anshelevich et al. fair cost-sharing / Rosenthal congestion
            # structure).  The k-th relay committing to a cluster receives only
            # rv/k of that cluster's value, computed below per-cluster.  A second
            # relay on an already-served cluster therefore gets at most rv/2, which
            # — combined with travel + the existing congestion term — makes RELAY
            # lose to SCOUT/SCAN there without any cliff.  "One relay per cluster"
            # now EMERGES from the equilibrium instead of being hard-coded, and the
            # term is an anonymous congestion/cost-sharing payoff, i.e. an exact
            # potential (verify with --verify-potential).  No early return.
            D_NORM = float(self.world.w + self.world.h)
            best_val  = 0.0
            best_dist = D_NORM
            rx, ry    = robot.pos
            for cid, (cl, covered) in cluster_info.items():
                if covered: continue
                cluster_type = self._shadow_zone_type.get(cl[0], 'none') if cl else 'none'
                is_boat = bool(robot.caps_mask & CAP_WATER) and not bool(robot.caps_mask & CAP_AIR)
                is_land_only = bool(robot.caps_mask & CAP_LAND) and not bool(robot.caps_mask & (CAP_STAIRS|CAP_AIR|CAP_WATER))
                # boats can relay any cluster — any robot at a shadow border
                # relays the signal regardless of terrain type.
                if is_boat and cluster_type == 'disc':
                    # Boat can only relay disc if there are water/bridge border cells
                    # adjacent to the disc shadow — boats cannot stand on land border cells.
                    cid_disc = self._shadow_cluster_id.get(cl[0])
                    _brc = getattr(self, '_boat_reach_cache', None)
                    if _brc is None or getattr(self, '_boat_reach_tick', -1) != self.timestep:
                        self._boat_reach_cache = {}; self._boat_reach_tick = self.timestep
                        _brc = self._boat_reach_cache
                    brc_key = (robot.name, cid_disc)
                    if brc_key not in _brc:
                        robot.reachable()
                        if robot._reachable_arr is None:
                            _brc[brc_key] = False
                        else:
                            # Only water/bridge border cells — boats cannot stand on land
                            water_disc_border = [
                                (int(pt[0]), int(pt[1]))
                                for pt in self._shadow_border_cells_arr
                                if self.world.grid[int(pt[0])][int(pt[1])]["t"] in (T_WATER, T_BRIDGE)
                                and any(
                                    self._shadow_zone_type.get(self.cell_to_zone(int(nx2), int(ny2))) == 'disc'
                                    and self._shadow_cluster_id.get(self.cell_to_zone(int(nx2), int(ny2))) == cid_disc
                                    for nx2, ny2 in self.world.neighbours((int(pt[0]), int(pt[1])))
                                    if self.radio_shadow[int(nx2), int(ny2)]
                                )
                            ]
                            _brc[brc_key] = any(robot._reachable_arr[x, y] for x, y in water_disc_border)
                    if not _brc[brc_key]: continue
                rv = self._relay_val(cl)
                if rv <= 0: continue
                # Public-good value-sharing (Model J): the k-th relay committing to
                # this cluster gets rv/k.  Count other active robots already relaying
                # this cluster from the joint ROLE profile (anonymous load), so the
                # term is a cost-sharing congestion payoff = exact potential.
                n_share = sum(1 for rr in active
                              if rr is not robot and rr.active
                              and rr.role == Role.RELAY and rr.task_zone in cl)
                rv = rv / (n_share + 1)
                min_d = min(
                    abs(rx - (z[0]*self.zone_w_cells + self.zone_w_cells//2))
                    + abs(ry - (z[1]*self.zone_h_cells + self.zone_h_cells//2))
                    for z in cl
                )
                net = rv - (min_d / D_NORM) * self._RELAY_TRAVEL_W
                best_net = best_val - (best_dist / D_NORM) * self._RELAY_TRAVEL_W
                if net > best_net:
                    best_val = rv; best_dist = min_d

            best_cl_zones = set()
            for cid2, (cl2, _) in cluster_info.items():
                if not cl2: continue
                rv2 = self._relay_val(cl2)
                min_d2 = min(
                    abs(robot.pos[0] - (z[0]*self.zone_w_cells + self.zone_w_cells//2))
                    + abs(robot.pos[1] - (z[1]*self.zone_h_cells + self.zone_h_cells//2))
                    for z in cl2
                )
                net2 = rv2 - (min_d2 / D_NORM) * self._RELAY_TRAVEL_W
                if abs(net2 - (best_val - (best_dist / D_NORM) * self._RELAY_TRAVEL_W)) < 0.01:
                    best_cl_zones = set(cl2); break

            n_relays_this_cluster = sum(
                1 for r2 in self.robots
                if r2 is not robot and r2.active and r2.role == Role.RELAY
                and r2.task_zone in best_cl_zones
            ) if best_cl_zones else 0
            congestion = (n_relays_this_cluster + 1) * self._GAMMA[Role.RELAY]
            travel     = (best_dist / D_NORM) * self._RELAY_TRAVEL_W

            explorer_penalty = 0.0
            if (bool(robot.caps_mask & (CAP_STAIRS | CAP_AIR))
                    and zone is not None
                    and self._shadow_zone_type.get(zone) == 'stair'
                    and self._relay_ok_flood.get(zone, False)):
                zx2, zy2 = zone
                zmin_x = zx2 * self.zone_w_cells; zmax_x = zmin_x + self.zone_w_cells
                zmin_y = zy2 * self.zone_h_cells; zmax_y = zmin_y + self.zone_h_cells
                rx2, ry2 = robot.pos
                dist_to_zone = max(0, zmin_x - rx2, rx2 - zmax_x + 1,
                                   zmin_y - ry2, ry2 - zmax_y + 1)
                fade = max(0.0, 1.0 - dist_to_zone / (robot.terrain_R * 4))
                explorer_penalty = 6.0 * fade

            # ── Robot-type relay preference ──────────────────────────────────
            # Rovers and Boats are low-opportunity-cost relays: they can't enter
            # buildings (Rovers) or traverse land (Boats), so giving up exploration
            # to hold a relay position loses relatively little.
            #
            # Legged and Drone robots are high-opportunity-cost relays: they are
            # the ONLY types that can enter buildings. Every tick spent relaying
            # is a tick not exploring stair zones that nobody else can reach.
            #
            # Boat exception: if there is significant unexplored water that only
            # the boat can reach, the boat should explore not relay.
            # All fractions cached per tick to avoid per-robot recomputation.

            is_boat      = bool(robot.caps_mask & CAP_WATER) and not bool(robot.caps_mask & CAP_AIR)
            is_land_only = bool(robot.caps_mask & CAP_LAND) and not bool(robot.caps_mask & (CAP_STAIRS|CAP_AIR|CAP_WATER))
            is_stair_capable = bool(robot.caps_mask & (CAP_STAIRS | CAP_AIR))

            # Tick-level cache for the aggregate terrain fractions
            _tp_cache = getattr(self, '_type_pref_cache', None)
            if _tp_cache is None or getattr(self, '_type_pref_tick', -1) != self.timestep:
                zuf = self._zone_uf_cache
                stair_ufs = [zuf.get(z, 0.0) for z, zt in self._shadow_zone_type.items() if zt == 'stair']
                water_ufs = [zuf.get(z, 0.0) for z in self.zone_tasks
                             if self._world_water_arr[
                                 z[0]*self.zone_w_cells + self.zone_w_cells//2,
                                 z[1]*self.zone_h_cells + self.zone_h_cells//2]]
                self._type_pref_cache = {
                    'stair_uf': float(np.mean(stair_ufs)) if stair_ufs else 0.0,
                    'water_uf': float(np.mean(water_ufs)) if water_ufs else 0.0,
                }
                self._type_pref_tick = self.timestep
                _tp_cache = self._type_pref_cache

            type_preference = 0.0

            if is_land_only:
                # Rovers are ideal relays (they cannot enter buildings, so relay
                # duty costs them little exploration value).  FIX (Model J): the old
                # preference scaled by a GLOBAL free-Rover count and reached +8, which
                # dwarfed the building value (~3.5) so relay election was driven by
                # "am I a Rover" rather than "is this building worth opening" — any
                # building with no nearby Rover was left unopened, the cross-seed
                # inconsistency.  It also read other robots' ROLES (the last non-
                # exact term).  Replace with a MODEST capability-based constant plus a
                # demand term scaled by the number of UNCOVERED buildings (fixed
                # state, not roles): a self-term that keeps Rovers clearly preferred
                # while letting the building value dominate, so every building with a
                # nearby suitable robot of ANY type gets opened.
                n_uncovered = sum(1 for _, (cl, cov) in cluster_info.items()
                                  if not cov and cl
                                  and self._shadow_zone_type.get(cl[0], 'none') != 'none')
                type_preference = 2.0 + 1.0 * min(2.0, 0.5 * n_uncovered)

            elif is_boat:
                unexplored_water_frac = _tp_cache['water_uf']
                other_boats_exploring = sum(
                    1 for rr in active
                    if rr is not robot
                    and bool(rr.caps_mask & CAP_WATER)
                    and not bool(rr.caps_mask & CAP_AIR)
                    and rr.role != Role.RELAY
                )
                solo_water_penalty = unexplored_water_frac * (1.5 if other_boats_exploring == 0 else 0.5)

                # When water is mostly explored, boats transition to relaying the
                # disc shadow — it often overlaps or borders the river.
                disc_relay_bonus = 0.0
                if unexplored_water_frac < 0.3:
                    n_disc_uncovered = sum(
                        1 for z, zt in self._shadow_zone_type.items()
                        if zt == 'disc'
                        and not self._relay_ok_flood.get(z, False)
                    )
                    # Check if any disc border cell is adjacent to known water
                    boat_can_reach_disc = getattr(self, '_boat_disc_reachable', None)
                    if boat_can_reach_disc is None or getattr(self, '_boat_disc_reach_tick', -1) != self.timestep:
                        brd_pts = self._shadow_border_cells_arr
                        self._boat_disc_reachable = any(
                            self._shadow_zone_type.get(
                                self.cell_to_zone(int(p[0])+dx, int(p[1])+dy), 'none') == 'disc'
                            and self.union_belief[
                                max(0,min(GRID_W-1,int(p[0])+dx2)),
                                max(0,min(GRID_H-1,int(p[1])+dy2))] == T_WATER
                            for p in brd_pts
                            for dx, dy in ((1,0),(-1,0),(0,1),(0,-1))
                            for dx2, dy2 in ((1,0),(-1,0),(0,1),(0,-1))
                        )
                        self._boat_disc_reach_tick = self.timestep
                        boat_can_reach_disc = self._boat_disc_reachable

                    if n_disc_uncovered > 0 and boat_can_reach_disc:
                        water_done_frac = 1.0 - unexplored_water_frac
                        disc_relay_bonus = 3.5 * water_done_frac * min(1.0, n_disc_uncovered / 2.0)

                type_preference = +1.5 - solo_water_penalty + disc_relay_bonus


            elif is_stair_capable:
                unexplored_stair_frac = _tp_cache['stair_uf']
                # Opportunity-cost penalty for a building-capable robot relaying.
                # FIX (Model J): the old version scaled by a GLOBAL free-Rover count,
                # which is the wrong signal for a spatial decision — when free Rovers
                # existed but were all far from THIS building, every nearby Legged was
                # strongly discouraged yet no Rover actually came (too far), so the
                # building went uncovered; and that term read other robots' ROLES,
                # breaking exactness.  Replace with a LOCAL test: yield relay duty to
                # a lower-opportunity-cost Rover/Boat only if one is actually CLOSER to
                # the building this robot would relay.  If this robot is the nearest
                # suitable relay, the penalty fades to ~0 so it takes the slot rather
                # than leave the building unprovisioned.  Uses fixed POSITIONS only
                # (not roles) -> a self-term that does not add to the potential
                # residual.
                closer_suitable = False
                if best_cl_zones:
                    _bz = [(z[0]*self.zone_w_cells + self.zone_w_cells//2,
                            z[1]*self.zone_h_cells + self.zone_h_cells//2)
                           for z in best_cl_zones]
                    for rr in active:
                        if rr is robot:
                            continue
                        rr_land = (bool(rr.caps_mask & CAP_LAND)
                                   and not bool(rr.caps_mask & (CAP_STAIRS | CAP_AIR | CAP_WATER)))
                        rr_boat = (bool(rr.caps_mask & CAP_WATER)
                                   and not bool(rr.caps_mask & CAP_AIR))
                        if not (rr_land or rr_boat):
                            continue
                        rr_d = min(abs(rr.pos[0] - cx) + abs(rr.pos[1] - cy) for cx, cy in _bz)
                        if rr_d < best_dist:
                            closer_suitable = True
                            break
                # nearest suitable relay -> ~no penalty (must take it); otherwise a
                # SOFT yield, not a hard one: the position test cannot see whether the
                # closer Rover is actually free, so a strong penalty starved buildings
                # whose nearest Rover was busy elsewhere (it yielded but nobody came).
                # A soft penalty keeps Rovers clearly preferred — the priority-sort +
                # value-sharing hand a free Rover the slot first — while leaving the
                # Legged a viable BACKUP that opens the building rather than let it sit
                # unopened.  Still position-only, so still a self-term.
                type_preference = -(1.5 * unexplored_stair_frac) if closer_suitable else \
                                  -(0.5 * unexplored_stair_frac)

            return best_val - congestion - travel - explorer_penalty + type_preference

        # ── Effort roles: SCOUT / SCAN / LOITER — heterogeneous global game ──
        #
        # Key upgrade over local game:
        #   exploration_gain = uf × w(role) × capability_yield(robot, zone)
        #
        # This means a Rover evaluating a stair zone gets yield=0.15, so its
        # exploration_gain is 15% of what a Legged robot would get for the same
        # zone.  The Rover won't outbid the Legged for that zone because its
        # marginal contribution is genuinely lower.
        #
        # Congestion is also capability-weighted: only count robots with
        # similar capability yield as true competitors for this zone.
        # A Rover and a Legged in a stair zone are not congesting each other
        # in the same way — the Legged can enter, the Rover cannot.

        _zuf = self._zone_uf_cache
        if zone is not None:
            uf = _zuf.get(zone, 0.0)
        else:
            uf = 0.0

        if uf < 0.05:
            # Use precomputed cache for best_uf scan — avoids 64 zone_stats calls
            best_uf = max(_zuf.values()) if _zuf else 0.0
            uf = best_uf

        # Capability yield for this robot in this zone
        yield_i = self._capability_yield(robot, zone)

        w = self._W[role]
        exploration_gain = uf * w * yield_i

        # ── Endgame reward-shaping (last-mile completeness) ──
        # uf×w decays toward zero as the map fills, so late in a run the best
        # available exploration gain sinks below the small LOITER baseline and
        # robots rationally stop exploring even though reachable unknown cells
        # (and possibly the last survivor) remain — the "motivation collapse"
        # last-mile stall.  But the true value of a cell does not fall to zero as
        # coverage rises; the final unknown cells are the MOST valuable because an
        # as-yet-unfound survivor must be in one of them.  So amplify exploration
        # value as global coverage approaches 1.0.  Below C0 the multiplier is
        # exactly 1.0 — the main exploration phase and its equilibrium are
        # untouched; only the endgame tail is reshaped.  This stays inside the
        # potential game: every robot still self-selects its role by utility, we
        # have only corrected a reward gradient that flattened too early.
        if role in (Role.SCOUT, Role.SCAN) and exploration_gain > 0.0:
            if getattr(self, '_union_cov_tick', -1) != self.timestep:
                self._union_cov_cache = float(np.mean(self.union_belief != T_UNKNOWN))
                self._union_cov_tick = self.timestep
            union_cov = self._union_cov_cache
            _C0 = 0.85
            if union_cov > _C0:
                exploration_gain *= 1.0 + 9.0 * (union_cov - _C0) / (1.0 - _C0)

        # ── Local building-closure incentive (Model J) ──
        # The global boost above only fires once the WHOLE map is ~85% covered;
        # mid-mission a robot that entered a building but left a few unknown cells
        # leaves for a larger frontier elsewhere, because the building's remaining
        # unknown fraction uf is tiny so uf*w*yield ~ 0.  Once THIS robot is in /
        # assigned to a stair building it can enter and that building is partly-
        # but-not-fully explored, add a flat closure incentive so finishing beats
        # leaving.  This is a SELF-TERM: it depends only on this robot's own role,
        # position, capability and the building's FIXED unknown fraction — never on
        # another robot's choice — so it cancels in every Monderer-Shapley 4-cycle
        # and leaves the potential residual unchanged (check --verify-potential).
        # It vanishes when the building is done (z_uf -> 0), so a robot is never
        # trapped in a finished building.
        if (role in (Role.SCOUT, Role.SCAN) and zone is not None and yield_i > 0.5
                and self._shadow_zone_type.get(zone) == 'stair'):
            z_uf = _zuf.get(zone, 0.0)
            if 0.0 < z_uf < 0.5:
                inside = bool(self.radio_shadow[robot.pos[0], robot.pos[1]])
                exploration_gain += (3.0 if inside else 1.0) * yield_i

        # SCOUT gets an additional hazard-zone bonus: it is specifically elected
        # to explore zones that SCAN robots route around due to high risk.
        # This makes SCOUT utility highest precisely where SCAN is least effective,
        # creating a clean role separation rather than degree-of-aggressiveness.
        if role == Role.SCOUT and zone is not None:
            stats = self.zone_stats(zone)
            avg_hazard = max(stats['avgT'] / max(1e-6, robot.temp_limit),
                             stats['avgR'] / max(1e-6, robot.rad_limit))
            # Bonus scales with hazard level — the more dangerous, the more
            # valuable it is to have a sacrifice robot rather than a SCAN robot
            # that will avoid the zone entirely
            exploration_gain += avg_hazard * 2.0 * yield_i

        # Capability-weighted congestion: only robots with yield > 0.5 in this
        # zone type count as true competitors.  Use a per-tick yield cache to
        # avoid recomputing _capability_yield for every (robot, zone) pair in
        # the inner loop — this keeps the BR iteration fast.
        _ycache = getattr(self, '_yield_cache', None)
        if _ycache is None or getattr(self, '_yield_cache_tick', -1) != self.timestep:
            self._yield_cache = {}; self._yield_cache_tick = self.timestep
            _ycache = self._yield_cache
        def _cy(r2, z2):
            k = (id(r2), z2)
            if k not in _ycache: _ycache[k] = self._capability_yield(r2, z2)
            return _ycache[k]

        n_effective_competitors = sum(
            1 for r2 in active
            if r2 is not robot
            and r2.role == role
            and (
                # Robot has a zone assigned — check capability yield properly
                (r2.task_zone is not None
                 and _cy(r2, r2.task_zone) > 0.5
                 and _cy(r2, zone) > 0.5)
                or
                # Robot has no zone yet (e.g. t=0) — count by spatial proximity
                # as a proxy for congestion. Robots within 30 cells are competing
                # for the same local frontier regardless of formal assignment.
                (r2.task_zone is None
                 and abs(r2.pos[0] - robot.pos[0]) + abs(r2.pos[1] - robot.pos[1]) < 30
                 and _cy(r2, zone) > 0.5)
            )
        )
        congestion = (n_effective_competitors + 1) * self._GAMMA[role]

        # Mean field signal: fleet-wide rebalancing bonus — use tick-level cache
        # to avoid recomputing identical zone×robot-type queries across the BR loop.
        if role != Role.LOITER:
            mf_cache_key = (zone, robot.caps_mask)
            _mf_cache = getattr(self, '_mf_cache', None)
            if _mf_cache is None or getattr(self, '_mf_cache_tick', -1) != self.timestep:
                self._mf_cache = {}; self._mf_cache_tick = self.timestep
                _mf_cache = self._mf_cache
            if mf_cache_key not in _mf_cache:
                _mf_cache[mf_cache_key] = self._mean_field_signal(robot, zone)
            mf_bonus = _mf_cache[mf_cache_key]
        else:
            mf_bonus = 0.0

        loiter_penalty = 0.0
        if role == Role.LOITER and uf > 0.05:
            loiter_penalty = 1.2 * uf
        if role == Role.LOITER and zone is not None:
            if (self._shadow_zone_type.get(zone) == 'stair'
                    and self._relay_ok_flood.get(zone, False)
                    and bool(robot.caps_mask & (CAP_STAIRS | CAP_AIR))):
                loiter_penalty += 4.0

        return exploration_gain - congestion - loiter_penalty + mf_bonus

    def _pg_best_response_roles(self, active, clusters):
        """
        Best-response iteration over {SCOUT, SCAN, LOITER, RELAY} for all eligible
        robots.  Converges to a pure Nash equilibrium because the game is an exact
        potential game with private costs (Monderer & Shapley 1996, Theorem 4.5 +
        Lemma 2.7).  Full proof in potential_game_proof.docx.

        cluster_info is built here and passed to _role_utility_pg so relay utility
        can compute marginal public-good value without re-scanning clusters per robot.
        """
        t      = self.timestep
        shadow = self.radio_shadow

        # Robots eligible for role reassignment this tick:
        #   - not role-locked (hold timer active)
        #   - not stranded inside uncovered shadow (those must evacuate first)
        eligible = [
            r for r in active
            if t >= r.role_locked_until
            and not (shadow[r.pos[0], r.pos[1]]
                     and not self.relay_ok_extended(
                         self.cell_to_zone(r.pos[0], r.pos[1])))
        ]
        if not eligible:
            return

        # Performance: build a signature of what has changed since last BR.
        # Robots whose position, role, and relay_ok environment are unchanged
        # are in the same Nash equilibrium state — skip them this tick.
        # Always include: robots in uncovered shadow (safety), newly-elected relays,
        # and any robot whose task_zone relay_ok status changed.
        relay_ok_sig = frozenset(z for z,ok in self._relay_ok_flood.items() if ok)
        self._last_relay_ok_sig = relay_ok_sig

        cluster_info = {}
        for cid, cl in enumerate(clusters):
            covered = any(self._relay_ok_flood.get(z, False) for z in cl)
            cluster_info[cid] = (cl, covered)

        all_roles = [Role.SCOUT, Role.SCAN, Role.LOITER, Role.RELAY]

        # Relay preference sort key: within each BR iteration, process low-opportunity-cost
        # robots (Rovers, Boats) before high-opportunity-cost ones (Legged, Drone) so that
        # when the relay cap is reached, it's Rovers/Boats holding the slots not Legged/Drone.
        # The shuffle still runs to avoid bias for non-relay roles; this sort just determines
        # which robots see open relay slots first.
        def _relay_priority(r):
            is_land_only = bool(r.caps_mask & CAP_LAND) and not bool(r.caps_mask & (CAP_STAIRS|CAP_AIR|CAP_WATER))
            is_boat = bool(r.caps_mask & CAP_WATER) and not bool(r.caps_mask & CAP_AIR)
            if is_land_only: return 0   # Rovers first
            if is_boat:      return 1   # Boats second
            return 2                    # Legged/Drone last

        # Best-response loop — terminates because each step strictly increases Φ
        # and the joint action space is finite.
        max_iters = len(eligible) + 1
        for _iter in range(max_iters):
            changed = False
            random.shuffle(eligible)   # avoid systematic ordering bias
            eligible.sort(key=_relay_priority)  # then stable-sort by relay preference

            for robot in eligible:
                # Skip robots that became locked during this BR iteration
                if t < robot.role_locked_until:
                    continue

                # Full active list passed to utility — global heterogeneous game
                # (no anonymous s_{-i} counts; robot identities fully visible)
                best_role = robot.role
                best_u    = self._role_utility_pg(
                    robot, robot.role, active, robot.task_zone, cluster_info)

                for role in all_roles:
                    if role == robot.role:
                        continue
                    u = self._role_utility_pg(
                        robot, role, active, robot.task_zone, cluster_info)
                    if u > best_u + 1e-9:
                        best_u = u; best_role = role

                if best_role == robot.role:
                    continue

                # Role change
                old_role  = robot.role
                robot.role = best_role
                changed    = True

                if best_role == Role.RELAY:
                    # No global head-count cap here.  One-relay-per-building is
                    # enforced per-cluster by the `already_held` skip in the best_cl
                    # selection loop below: a robot can only claim an uncovered
                    # building that no other relay is already assigned to, and if no
                    # such building exists best_cl stays None and the robot reverts
                    # to its old role.  Over-election is therefore impossible (no
                    # unclaimed building => no relay), and a relay that finishes its
                    # building can rotate onto the next unclaimed one — which the old
                    # count-based cap prevented by counting parked-on-finished relays
                    # as supply.

                    # Initialise relay navigation state for _move_relay
                    # Target: most-valuable uncovered cluster nearest to this robot
                    rx, ry = robot.pos
                    D_NORM = float(self.world.w + self.world.h)
                    best_cl = None; best_net = -1e9
                    for cid, (cl, covered) in cluster_info.items():
                        if covered: continue
                        # Boats can't serve stair clusters (land-only border).
                        # Land-only robots (Rovers) also can't serve stair clusters —
                        # they can't enter the building so a Rover relay wastes a slot
                        # that should go to an actual building-capable explorer.
                        # Boats also can't serve a disc cluster unless the disc border
                        # has water — a landlocked disc has no water route to its border
                        # so a boat relay would be elected but never arrive.
                        cluster_type = self._shadow_zone_type.get(cl[0], 'none') if cl else 'none'
                        is_boat = bool(robot.caps_mask & CAP_WATER) and not bool(robot.caps_mask & CAP_AIR)
                        is_land_only = bool(robot.caps_mask & CAP_LAND) and not bool(robot.caps_mask & (CAP_STAIRS|CAP_AIR|CAP_WATER))
                        if is_boat and cluster_type == 'disc':
                            cid_disc = self._shadow_cluster_id.get(cl[0])
                            _brc = getattr(self, '_boat_reach_cache', None)
                            if _brc is None or getattr(self, '_boat_reach_tick', -1) != self.timestep:
                                self._boat_reach_cache = {}; self._boat_reach_tick = self.timestep
                                _brc = self._boat_reach_cache
                            brc_key = (robot.name, cid_disc)
                            if brc_key not in _brc:
                                robot.reachable()
                                if robot._reachable_arr is None:
                                    _brc[brc_key] = False
                                else:
                                    water_disc_border = [
                                        (int(pt[0]), int(pt[1]))
                                        for pt in self._shadow_border_cells_arr
                                        if self.world.grid[int(pt[0])][int(pt[1])]["t"] in (T_WATER, T_BRIDGE)
                                        and any(
                                            self._shadow_zone_type.get(self.cell_to_zone(int(nx2), int(ny2))) == 'disc'
                                            and self._shadow_cluster_id.get(self.cell_to_zone(int(nx2), int(ny2))) == cid_disc
                                            for nx2, ny2 in self.world.neighbours((int(pt[0]), int(pt[1])))
                                            if self.radio_shadow[int(nx2), int(ny2)]
                                        )
                                    ]
                                    _brc[brc_key] = any(robot._reachable_arr[x, y] for x, y in water_disc_border)
                            if not _brc[brc_key]: continue
                        # Prefer not to elect a stair-capable robot as relay for
                        # ANY shadow cluster if land-only robots can serve instead.
                        # Applies to both stair AND disc clusters — a Drone relaying
                        # a disc is wasteful when a Rover could do it instead.
                        can_enter = bool(robot.caps_mask & (CAP_STAIRS | CAP_AIR))
                        if can_enter:
                            free_explorers = sum(
                                1 for rr in active
                                if rr is not robot
                                and rr.role != Role.RELAY
                                and bool(rr.caps_mask & (CAP_STAIRS | CAP_AIR))
                            )
                            free_land_only = sum(
                                1 for rr in active
                                if bool(rr.caps_mask & CAP_LAND)
                                and not bool(rr.caps_mask & (CAP_STAIRS|CAP_AIR|CAP_WATER))
                                and rr.role != Role.RELAY
                            )
                            if free_explorers == 0:
                                pass   # no alternative — allow it
                            elif free_land_only > 0:
                                continue  # land-only robots available, keep stair/air robots exploring
                            elif free_explorers < 2:
                                continue  # spare the last stair-capable explorer
                        # Extra guard: skip if ANY currently-locked relay already
                        # owns a zone in this cluster — prevents double-election
                        # when role_locked_until keeps both robots in the skip path
                        # but the cluster_info covered flag wasn't set yet.
                        already_held = any(
                            rr.role == Role.RELAY and rr.task_zone in cl
                            for rr in active
                        )
                        if already_held: continue
                        rv    = self._relay_val(cl)
                        min_d = min(
                            abs(rx - (z[0]*self.zone_w_cells + self.zone_w_cells//2))
                            + abs(ry - (z[1]*self.zone_h_cells + self.zone_h_cells//2))
                            for z in cl
                        )
                        cl_key = frozenset(cl)
                        last_cov = getattr(self, '_cluster_last_covered_zones', {}).get(cl_key, 0)
                        ticks_unc = self.timestep - last_cov
                        # Hard override: if a building has never had relay coverage
                        # (last_cov=0 and ticks_unc > 150), force it to be chosen.
                        # Lowered from 500 — corner buildings should not wait 1/6 of
                        # the mission before getting a relay.
                        if last_cov == 0 and ticks_unc > 150:
                            net = rv + 10.0
                        else:
                            net = rv - (min_d / D_NORM) * self._RELAY_TRAVEL_W
                        if net > best_net:
                            best_net = net; best_cl = cl
                    if best_cl is not None:
                        # Pick the zone with the most shadow cells as task_zone.
                        # The old min(coverage) picked fringe/dilation zones with
                        # almost no shadow, meaning the relay sat at a border that
                        # covered barely anything. Highest-sf zone is the zone with
                        # most shadow to bridge — the relay anchor lands centrally.
                        robot.task_zone = max(
                            best_cl,
                            key=lambda z: self._zone_shadow_frac.get(z, 0.0))
                        # Clear bundle: relay robots don't explore zones, they sit
                        # at the shadow border. Keeping zones in the bundle blocks
                        # other robots from bidding on them indefinitely.
                        # Release all owned zone tasks back to free so CBBA
                        # can reassign them to active explorers next cycle.
                        for z in robot.bundle:
                            task = self.zone_tasks.get(z)
                            if task and robot.name in task.owners:
                                task.owners.remove(robot.name)
                                if not task.owners:
                                    task.status = 'free'; task.expires_at = 0
                        robot.bundle = []; robot.assigned_zones = []
                        # Mark cluster as now covered so subsequent robots in this
                        # iteration don't also elect themselves for the same cluster
                        for cid2, (cl2, _) in cluster_info.items():
                            if cl2 is best_cl:
                                cluster_info[cid2] = (cl2, True)
                                break
                        # Only lock when the relay is actually confirmed — if no
                        # uncovered cluster was found the robot reverts to old_role
                        # and must NOT be locked (that would freeze it as a useless
                        # LOITER/SCAN for 150 ticks while buildings go unexplored).
                        # Travel grace: a relay must stay committed long enough to
                        # actually REACH its target border before BR can re-elect it.
                        # A fixed 30-tick grace was far too short for distant
                        # buildings (100+ cells away on a large map): the lock
                        # expired mid-travel, the relay got demoted/re-elected as a
                        # different robot, and far buildings were never reached —
                        # the multi-relay churn that made raising the cap fail.
                        # Scale the grace with Manhattan distance to the target
                        # cluster (paths aren't straight, so ~1.8 ticks/cell + buffer).
                        # _move_relay can still demote a genuinely-unreachable relay
                        # directly, so an over-long lock is self-correcting.
                        zw = self.zone_w_cells; zh = self.zone_h_cells
                        best_d = min(
                            abs(robot.pos[0] - (z[0]*zw + zw//2))
                            + abs(robot.pos[1] - (z[1]*zh + zh//2))
                            for z in best_cl)
                        travel_grace = int(best_d * 1.8) + 30
                        robot.relay_hold_until   = t + travel_grace
                        robot.role_locked_until  = t + travel_grace
                        robot.relay_last_occupied = t
                        robot.relay_anchor       = None
                    else:
                        # No uncovered cluster to serve — don't become relay, no lock
                        robot.role = old_role
                        changed = (old_role != robot.role)

                elif old_role == Role.RELAY:
                    # Relay stepped down — verify no explorer is stranded first.
                    # Check the cluster this relay was serving; if any explorer is
                    # inside its shadow with no other relay covering them, block
                    # the demotion entirely — it would strand them with no comms.
                    old_task = robot.relay_anchor_zone or robot.task_zone
                    if old_task is not None:
                        old_cid = self._shadow_cluster_id.get(old_task, -1)
                        old_cz_set = {z for z, c in self._shadow_cluster_id.items() if c == old_cid}
                        # Would any explorer be uncovered after this demotion?
                        other_relay_covers = any(
                            rr.role == Role.RELAY and rr is not robot
                            and self._shadow_cluster_id.get(rr.task_zone, -2) == old_cid
                            for rr in self.robots if rr.active
                        )
                        if not other_relay_covers:
                            explorer_trapped = any(
                                rr.active and rr.role != Role.RELAY
                                and self.radio_shadow[rr.pos[0], rr.pos[1]]
                                and self.cell_to_zone(rr.pos[0], rr.pos[1]) in old_cz_set
                                for rr in self.robots
                            )
                            if explorer_trapped:
                                # Revert — cannot demote while explorers depend on this relay
                                robot.role = old_role
                                changed = False
                                continue
                    # Safe to demote — clear navigation state
                    robot.relay_anchor      = None
                    robot.relay_anchor_zone = None
                    robot.relay_hold_until  = 0
                    robot.role_locked_until = 0
                    for rr in self.robots:
                        rr._reachable_cache = None

            if not changed:
                break   # Nash equilibrium reached

    # ── relay anchor logic ────────────────────────────────────────────────────
    def _move_relay(self, r, occupied):
        """
        Move relay to the shadow border cell nearest to its task_zone centre,
        reachable without entering shadow.  Once there, hold position.

        SAFETY INVARIANT: if any non-relay explorer is inside this cluster's
        shadow, this relay must never demote — checked as an absolute hard gate
        before any other demotion logic.
        """
        if r.task_zone is None:
            return

        # ── Hard safety gate: explorer inside → hold unconditionally ─────────
        my_cid = self._shadow_cluster_id.get(r.task_zone, -1)
        if my_cid >= 0:
            cz_set_safety = {z for z, c in self._shadow_cluster_id.items() if c == my_cid}
            for rr in self.robots:
                if (rr.active and rr.role != Role.RELAY
                        and self.radio_shadow[rr.pos[0], rr.pos[1]]
                        and self.cell_to_zone(rr.pos[0], rr.pos[1]) in cz_set_safety):
                    # Explorer inside — extend demotion hold but NOT role_locked_until.
                    # role_locked_until would block BR from reassigning to other clusters.
                    if r.relay_hold_until < self.timestep + RELAY_MIN_HOLD:
                        r.relay_hold_until = self.timestep + RELAY_MIN_HOLD
                    r.relay_last_occupied = self.timestep
                    break

        W, H   = self.world.w, self.world.h
        shadow = self.radio_shadow
        tb_arr = r.terrain_belief
        mask4  = r.caps_mask & 0xF
        trav   = _TRAV_LUT

        # Shadow-border mask: outside shadow AND has a shadow-cell neighbour (cached)
        border_mask = self._shadow_border_mask_cache

        # Task zone centre (target area)
        zx, zy = r.task_zone
        cx = zx * self.zone_w_cells + self.zone_w_cells // 2
        cy = zy * self.zone_h_cells + self.zone_h_cells // 2

        # Demotion check — runs whenever relay is at border (not just while hold active).
        # When hold expires, the relay should evaluate whether to stay or leave.
        if border_mask[r.pos[0], r.pos[1]]:
            cluster_zones = [z for z, c in self._shadow_cluster_id.items()
                             if c == self._shadow_cluster_id.get(r.task_zone, -1)]
            cluster_uf = float(np.mean([self._zone_uf_cache.get(z,0.0) for z in cluster_zones])) if cluster_zones else 0.0
            cz_set = set(cluster_zones)
            # Single pass over robots for all three checks
            exp_inside = exp_approaching = exp_assigned = False
            for rr in self.robots:
                if not rr.active or rr.role == Role.RELAY: continue
                if self.radio_shadow[rr.pos[0], rr.pos[1]] and self.cell_to_zone(rr.pos[0], rr.pos[1]) in cz_set:
                    exp_inside = True
                if bool(rr.caps_mask & (CAP_STAIRS | CAP_AIR)):
                    if rr.task_zone in cz_set and abs(rr.pos[0]-cx)+abs(rr.pos[1]-cy) < 40:
                        exp_approaching = True
                    if (rr.task_zone in cz_set or any(z in cz_set for z in rr.bundle)
                            or any(rr.name in self.zone_tasks[z].owners
                                   for z in cz_set if z in self.zone_tasks)):
                        exp_assigned = True
            explorer_approaching = exp_approaching
            explorer_assigned = exp_assigned

            # Use stair-cell coverage for demotion decision — more accurate than
            # zone-level uf which includes surrounding open terrain.
            # A Rover relay should leave as soon as nobody needs to cross:
            # no explorer inside, none approaching, and either building is fully
            # done OR no capable robot is assigned to this cluster.
            is_rover_relay = (bool(r.caps_mask & CAP_LAND)
                              and not bool(r.caps_mask & (CAP_STAIRS|CAP_AIR|CAP_WATER)))

            # Stair cell coverage — cached per cluster per tick (called for every relay every step)
            _sc = getattr(self, '_relay_stair_cov_cache', None)
            if _sc is None or getattr(self, '_relay_stair_cov_tick', -1) != self.timestep:
                self._relay_stair_cov_cache = {}; self._relay_stair_cov_tick = self.timestep
                _sc = self._relay_stair_cov_cache
            my_cid = self._shadow_cluster_id.get(r.task_zone, -1)
            if my_cid not in _sc:
                cs_cells = cs_unk = 0
                for cz in cluster_zones:
                    czx,czy = cz
                    x0c=czx*self.zone_w_cells; x1c=min(x0c+self.zone_w_cells,self.world.w)
                    y0c=czy*self.zone_h_cells; y1c=min(y0c+self.zone_h_cells,self.world.h)
                    sm = self._world_stair_arr[x0c:x1c,y0c:y1c]
                    cs_cells += int(np.sum(sm))
                    cs_unk   += int(np.sum((self.union_belief[x0c:x1c,y0c:y1c]==T_UNKNOWN) & sm))
                _sc[my_cid] = (1.0 - cs_unk/cs_cells if cs_cells > 0 else 1.0)
            stair_cov = _sc[my_cid]
            # Endgame: hold a building until it is nearly fully swept, instead of
            # releasing the relay at 95%.  The moment a relay demotes, coverage
            # stops flooding the shadow, so any survivor in the last few percent
            # is stranded and never found (the "drone just missed the last
            # survivor in the building" case).  In the main phase keep 0.95 so a
            # relay doesn't linger on an essentially-done building while others
            # wait.  The idle-demotion below is the escape hatch: if the remaining
            # cells are genuinely unreachable/LOS-blocked, no explorer makes
            # progress for RELAY_IDLE_TICKS and the relay releases anyway.
            if getattr(self, '_union_cov_tick', -1) != self.timestep:
                self._union_cov_cache = float(np.mean(self.union_belief != T_UNKNOWN))
                self._union_cov_tick = self.timestep
            _done_thr = 0.99 if self._union_cov_cache > 0.90 else 0.95
            building_complete = stair_cov >= _done_thr

            # Track idle time: reset when an explorer is inside
            if exp_inside:
                r.relay_last_occupied = self.timestep
            idle_ticks = self.timestep - r.relay_last_occupied

            # A building still NEEDS its relay as long as unexplored stair cells
            # remain AND at least one stair-capable explorer is alive to reach it.
            # The old idle / stair_cov==0 demotions abandoned a relay on an
            # unexplored building before CBBA (which runs every 50 ticks and only
            # assigns explorers to relay-COVERED buildings) had a chance to route
            # an explorer in — a chicken-and-egg trap that left whole buildings
            # permanently unexplored.  We now refuse to abandon a building that
            # still has demand; only the building-complete and explorer-inside
            # gates apply there.  The safety valve: if NO stair-capable explorer
            # is alive at all, holding is pointless, so demand falls to False.
            any_stair_explorer = any(
                rr.active and rr.battery > 0 and rr.role != Role.RELAY
                and bool(rr.caps_mask & (CAP_STAIRS | CAP_AIR))
                for rr in self.robots
            )
            building_has_demand = (stair_cov < 0.90) and any_stair_explorer

            # Demote immediately if building is done — don't wait for hold expiry.
            # SAFETY: never demote if any explorer is currently inside shadow.
            should_demote = (
                not exp_inside                 # HARD GATE: explorer inside = never demote
                and not explorer_approaching
                and not building_has_demand    # never abandon an unexplored building
                and (
                    building_complete                               # done — leave now
                    or (idle_ticks >= RELAY_IDLE_TICKS             # long idle AND no assignment
                        and not explorer_assigned)
                    or (r.relay_hold_until <= self.timestep and (  # hold expired AND:
                        (is_rover_relay and not explorer_assigned)  # Rover with nobody assigned
                        or stair_cov == 0.0))                       # nobody ever entered
                )
            )
            if should_demote:
                r.role = Role.SCAN
                r.relay_hold_until = 0; r.relay_anchor = None
                r.role_locked_until = self.timestep + 60
                return

        # If already on a border cell that is adjacent to our own cluster — hold
        if border_mask[r.pos[0], r.pos[1]]:
            rx, ry = r.pos
            # Verify this border cell actually touches our task_zone cluster
            own_cluster_border = any(
                self.radio_shadow[nx, ny]
                and self._shadow_zone_type.get(self.cell_to_zone(nx, ny)) ==
                    self._shadow_zone_type.get(r.task_zone)
                and self._same_shadow_cluster(self.cell_to_zone(nx, ny), r.task_zone)
                for nx, ny in self.world.neighbours((rx, ry))
            )
            if own_cluster_border:
                # Refresh hold clock whenever relay is at a border cell that
                # is different from the stored anchor (moved to better position)
                # or when hold is running low. Use anchor to detect position changes.
                new_position = (r.relay_anchor is None or r.relay_anchor != r.pos)
                if new_position:
                    r.relay_hold_until  = self.timestep + RELAY_MIN_HOLD
                    r.role_locked_until = self.timestep + 30
                    r.relay_anchor      = r.pos
                    r.relay_anchor_zone = r.task_zone
                    # Trigger CBBA rebid once so explorers get assigned promptly
                    last_border_cbba = getattr(self, '_relay_border_cbba_tick', -999)
                    if self.timestep - last_border_cbba >= 10:
                        self._relay_border_cbba_tick = self.timestep
                        self._assign_zones_cbba()
                        self._relay_ok_prev = {}
                cluster_zones = set(
                    (zx2, zy2)
                    for zx2 in range(self.zone_nx)
                    for zy2 in range(self.zone_ny)
                    if self._same_shadow_cluster((zx2, zy2), r.task_zone)
                    and self._shadow_frac_for_zone((zx2, zy2)) > 0.05
                )
                explorers_inside = any(
                    rr.active and rr.role != Role.RELAY
                    and self.radio_shadow[rr.pos[0], rr.pos[1]]
                    and self.cell_to_zone(rr.pos[0], rr.pos[1]) in cluster_zones
                    for rr in self.robots
                )
                if explorers_inside and r.relay_hold_until < self.timestep + 20:
                    r.relay_hold_until = self.timestep + RELAY_MIN_HOLD
                    # Don't extend role_locked_until — let BR re-evaluate freely
                if r._reveal_all(): r._scan_dirty = True
                return
            # Wrong cluster border — only clear anchor after 3 consecutive ticks
            # to avoid 1-tick oscillation during final approach to correct border
            wrong_ticks = getattr(r, '_wrong_border_ticks', 0) + 1
            r._wrong_border_ticks = wrong_ticks
            if wrong_ticks >= 3:
                r.relay_anchor = None; r.relay_anchor_zone = None
                r._wrong_border_ticks = 0
        else:
            r._wrong_border_ticks = 0

        # Recompute anchor only if needed (zone changed or lost anchor)
        if r.relay_anchor_zone != r.task_zone or r.relay_anchor is None:
            zone_type = self._shadow_zone_type.get(r.task_zone, 'none')

            # Build candidate border cells: only those adjacent to our cluster's shadow
            # AND physically traversable by this robot type.
            # Boats can only stand on water/bridge — land border cells are excluded.
            # Land robots cannot stand on water — water border cells are excluded.
            is_boat_relay = bool(r.caps_mask & CAP_WATER) and not bool(r.caps_mask & CAP_AIR)
            cluster_border = []
            for pt in self._shadow_border_cells_arr:
                bx, by = int(pt[0]), int(pt[1])
                if (bx, by) in getattr(r, '_relay_anchor_blacklist', set()):
                    continue
                bt = self.world.grid[bx][by]["t"]
                if is_boat_relay:
                    if bt not in (T_WATER, T_BRIDGE):
                        continue   # boat cannot stand on land border cell
                else:
                    if bt == T_WATER:
                        continue   # land robot cannot stand on water border cell
                for nx, ny in self.world.neighbours((bx, by)):
                    if not self.radio_shadow[nx, ny]: continue
                    nz = self.cell_to_zone(nx, ny)
                    if (self._shadow_zone_type.get(nz) == zone_type
                            and self._same_shadow_cluster(nz, r.task_zone)):
                        cluster_border.append((bx, by))
                        break

            if not cluster_border:
                # Blacklist exhausted or no border exists — clear blacklist and retry
                r._relay_anchor_blacklist = set()
                r.role = Role.SCAN; r.relay_hold_until = 0; return

            # Prefer the border cell nearest the ROBOT that it can actually reach.
            # Coverage floods through the whole connected shadow cluster no matter
            # which border cell the relay occupies, so any reachable border gives
            # full coverage — there is no reason to march to the far side nearest
            # the zone centre (the old rule), which made relays crawl 80-100 cells.
            # If the reachability flood happens to flag none (e.g. stale flood mid
            # travel), fall back to the nearest border by Manhattan distance and let
            # the A*/blacklist machinery below sort out any genuinely unreachable
            # cell — never demote here, which previously caused relays to drop a
            # building and re-claim it in a loop.
            reachable_border = cluster_border
            r.reachable()
            if r._reachable_arr is not None:
                rb = [(bx, by) for (bx, by) in cluster_border if r._reachable_arr[bx, by]]
                if rb:
                    reachable_border = rb

            best_border = min(reachable_border,
                              key=lambda p: abs(p[0]-r.pos[0]) + abs(p[1]-r.pos[1]))
            r.relay_anchor = best_border
            r.relay_anchor_zone = r.task_zone

        target = r.relay_anchor

        # Plan shadow-free path to anchor.
        # Staggered: relay i only runs A* on tick (timestep % 5) == (robot_idx % 5)
        # so at most 1 relay calls A* per tick regardless of fleet size.
        # This eliminates the 5x A* spike when all relays elect simultaneously.
        at_goal = (r.pos == target)
        robot_idx = next((i for i, rr in enumerate(self.robots) if rr is r), 0)
        my_slot = (self.timestep % 5) == (robot_idx % 5)
        last_replan = getattr(r, '_relay_last_replan', -999)
        overdue = (self.timestep - last_replan) >= 10  # safety: force replan if very stale

        need_replan = False
        if not at_goal:
            if (r.goal != target or not r.path) and (my_slot or overdue):
                need_replan = True
            elif r.path and my_slot:
                for cell in r.path[:4]:
                    if shadow[cell[0], cell[1]]:
                        need_replan = True; break

        if need_replan:
            # Relay A*: use minimal config — relay just needs a shadow-free path,
            # no hazard avoidance, no traffic penalties, no info gain.
            # This is ~3x faster than the full explorer A* config.
            _zero_traffic = np.zeros((GRID_W, GRID_H), dtype=np.uint16)
            _zero_chunked = np.zeros((2, GRID_W//CHUNK_SIZE, GRID_H//CHUNK_SIZE), dtype=np.float32)
            path = AStar.search(
                start=r.pos, goal=target,
                caps_mask=r.caps_mask, terrain_u8=tb_arr,
                temp_f32=r.temp_belief, rad_f32=r.rad_belief,
                chunked_risk=_zero_chunked,
                temp_limit=9999.0, rad_limit=9999.0,
                radio_shadow=shadow, relay_ok_fn=lambda z: False,
                cell_to_zone_fn=self.cell_to_zone,
                global_cov=1.0,
                unk_pen=0.0, info_w=0.0, unk_prior=0.0,
                alpha_mult=1.0, beta_mult=1.0, soft_frac=1.0,
                traffic_u16=_zero_traffic, traffic_w=0.0,
                shadow_border=self._shadow_border_mask_cache,
            )
            if not path:
                # Can't reach this border cell — blacklist it so anchor selection
                # picks a different one next tick instead of freezing on the same cell
                if not hasattr(r, '_relay_anchor_blacklist'):
                    r._relay_anchor_blacklist = set()
                r._relay_anchor_blacklist.add(target)
                r.relay_anchor = None; r.relay_anchor_zone = None
                r.relay_failed_path_count += 1
                return
            r.goal = target; r.path = path; r.goal_commit = 20
            r._relay_last_replan = self.timestep
            r.relay_failed_path_count = 0   # successful replan, reset counter

        if r.path:
            next_cell = r.path[0]
            if shadow[next_cell[0], next_cell[1]]:
                r.relay_anchor = None; return  # anchor crept into shadow, replan
            # Hard terrain safety: relay must not walk into water/OBS either
            true_t = self.world.grid[next_cell[0]][next_cell[1]]["t"]
            if true_t == T_OBS:
                r.relay_anchor = None; r.path = []; return
            if true_t == T_WATER and not bool(r.caps_mask & (CAP_WATER|CAP_AIR)):
                r.terrain_belief[next_cell[0],next_cell[1]] = true_t
                r.known_mask[next_cell[0],next_cell[1]] = True
                r.relay_anchor = None; r.path = []; return
            # Reverse check: boat must not walk onto land
            if true_t in (T_FREE, T_STAIRS) and bool(r.caps_mask & CAP_WATER) and not bool(r.caps_mask & CAP_AIR):
                r.terrain_belief[next_cell[0],next_cell[1]] = true_t
                r.known_mask[next_cell[0],next_cell[1]] = True
                r.relay_anchor = None; r.path = []; return
            occupied.discard(r.pos); occupied.add(next_cell)
            r.pos = r.path.pop(0)
            if r._reveal_all(): r._scan_dirty = True
            drain = {"Legged": 1.0, "Drone": 2.0, "Boat": 2.0, "Rover": 0.4}
            r.battery -= drain.get(robot_type(r.name), 1.0) * 1.1

    # ── choose exploration goal ───────────────────────────────────────────────
    def _uncovered_shadow_mask(self):
        """Cell-level bool: shadow cells whose zone has no relay coverage.
        Cached per tick — derived from the per-zone _relay_ok_flood dict by
        expanding the zone grid to cell resolution."""
        if getattr(self, '_ucsm_tick', -1) == self.timestep:
            return self._ucsm_cache
        nx, ny = self.zone_nx, self.zone_ny
        zw, zh = self.zone_w_cells, self.zone_h_cells
        covered_z = np.zeros((nx, ny), dtype=bool)
        flood = self._relay_ok_flood
        for (zx, zy), ok in flood.items():
            if ok and 0 <= zx < nx and 0 <= zy < ny:
                covered_z[zx, zy] = True
        covered_cells = np.zeros((GRID_W, GRID_H), dtype=bool)
        ce = np.repeat(np.repeat(covered_z, zw, axis=0), zh, axis=1)
        covered_cells[:ce.shape[0], :ce.shape[1]] = ce
        self._ucsm_cache = self.radio_shadow & ~covered_cells
        self._ucsm_tick = self.timestep
        return self._ucsm_cache

    def _choose_goal(self, r) -> tuple | None:
        """
        Pick the best frontier for robot r given its task_zone.
        Scores by: info-gain (unknown neighbours revealed) + distance + chunk risk.

        If task_zone is a shadow zone without relay coverage, steer to the nearest
        non-shadow cell on the zone boundary and hold there (border loiter).
        This avoids the robot wandering mid-map while waiting for a relay to arrive.
        """
        union = self.union_belief
        r.reachable()  # ensure BFS run; _reachable_arr populated
        reach_arr = r._reachable_arr  # bool array for fast membership

        # ── Shadow border loiter: task zone is blocked, park as close as possible ──
        if (r.task_zone is not None
                and self._shadow_frac_for_zone(r.task_zone) > 0.2
                and not self._relay_ok_flood.get(r.task_zone, False)):
            zx, zy = r.task_zone
            # Zone centre
            cx = zx*self.zone_w_cells + self.zone_w_cells//2
            cy = zy*self.zone_h_cells + self.zone_h_cells//2
            # Find nearest reachable non-shadow free cell to zone centre
            # Search outward from zone centre — pick the closest reachable cell
            x0 = max(0, zx*self.zone_w_cells - 4)
            x1 = min(self.world.w, (zx+1)*self.zone_w_cells + 4)
            y0 = max(0, zy*self.zone_h_cells - 4)
            y1 = min(self.world.h, (zy+1)*self.zone_h_cells + 4)
            sub_u = union[x0:x1, y0:y1]
            ok = (~self.radio_shadow[x0:x1, y0:y1]
                  & (sub_u != T_OBS) & (sub_u != T_UNKNOWN))
            if reach_arr is not None:
                ok &= reach_arr[x0:x1, y0:y1]
            if np.any(ok):
                xs = np.arange(x0, x1)[:, None]; ys = np.arange(y0, y1)[None, :]
                dist = np.abs(xs - cx) + np.abs(ys - cy)
                dist = np.where(ok, dist, 1 << 30)
                fi = int(np.argmin(dist))
                bx, by = divmod(fi, ok.shape[1])
                return (x0 + bx, y0 + by)
            # No reachable cell near zone — fall through to normal goal selection

        # frontiers in assigned zone (preferred), then global
        if r.task_zone is not None:
            candidates = self.zone_frontiers_for(r, r.task_zone)

            # Special case: stair zone with relay active but entirely unknown in
            # local belief — no frontier exists because robot never entered.
            # Pick the nearest reachable cell inside the zone to bootstrap entry.
            if (not candidates
                    and self._shadow_zone_type.get(r.task_zone) == 'stair'
                    and self._relay_ok_flood.get(r.task_zone, False)
                    and bool(r.caps_mask & (CAP_STAIRS | CAP_AIR))):
                zx_e, zy_e = r.task_zone
                x0_e = zx_e*self.zone_w_cells
                x1_e = min(x0_e+self.zone_w_cells, GRID_W)
                y0_e = zy_e*self.zone_h_cells
                y1_e = min(y0_e+self.zone_h_cells, GRID_H)
                # Target: nearest shadow cell in zone that relay covers
                # Whole task_zone is relay-covered (checked above), so every
                # shadow cell in it qualifies — no per-cell zone lookup needed.
                sub_sh = self.radio_shadow[x0_e:x1_e, y0_e:y1_e]
                ok_e = sub_sh.copy()
                if reach_arr is not None:
                    ok_e &= reach_arr[x0_e:x1_e, y0_e:y1_e]
                if np.any(ok_e):
                    xs = np.arange(x0_e, x1_e)[:, None]; ys = np.arange(y0_e, y1_e)[None, :]
                    dist = np.abs(xs - r.pos[0]) + np.abs(ys - r.pos[1])
                    dist = np.where(ok_e, dist, 1 << 30)
                    fi = int(np.argmin(dist))
                    bx, by = divmod(fi, ok_e.shape[1])
                    candidates = [(x0_e + bx, y0_e + by)]
        else:
            candidates = []

        if not candidates and reach_arr is not None:
            # global frontiers from reachable array
            is_boat = bool(r.caps_mask & CAP_WATER) and not bool(r.caps_mask & CAP_AIR)
            # When the task is a relay-open stair zone, restrict global candidates to
            # shadow cells only — open-terrain unknowns elsewhere compete via scoring
            # and cause goal-switching between the new building and distant open cells.
            stair_task = (r.task_zone is not None
                          and self._shadow_zone_type.get(r.task_zone) == 'stair'
                          and self._relay_ok_flood.get(r.task_zone, False)
                          and bool(r.caps_mask & (CAP_STAIRS | CAP_AIR)))
            uncov = self._uncovered_shadow_mask()
            base = reach_arr & ~uncov
            if stair_task:
                base &= self.radio_shadow
            if is_boat:
                wb = (union == T_WATER) | (union == T_BRIDGE)
                unkm = (union == T_UNKNOWN)
                nbr_unk = np.zeros((GRID_W, GRID_H), dtype=bool)
                nbr_unk[1:, :]  |= unkm[:-1, :]; nbr_unk[:-1, :] |= unkm[1:, :]
                nbr_unk[:, 1:]  |= unkm[:, :-1]; nbr_unk[:, :-1] |= unkm[:, 1:]
                cand_mask = base & wb & nbr_unk
            else:
                cand_mask = base & (union == T_UNKNOWN)
            candidates = [(int(p[0]), int(p[1])) for p in np.argwhere(cand_mask)]

        if not candidates and reach_arr is not None:
            # last resort: reveal-frontiers (known cells adjacent to unknown)
            uncov = self._uncovered_shadow_mask()
            unkm = (union == T_UNKNOWN)
            nbr_unk = np.zeros((GRID_W, GRID_H), dtype=bool)
            nbr_unk[1:, :]  |= unkm[:-1, :]; nbr_unk[:-1, :] |= unkm[1:, :]
            nbr_unk[:, 1:]  |= unkm[:, :-1]; nbr_unk[:, :-1] |= unkm[:, 1:]
            cand_mask = reach_arr & ~uncov & (union != T_UNKNOWN) & nbr_unk
            candidates = [(int(p[0]), int(p[1])) for p in np.argwhere(cand_mask)]

        if not candidates: return None

        # filter failed goals
        candidates = [c for c in candidates if self.timestep >= r.failed_goals.get(c, 0)]
        if not candidates: return None

        if len(candidates) == 1:
            return candidates[0]

        # Vectorised scoring — build arrays once, avoid per-candidate Python loops
        cxy = np.array(candidates, dtype=np.int32)   # (N, 2)
        cx_a, cy_a = cxy[:, 0], cxy[:, 1]
        N = len(candidates)

        # Distance
        dist = np.abs(cx_a - r.pos[0]) + np.abs(cy_a - r.pos[1])

        # Info gain: unknown neighbours (fast via convolution-style shift sums)
        # For small candidate sets, direct lookup is fine
        unknown_mask = (union == T_UNKNOWN)
        # Count unknown 4-neighbours for each candidate
        info = np.zeros(N, dtype=np.float32)
        for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
            nx2 = np.clip(cx_a+dx, 0, GRID_W-1); ny2 = np.clip(cy_a+dy, 0, GRID_H-1)
            info += unknown_mask[nx2, ny2].astype(np.float32)
        # Bonus for landing on unknown cell itself
        info += (union[cx_a, cy_a] == T_UNKNOWN).astype(np.float32) * 4.0

        # Risk
        chunk_x = cx_a // CHUNK_SIZE; chunk_y = cy_a // CHUNK_SIZE
        lT = r.chunked[0, chunk_x, chunk_y].astype(np.float32)
        lR = r.chunked[1, chunk_x, chunk_y].astype(np.float32)
        eT = lT / max(1e-6, r.temp_limit)
        eR = lR / max(1e-6, r.rad_limit)
        e  = np.maximum(eT, eR)
        risk_sign = -1.0 if r.name.startswith("Rover") else 1.0
        risk_term = risk_sign * ALPHA * (e ** P)

        # Crowd penalty: stronger for same-locomotion type robots.
        # Checks both committed goals (rr.goal) AND this-tick reservations
        # so robots picking goals in the same loop iteration diverge immediately.
        crowd = np.zeros(N, dtype=np.float32)
        my_caps = r.caps_mask
        reservations = getattr(self, '_goal_reservations', {})
        for rr in self.robots:
            if rr is r or not rr.active or rr.goal is None: continue
            gx2, gy2 = rr.goal
            close = (np.abs(cx_a - gx2) + np.abs(cy_a - gy2) < 10).astype(np.float32)
            if rr.caps_mask == my_caps:
                crowd += close * 3.0   # same locomotion: heavy repulsion
            else:
                crowd += close * 1.0   # different type: light repulsion
        # Also penalise cells reserved this tick by same-caps robots
        for res_pos, res_caps in reservations.items():
            gx2, gy2 = res_pos
            close = (np.abs(cx_a - gx2) + np.abs(cy_a - gy2) < 10).astype(np.float32)
            if res_caps == my_caps:
                crowd += close * 4.0   # same-tick same-type: strongest repulsion
            else:
                crowd += close * 1.5

        # Shadow-pull bonus: when relay is active for a stair zone and this robot
        # can enter stairs, strongly prefer shadow-interior cells over edge frontiers.
        shadow_pull = np.zeros(N, dtype=np.float32)
        if (r.task_zone is not None
                and self._shadow_zone_type.get(r.task_zone) == 'stair'
                and self._relay_ok_flood.get(r.task_zone, False)
                and bool(r.caps_mask & (CAP_STAIRS | CAP_AIR))):
            zone_cov = self.zone_coverage(self.union_belief, r.task_zone)
            robot_inside = self.world.grid[r.pos[0]][r.pos[1]]["t"] == T_STAIRS

            # How much has this robot personally scanned in this zone?
            zx2, zy2 = r.task_zone
            zx0 = zx2*self.zone_w_cells; zx1 = min(zx0+self.zone_w_cells, GRID_W)
            zy0 = zy2*self.zone_h_cells; zy1 = min(zy0+self.zone_h_cells, GRID_H)
            personal_frac = float(np.mean(r.personally_scanned[zx0:zx1, zy0:zy1]))

            # Pull strongly into shadow while zone is substantially unexplored.
            # Use SHADOW_ZONE_DONE as the threshold (not -0.10) so the pull
            # persists right up to the completion threshold — previously the
            # pull dropped off at 85% causing premature exits at 85-95%.
            if zone_cov < SHADOW_ZONE_DONE:
                shadow_pull = self.radio_shadow[cx_a, cy_a].astype(np.float32) * 90.0

            elif robot_inside:
                # Zone is nominally done but robot is inside — check if there are
                # genuinely unknown stair cells remaining in this robot's view.
                ub_sl  = union[zx0:zx1, zy0:zy1]
                unk_sl = (ub_sl == T_UNKNOWN)

                wt_sl = self._world_terrain_arr[zx0:zx1, zy0:zy1] \
                        if hasattr(self, '_world_terrain_arr') else None
                if wt_sl is None:
                    self._world_terrain_arr = np.array(
                        [[self.world.grid[x][y]["t"] for y in range(GRID_H)]
                         for x in range(GRID_W)], dtype=np.uint8)
                    wt_sl = self._world_terrain_arr[zx0:zx1, zy0:zy1]

                stair_sl  = (wt_sl == T_STAIRS)
                unk_stair = unk_sl & stair_sl

                wall_sl = (wt_sl == T_OBS) | (wt_sl == T_STAIRS)
                W2, H2 = wall_sl.shape
                adj_wall = np.zeros((W2, H2), dtype=bool)
                adj_wall[:-1, :] |= wall_sl[1:, :]
                adj_wall[1:,  :] |= wall_sl[:-1, :]
                adj_wall[:, :-1] |= wall_sl[:, 1:]
                adj_wall[:, 1:]  |= wall_sl[:, :-1]
                wall_adj_mask = unk_sl & adj_wall

                if np.any(unk_stair):
                    cx_local = cx_a - zx0; cy_local = cy_a - zy0
                    in_zone = ((cx_local >= 0) & (cx_local < zx1-zx0) &
                               (cy_local >= 0) & (cy_local < zy1-zy0))
                    shadow_pull = np.zeros(N, dtype=np.float32)
                    wall_hit  = in_zone & wall_adj_mask[cx_local.clip(0,zx1-zx0-1), cy_local.clip(0,zy1-zy0-1)]
                    stair_hit = in_zone & unk_stair[cx_local.clip(0,zx1-zx0-1), cy_local.clip(0,zy1-zy0-1)]
                    shadow_pull[stair_hit] = 70.0
                    shadow_pull[wall_hit]  = 100.0
                else:
                    # Robot has personally seen all stair cells — safe to exit
                    shadow_pull = -self.radio_shadow[cx_a, cy_a].astype(np.float32) * 40.0
            else:
                # Robot outside and zone is complete — bias toward exit
                shadow_pull = -self.radio_shadow[cx_a, cy_a].astype(np.float32) * 40.0

        # Inside a building, distance matters much less — every cell is guaranteed
        # stair terrain and the robot should commit to full interior coverage rather
        # than being pulled back toward the door by nearby open-terrain candidates.
        # Halve the distance weight when the robot is inside a stair zone.
        robot_in_stair = (self.world.grid[r.pos[0]][r.pos[1]]["t"] == T_STAIRS)
        dist_weight = 0.4 if robot_in_stair else 1.0

        scores = dist.astype(np.float32) * dist_weight - 2.5*info + risk_term + 10.0*crowd - shadow_pull

        # Late-game goal hysteresis (anti-churn): bias toward the robot's CURRENT
        # goal so sparse, far frontier stops causing target-switching (flip-flop).
        # Scales with coverage — negligible early, strong near completion.  This
        # only stabilises WHICH cell an already-tasked robot heads to; it never
        # changes which robot is suited to a task (capability + zone assignment
        # are decided upstream by the role game and CBBA), so it cannot make a
        # wrong locomotion type take work it shouldn't, nor silence a useful one.
        if r.goal is not None:
            cov = getattr(self, 'global_cov', 0.0)
            stick = 6.0 + 30.0 * max(0.0, cov - 0.85) / 0.15   # ~6 early → ~36 near 100%
            near_old = (np.abs(cx_a - r.goal[0]) + np.abs(cy_a - r.goal[1]) <= 4).astype(np.float32)
            scores = scores - stick * near_old

        return candidates[int(np.argmin(scores))]

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

        # ── CBBA reassignment ──
        # Run every 50 ticks normally, every 20 ticks when coverage is high
        # (late game: many zones done, need fast reassignment of remaining ones)
        # Cache union coverage — computed once per tick, used for CBBA cadence
        # and LOITER rescue. np.mean over 16k cells is expensive per step.
        if not hasattr(self, '_union_cov_cache') or self._union_cov_tick != self.timestep:
            self._union_cov_cache = float(np.mean(self.union_belief != T_UNKNOWN))
            self._union_cov_tick  = self.timestep
        union_cov = self._union_cov_cache
        cbba_cadence = 20 if union_cov > 0.70 else 50
        needs_cbba = (self.timestep % cbba_cadence == 0)

        # LOITER rescue: any robot that has been LOITER for >60 ticks gets
        # its bundle cleared — prevents permanent loiter lock.
        # Only triggers a CBBA rebid once per tick (not per robot).
        for r in self.robots:
            if not r.active: continue
            if r.role == Role.LOITER:
                if not hasattr(r, '_loiter_since'):
                    r._loiter_since = self.timestep
                elif self.timestep - r._loiter_since > 60:
                    loiter_zone = r.task_zone
                    r.bundle = []; r.assigned_zones = []
                    r.task_zone = None
                    if loiter_zone is not None:
                        r.blacklist[loiter_zone] = self.timestep + 300
                    r._loiter_since = self.timestep
                    needs_cbba = True
            else:
                r._loiter_since = self.timestep

        # ── Relay handoff: release stair-capable relays when Rovers are available ──
        # If a Legged or Drone is locked as relay but land-only robots are free
        # and the cluster is still unexplored (relay elected before Rovers arrived),
        # shorten the lock so the potential game can reassign next tick.
        # Only fires every 30 ticks to avoid thrash.
        if self.timestep % 30 == 0:
            active_now = [r for r in self.robots if r.active and r.battery > 0]
            free_land_only = sum(
                1 for r in active_now
                if bool(r.caps_mask & CAP_LAND)
                and not bool(r.caps_mask & (CAP_STAIRS|CAP_AIR|CAP_WATER))
                and r.role != Role.RELAY
            )
            if free_land_only >= 1:
                for r in active_now:
                    if r.role != Role.RELAY: continue
                    if not bool(r.caps_mask & (CAP_STAIRS|CAP_AIR)): continue
                    if r.task_zone is None: continue
                    cz = [z for z, c in self._shadow_cluster_id.items()
                          if c == self._shadow_cluster_id.get(r.task_zone, -1)]
                    cluster_uf = float(np.mean([self.zone_stats(z)['unknown_frac']
                                                for z in cz])) if cz else 0.0
                    exp_inside = any(
                        rr.active and rr.role != Role.RELAY
                        and self.radio_shadow[rr.pos[0], rr.pos[1]]
                        and self.cell_to_zone(rr.pos[0], rr.pos[1]) in set(cz)
                        for rr in self.robots
                    )
                    if cluster_uf > 0.7 and not exp_inside:
                        r.role_locked_until = min(r.role_locked_until, self.timestep + 5)
                        needs_cbba = True

        if needs_cbba:
            self._assign_zones_cbba()

        # ── rebuild per-tick caches ──
        self._frontier_cache.clear()
        self._reachable_by_mask.clear()
        self._reservations.clear()   # fresh reservation table each tick

        # ── traffic map ──
        self.traffic_u16.fill(0)
        for r in self.robots:
            if r.active and r.path:
                x0,y0 = r.pos
                self.traffic_u16[x0,y0] = min(65535, int(self.traffic_u16[x0,y0])+2)
                for px,py in r.path[:TRAFFIC_LOOKAHEAD]:
                    self.traffic_u16[px,py] = min(65535, int(self.traffic_u16[px,py])+1)
        # Relay robots that are still travelling (not yet at shadow border) act
        # as moving obstacles. Mark their position as high-traffic so explorers
        # route around them en-route. However, a relay HOLDING at the shadow
        # border must NOT be penalised — explorers need to pass close to it to
        # enter the building, and a +50 cost detours them around the whole building.
        for r in self.robots:
            if not r.active or r.role != Role.RELAY: continue
            rx, ry = r.pos
            # Only penalise if not already at shadow border
            if not self._shadow_border_mask_cache[rx, ry]:
                self.traffic_u16[rx, ry] = min(65535, int(self.traffic_u16[rx,ry]) + 200)

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
            deliver_at = t + delay
            # Wrap as (deliver_at, msg_tuple) — avoids mutating the tuple
            for msg in r.outbox:
                self._pending_msgs.append((deliver_at, msg))
            r.outbox.clear()

        # Phase 5: deliver matured messages to all comms-capable robots
        still_pending = []
        deliverable = []
        for item in self._pending_msgs:
            if t >= item[0]:
                deliverable.append(item[1])
            else:
                still_pending.append(item)
        self._pending_msgs = still_pending

        # Deduplicate: keep only freshest per (x, y) cell.
        # Tuple format: (x, y, terrain, temp, rad, ts)
        if deliverable:
            dedup = {}
            for msg in deliverable:
                key = (msg[0], msg[1])
                if key not in dedup or msg[5] > dedup[key][5]:
                    dedup[key] = msg
            deliverable = list(dedup.values())

        # Build fleet-shared update arrays — one write, N reads instead of N×N inbox ops.
        # Each comms-capable robot merges from these arrays in _process_inbox.
        if deliverable:
            n = len(deliverable)
            _fleet_xs   = np.empty(n, dtype=np.int16)
            _fleet_ys   = np.empty(n, dtype=np.int16)
            _fleet_terr = np.empty(n, dtype=np.uint8)
            _fleet_temp = np.empty(n, dtype=np.float32)
            _fleet_rad  = np.empty(n, dtype=np.float32)
            _fleet_ts   = np.empty(n, dtype=np.int32)
            for i, m in enumerate(deliverable):
                _fleet_xs[i]=m[0]; _fleet_ys[i]=m[1]; _fleet_terr[i]=m[2]
                _fleet_temp[i]=m[3]; _fleet_rad[i]=m[4]; _fleet_ts[i]=m[5]
            self._fleet_update = (_fleet_xs, _fleet_ys, _fleet_terr, _fleet_temp, _fleet_rad, _fleet_ts)
        else:
            self._fleet_update = None

        # Phase 6: each robot merges fleet update directly (no per-robot inbox copy)
        for r in active_robots:
            if self._fleet_update is not None and comms_ok(r):
                r._merge_fleet_update(self._fleet_update, t)
            r.inbox.clear()  # clear any legacy inbox entries
        # Recompute chunked hazard maps once per robot per tick (deferred from move_step)
        for r in active_robots:
            if r._inbox_dirty or r._scan_dirty:
                # Throttle: chunked hazard map only needs updating every 3 ticks.
                # It's used by A* for risk avoidance — slight staleness is fine.
                last_chunk = getattr(r, '_last_chunked_tick', -999)
                if self.timestep - last_chunk >= 3:
                    r._recompute_chunked()
                    r._last_chunked_tick = self.timestep
                r._inbox_dirty = False
                r._scan_dirty  = False

        # ── rebuild union belief ──────────────────────────────────────────────
        self.union_belief = self._union_terrain()
        self.union_T      = self._union_temp()
        self.union_R      = self._union_rad()

        # ── per-tick zone unknown-frac cache ──────────────────────────────────
        # Single vectorised reshape — replaces nested Python loop over 64 zones.
        ub = self.union_belief
        zw, zh = self.zone_w_cells, self.zone_h_cells
        nx, ny = self.zone_nx, self.zone_ny
        ub_crop = ub[:nx*zw, :ny*zh]   # exact fit — no padding needed when grid divides evenly
        # Reshape to (nx, zw, ny, zh) then count unknowns per zone block
        unk_counts = (ub_crop.reshape(nx, zw, ny, zh) == T_UNKNOWN).sum(axis=(1, 3))
        total_cells = zw * zh
        zuf = {}
        for zx in range(nx):
            for zy in range(ny):
                zuf[(zx, zy)] = float(unk_counts[zx, zy]) / total_cells
        # Handle edge zones if grid doesn't divide evenly (rare)
        if nx*zw < GRID_W or ny*zh < GRID_H:
            for zx in range(nx):
                x0=zx*zw; x1=min(x0+zw,GRID_W)
                for zy in range(ny):
                    y0=zy*zh; y1=min(y0+zh,GRID_H)
                    slc=ub[x0:x1,y0:y1]
                    zuf[(zx,zy)]=float(np.count_nonzero(slc==T_UNKNOWN))/slc.size
        self._zone_uf_cache = zuf
        self._zone_uf_cache_tick = self.timestep
        # Detection radius matches the sensor scan disc exactly (reveal_R).
        # Line-of-sight is checked using Bresenham's ray from robot to survivor —
        # walls (T_OBS) and building interiors (T_STAIRS from outside) block detection.
        # ── survivor detection ────────────────────────────────────────────────
        # Uses the same radius (terrain_R) and LOS check (_has_los) as _reveal_all,
        # so a survivor is detectable if and only if the robot could also see the
        # terrain at that cell — i.e. within sensor range and not behind a wall.
        for r in self.robots:
            if not r.active: continue
            R = r.terrain_R
            rx, ry = r.pos
            r_inside_building = self.world.grid[rx][ry]["t"] == T_STAIRS
            for s in self.survivors:
                if s in self.found: continue
                sx, sy = s
                if (rx-sx)**2 + (ry-sy)**2 > R*R: continue
                if self._has_los(rx, ry, sx, sy, r_inside_building):
                    self.found.add(s)

        # ── role decisions ──
        # Run every tick — relay demotion safety is handled inside _move_relay
        # (hard safety gate: never demote when explorer is inside shadow) and
        # in the BR step-down guard (check for trapped explorers before demoting).
        # Skipping _decide_roles entirely when explorers are inside covered shadow
        # prevents new relays from being elected for OTHER unserved clusters.
        self._decide_roles()

        # ── occupation set (collision reservation) ──
        occupied = {r.pos for r in self.robots if r.active and r.battery>0}

        # ── PHASE 1: move relays first, then update relay_ok ──
        for r in self.robots:
            if not r.active: continue
            if r.role == Role.RELAY:
                self._move_relay(r, occupied)

        # update relay_ok AFTER relays have settled.
        # Cell-level BFS from each relay's position through shadow cells —
        # marks a zone covered only if the relay's signal can actually reach
        # shadow cells within it. Cached: only recompute when relay positions change.
        relay_signature = tuple(
            (r.name, r.pos, r.role.name, r.task_zone)
            for r in self.robots if r.active and r.role == Role.RELAY
        )
        if not hasattr(self, '_relay_ok_sig') or self._relay_ok_sig != relay_signature:
            self._relay_ok_sig = relay_signature
            self.relay_ok = {(zx,zy): False
                             for zx in range(self.zone_nx) for zy in range(self.zone_ny)}
            W, H = self.world.w, self.world.h
            for r in self.robots:
                if not r.active or r.role != Role.RELAY: continue
                if r.task_zone is None: continue
                rx, ry = r.pos
                if self.radio_shadow[rx, ry]: continue
                zone_type = self._shadow_zone_type.get(r.task_zone, 'none')
                if zone_type == 'none': continue
                cluster_id = self._shadow_cluster_id.get(r.task_zone)
                if cluster_id is None: continue

                shadow_queue = deque()
                shadow_visited = set()
                for nx, ny in self.world.neighbours((rx, ry)):
                    if not self.radio_shadow[nx, ny]: continue
                    nz = self.cell_to_zone(nx, ny)
                    if nz is None: continue
                    if self._shadow_zone_type.get(nz, 'none') != zone_type: continue
                    if self._shadow_cluster_id.get(nz) != cluster_id: continue
                    if (nx, ny) not in shadow_visited:
                        shadow_visited.add((nx, ny))
                        shadow_queue.append((nx, ny))
                        self.relay_ok[nz] = True

                while shadow_queue:
                    cx2, cy2 = shadow_queue.popleft()
                    for dx2, dy2 in NBR4:
                        nx2, ny2 = cx2+dx2, cy2+dy2
                        if not (0 <= nx2 < W and 0 <= ny2 < H): continue
                        if (nx2, ny2) in shadow_visited: continue
                        if not self.radio_shadow[nx2, ny2]: continue
                        nz2 = self.cell_to_zone(nx2, ny2)
                        if nz2 is None: continue
                        if self._shadow_zone_type.get(nz2, 'none') != zone_type: continue
                        if self._shadow_cluster_id.get(nz2) != cluster_id: continue
                        shadow_visited.add((nx2, ny2))
                        shadow_queue.append((nx2, ny2))
                        self.relay_ok[nz2] = True

            self._compute_relay_flood()

        # Invalidate reachable caches only when relay_ok actually changed.
        if self._relay_ok_flood != self._relay_ok_prev:
            for r in self.robots:
                r._reachable_cache = None
                if r.active and r.role != Role.RELAY:
                    # Clear path if goal or any near waypoint is in uncovered shadow
                    stale = False
                    if r.goal is not None:
                        gx, gy = r.goal
                        if (self.radio_shadow[gx, gy]
                                and not self._relay_ok_flood.get(self.cell_to_zone(gx, gy), False)):
                            stale = True
                    if not stale:
                        for wx, wy in r.path[:5]:
                            if (self.radio_shadow[wx, wy]
                                    and not self._relay_ok_flood.get(self.cell_to_zone(wx, wy), False)):
                                stale = True; break
                    if stale:
                        r.path = []; r.goal = None; r.goal_commit = 0

            # ── Relay-open event: immediately rebid CBBA so capable robots
            # can claim newly-unlocked stair zones without waiting up to 50 ticks.
            # Only trigger when a stair zone newly became available (not on drops).
            newly_open_stair = any(
                self._relay_ok_flood.get(z, False)
                and not self._relay_ok_prev.get(z, False)
                and self._shadow_zone_type.get(z) == 'stair'
                for z in self._relay_ok_flood
            )
            if newly_open_stair:
                # Release capable robots from non-stair bundles so they can rebid
                # for the newly open building. Only robots not already heading
                # somewhere productive (goal_commit expired or no path).
                for r in self.robots:
                    if not r.active or r.role == Role.RELAY: continue
                    if not bool(r.caps_mask & (CAP_STAIRS | CAP_AIR)): continue
                    if self.radio_shadow[r.pos[0], r.pos[1]]: continue
                    if r.goal and self.radio_shadow[r.goal[0], r.goal[1]]: continue
                    # Release current task_zone if it isn't a stair zone
                    if (r.task_zone is not None
                            and self._shadow_zone_type.get(r.task_zone) != 'stair'):
                        self._release_zone(r, "relay_opened_stair")
                    r.bundle = [z for z in r.bundle
                                if self._shadow_zone_type.get(z) == 'stair']
                self._assign_zones_cbba()

            self._relay_ok_prev = dict(self._relay_ok_flood)

        # Track per-cluster relay coverage time for the relay age bonus
        if not hasattr(self, '_cid_zone_frozensets'):
            cid_zones = {}
            for z2 in self._shadow_zone_type:
                c2 = self._shadow_cluster_id.get(z2)
                if c2 is not None:
                    cid_zones.setdefault(c2, []).append(z2)
            self._cid_zone_frozensets = {c: frozenset(zs) for c, zs in cid_zones.items()}
        if not hasattr(self, '_cluster_last_covered_zones'):
            self._cluster_last_covered_zones = {}
        for z, ok in self._relay_ok_flood.items():
            if ok:
                cid = self._shadow_cluster_id.get(z)
                if cid is not None:
                    cl_key = self._cid_zone_frozensets.get(cid)
                    if cl_key:
                        self._cluster_last_covered_zones[cl_key] = self.timestep

        # ── Revive: robots that lost comms come back if relay now covers their zone ──
        for r in self.robots:
            if r.active: continue
            if r.hazard_killed: continue          # permanent — no revive from temp/rad
            if r.death_reason != "lost comms — relay dropped": continue
            z = self.cell_to_zone(r.pos[0], r.pos[1])
            if self.relay_ok_extended(z):
                r.active       = True
                r.death_reason = None
                r.path         = []
                r.goal         = None
                r.goal_commit  = 0
                self.dead_robots = [(n,d) for n,d in self.dead_robots if n != r.name]

        # ── PHASE 2: task management + movement for non-relay robots ──
        # Safety sweep: any explorer in uncovered shadow triggers an emergency.
        # Rather than killing the explorer, we force-elect the nearest eligible
        # robot as relay for their cluster so they stay covered.
        shadow_at_risk = [
            r for r in self.robots if r.active and r.role != Role.RELAY
            and r.battery > 0
            and self.radio_shadow[r.pos[0], r.pos[1]]
            and not self.relay_ok_extended(self.cell_to_zone(r.pos[0], r.pos[1]))
        ]
        if shadow_at_risk:
            # Group victims by cluster and find the best available relay per cluster
            victim_clusters = {}
            for vr in shadow_at_risk:
                z = self.cell_to_zone(vr.pos[0], vr.pos[1])
                if z is None: continue
                cid = self._shadow_cluster_id.get(z, -1)
                if cid >= 0:
                    victim_clusters.setdefault(cid, []).append(vr)

            for cid, victims in victim_clusters.items():
                # Already have a relay for this cluster? Just update coverage.
                has_relay = any(
                    r.role == Role.RELAY
                    and self._shadow_cluster_id.get(r.task_zone, -1) == cid
                    for r in self.robots if r.active
                )
                if not has_relay:
                    # Find nearest non-relay robot that can reach the shadow border
                    cluster_zones = [z for z, c in self._shadow_cluster_id.items() if c == cid]
                    cx = int(sum(z[0]*self.zone_w_cells + self.zone_w_cells//2 for z in cluster_zones) / max(1,len(cluster_zones)))
                    cy = int(sum(z[1]*self.zone_h_cells + self.zone_h_cells//2 for z in cluster_zones) / max(1,len(cluster_zones)))
                    best_r = None; best_d = 1e9
                    for rr in self.robots:
                        if not rr.active or rr.battery <= 0: continue
                        if rr.role == Role.RELAY: continue
                        if self.radio_shadow[rr.pos[0], rr.pos[1]]: continue  # can't relay from inside
                        d = abs(rr.pos[0]-cx)+abs(rr.pos[1]-cy)
                        if d < best_d:
                            best_d = d; best_r = rr
                    if best_r is not None:
                        # Force-elect as relay for this cluster
                        for z in best_r.bundle:
                            task = self.zone_tasks.get(z)
                            if task and best_r.name in task.owners:
                                task.owners.remove(best_r.name)
                                if not task.owners: task.status='free'; task.expires_at=0
                        best_r.bundle = []; best_r.assigned_zones = []
                        best_r.role = Role.RELAY
                        best_r.task_zone = cluster_zones[0] if cluster_zones else None
                        best_r.relay_hold_until = self.timestep + RELAY_MIN_HOLD
                        best_r.role_locked_until = self.timestep + RELAY_MIN_HOLD
                        best_r.relay_last_occupied = self.timestep

            # Re-run relay moves and recompute coverage
            self._decide_roles()
            for r in self.robots:
                if r.active and r.role == Role.RELAY:
                    self._move_relay(r, occupied)
            # Full relay_ok recompute
            self.relay_ok = {(zx,zy): False
                             for zx in range(self.zone_nx) for zy in range(self.zone_ny)}
            for r2 in self.robots:
                if not r2.active or r2.role != Role.RELAY or r2.task_zone is None: continue
                r2x, r2y = r2.pos
                if self.radio_shadow[r2x, r2y]: continue
                cid2 = self._shadow_cluster_id.get(r2.task_zone)
                if cid2 is None: continue
                touching = any(
                    self.radio_shadow[nx2, ny2]
                    and self._shadow_cluster_id.get(self.cell_to_zone(nx2, ny2)) == cid2
                    for nx2, ny2 in self.world.neighbours((r2x, r2y))
                )
                if touching:
                    for z2, c2 in self._shadow_cluster_id.items():
                        if c2 == cid2: self.relay_ok[z2] = True
            self._compute_relay_flood()

        # Per-tick goal reservations: when robot A picks a goal this tick,
        # subsequent robots in the same loop see it and avoid it.
        # This prevents two robots simultaneously choosing the same frontier
        # which is the primary cause of SCAN robot flip-flopping.
        self._goal_reservations = {}  # pos -> caps_mask of reserving robot

        for r in self.robots:
            if not r.active: continue
            if r.role == Role.RELAY: continue
            if r.battery <= 0:
                if r.active:
                    r.active = False; r.death_reason = "battery depleted"
                    self.dead_robots.append((r.name, r.death_reason))
                continue

            self._manage_task_zone(r)
            self._move_robot(r, occupied)

        if len(self.found) == len(self.survivors):
            return False

        # Update zone age: mark zones where active robots currently are
        if hasattr(self, '_zone_last_visited'):
            for r in self.robots:
                if not r.active: continue
                z = self.cell_to_zone(r.pos[0], r.pos[1])
                if z: self._zone_last_visited[z] = self.timestep

        alive = any(r.active and r.battery>0 for r in self.robots)
        return alive

    def _manage_task_zone(self, r):
        """Select and maintain a task zone for robot r."""
        lease_active = (r.task_zone is not None and self.timestep < r.zone_lease_until)

        # ── Preemption: relay-open stair zone beats any non-stair task ──────
        # Throttled to every 10 ticks — frontier checks are expensive.
        if (r.task_zone is not None
                and bool(r.caps_mask & (CAP_STAIRS | CAP_AIR))
                and not self.radio_shadow[r.pos[0], r.pos[1]]):
            current_is_open_stair = (
                self._shadow_zone_type.get(r.task_zone) == 'stair'
                and self._relay_ok_flood.get(r.task_zone, False)
            )
            if not current_is_open_stair:
                # Check both assigned_zones AND bundle — a zone may be in the
                # bundle but not yet promoted to assigned_zones if CBBA hasn't
                # registered ownership yet. Either way we should preempt.
                candidates = set(r.assigned_zones) | set(r.bundle)
                for z in candidates:
                    if z == r.task_zone: continue
                    if self._shadow_zone_type.get(z) != 'stair': continue
                    if not self._relay_ok_flood.get(z, False): continue
                    zx_p,zy_p=z; x0_p=zx_p*self.zone_w_cells; x1_p=min(x0_p+self.zone_w_cells,GRID_W)
                    y0_p=zy_p*self.zone_h_cells; y1_p=min(y0_p+self.zone_h_cells,GRID_H)
                    sm_p=self._world_stair_arr[x0_p:x1_p,y0_p:y1_p]
                    if not int(np.sum((self.union_belief[x0_p:x1_p,y0_p:y1_p]==T_UNKNOWN)&sm_p)): continue
                    self._release_zone(r, "preempt_for_open_stair")
                    break

        # try to select a zone if none held
        if r.task_zone is None:
            can_enter_stairs = bool(r.caps_mask & (CAP_STAIRS | CAP_AIR))
            robot_inside_stair = (can_enter_stairs
                                  and self.world.grid[r.pos[0]][r.pos[1]]["t"] == T_STAIRS)

            # If robot is physically inside a stair building, first try to
            # claim another zone in the same cluster before looking elsewhere.
            # This prevents robots from exiting mid-building to fetch a zone
            # on the other side of the map.
            if robot_inside_stair:
                current_cluster_id = self._shadow_cluster_id.get(
                    self.cell_to_zone(r.pos[0], r.pos[1]))
                if current_cluster_id is not None:
                    cluster_zones = [z for z, c in self._shadow_cluster_id.items()
                                     if c == current_cluster_id]
                    best_z = None; best_uf = 0.0
                    for z in cluster_zones:
                        if r.blacklist.get(z, -1) > self.timestep: continue
                        if not self._relay_ok_flood.get(z, False): continue
                        # Cheap stair-cell check instead of full frontier BFS
                        zx_c,zy_c=z; x0_c=zx_c*self.zone_w_cells; x1_c=min(x0_c+self.zone_w_cells,GRID_W)
                        y0_c=zy_c*self.zone_h_cells; y1_c=min(y0_c+self.zone_h_cells,GRID_H)
                        sm_c=self._world_stair_arr[x0_c:x1_c,y0_c:y1_c]
                        n_s=int(np.sum(sm_c))
                        if n_s == 0: continue
                        unk_s=int(np.sum((self.union_belief[x0_c:x1_c,y0_c:y1_c]==T_UNKNOWN)&sm_c))
                        if unk_s == 0: continue  # stair cells all explored
                        uf = unk_s / n_s
                        if uf > best_uf:
                            best_uf = uf; best_z = z
                    if best_z is not None:
                        r.task_zone = best_z
                        r.zone_lease_until = self.timestep + LEASE_T
                        r.task_no_progress = 0; r.task_last_known = 0
                        t2 = self.zone_tasks[best_z]
                        if r.name not in t2.owners:
                            t2.owners.append(r.name)
                        t2.status = "held"; t2.expires_at = self.timestep + LEASE_T
                        if best_z not in r.bundle: r.bundle.append(best_z)
                        if best_z not in r.assigned_zones: r.assigned_zones.append(best_z)

            def zone_priority(z):
                stats = self.zone_stats(z)
                sf = stats["shadow_frac"]
                needs_relay = sf > 0.2 and not self._relay_ok_flood.get(z, False)
                relay_open  = sf > 0.2 and self._relay_ok_flood.get(z, False)
                is_stair    = self._shadow_zone_type.get(z) == 'stair'
                # Local unknown fraction — what this robot sees as unexplored
                local_uf = self._local_zone_unknown_frac(r, z)
                # Stair-capable robots: open stair zones rank highest (0),
                # then non-shadow zones (1), then blocked zones (2).
                # This ensures the relay-open rebid actually switches task_zone.
                if can_enter_stairs and relay_open and is_stair:
                    return (0, -local_uf)
                if needs_relay:
                    return (2, -local_uf)
                return (1, -local_uf)

            for z in sorted(r.assigned_zones, key=zone_priority):
                if r.blacklist.get(z,-1) > self.timestep: continue
                stats = self.zone_stats(z)
                if not self.zone_feasible(r, stats, zone=z): continue

                shadow_needs_relay = (stats["shadow_frac"] > 0.2
                                      and not self._relay_ok_flood.get(z, False)
                                      and self._local_zone_unknown_frac(r, z) > 1.0 - ZONE_DONE)
                if shadow_needs_relay:
                    # Only wait at shadow border if a relay is actively travelling
                    # to this cluster — otherwise it's a permanent deadlock.
                    my_cid = self._shadow_cluster_id.get(z)
                    relay_en_route = any(
                        rr.active and rr.role == Role.RELAY
                        and self._shadow_cluster_id.get(rr.task_zone) == my_cid
                        for rr in self.robots
                    )
                    if not relay_en_route:
                        continue  # skip — no relay coming, don't park here
                else:
                    # If robot is physically inside this zone's shadow, it is
                    # actively exploring — never drop the task_zone due to a
                    # stale frontier check. The local belief updates one cell at
                    # a time; frontiers appear as the robot moves.
                    robot_in_this_zone_shadow = (
                        self.radio_shadow[r.pos[0], r.pos[1]]
                        and self.cell_to_zone(r.pos[0], r.pos[1]) == z
                    )
                    if not robot_in_this_zone_shadow and not self.zone_has_frontiers(r, z):
                        continue

                r.task_zone = z
                r.zone_lease_until = self.timestep + LEASE_T
                r.task_no_progress = 0; r.task_last_known = 0
                break

            # If the bundle is non-empty but no zone could be selected (all
            # blacklisted, infeasible, or frontierless), the bundle is stale.
            # Clear immediately so CBBA can reassign next cycle rather than
            # waiting IDLE_RESCUE_K ticks of aimless wandering.
            if r.task_zone is None and r.assigned_zones:
                r.bundle = []; r.assigned_zones = []

        if r.task_zone is None: return

        # Ownership check: if CBBA ran and pruned this robot from task_zone owners,
        # release immediately rather than holding a stale task_zone.
        # This prevents robots from crowding a zone they're no longer assigned to.
        zt_owners = self.zone_tasks.get(r.task_zone)
        if (zt_owners is not None
                and zt_owners.status == "held"
                and zt_owners.owners
                and r.name not in zt_owners.owners
                and r.task_zone not in r.assigned_zones
                and not self.radio_shadow[r.pos[0], r.pos[1]]):
            self._release_zone(r, "ownership_revoked")
            return

        # Coverage from robot's local belief (comms-gated, age-decayed)
        local_cov = 1.0 - self._local_zone_unknown_frac(r, r.task_zone)
        # Also check union (fleet-wide) coverage — if fleet says done, trust it
        union_cov = self.zone_coverage(self.union_belief, r.task_zone)
        cov = max(local_cov, union_cov)

        stats = self.zone_stats(r.task_zone)

        # Shadow zones use a lower completion threshold — get in, sweep the bulk,
        # get out. Chasing the last 15% of a building interior is not worth keeping
        # a relay running and an explorer committed to a single zone.
        is_shadow_zone = self._shadow_frac_for_zone(r.task_zone) > 0.2
        done_threshold = SHADOW_ZONE_DONE if is_shadow_zone else ZONE_DONE

        # For stair zones, also check stair-cell-specific completion.
        # A zone may be only 40% explored by total cells (most are open terrain
        # surrounding the building) but 98% of the actual stair cells are done.
        # In that case the building IS complete — the robot should move on.
        # Use the world-truth stair mask (always known) for this check.
        if (is_shadow_zone and self._shadow_zone_type.get(r.task_zone) == 'stair'):
            zx_s, zy_s = r.task_zone
            x0_s = zx_s*self.zone_w_cells; x1_s = min(x0_s+self.zone_w_cells, GRID_W)
            y0_s = zy_s*self.zone_h_cells; y1_s = min(y0_s+self.zone_h_cells, GRID_H)
            stair_mask_s = self._world_stair_arr[x0_s:x1_s, y0_s:y1_s]
            n_stair = int(np.sum(stair_mask_s))
            if n_stair > 0:
                unk_stair = int(np.sum(
                    (self.union_belief[x0_s:x1_s, y0_s:y1_s] == T_UNKNOWN)
                    & stair_mask_s))
                stair_cov = 1.0 - unk_stair / n_stair
                if stair_cov >= STAIR_CELL_DONE:
                    # All stair cells seen — building complete regardless of zone uf
                    cov = max(cov, done_threshold)

        # Guard: if robot is physically inside a stair zone, block the
        # general coverage-based release until stair cells are done.
        # Zone cov hits SHADOW_ZONE_DONE=0.95 quickly via open terrain around
        # the building, but stair cells may still be 80-90% unexplored.
        _block_done = False
        if (is_shadow_zone
                and self.radio_shadow[r.pos[0], r.pos[1]]
                and self._shadow_zone_type.get(r.task_zone) == 'stair'
                and cov >= done_threshold):
            zx_g, zy_g = r.task_zone
            x0_g = zx_g*self.zone_w_cells; x1_g = min(x0_g+self.zone_w_cells, GRID_W)
            y0_g = zy_g*self.zone_h_cells; y1_g = min(y0_g+self.zone_h_cells, GRID_H)
            sm_g = self._world_stair_arr[x0_g:x1_g, y0_g:y1_g]
            n_s_g = int(np.sum(sm_g))
            if n_s_g > 0:
                unk_s_g = int(np.sum(
                    (self.union_belief[x0_g:x1_g, y0_g:y1_g] == T_UNKNOWN) & sm_g))
                if (1.0 - unk_s_g / n_s_g) < STAIR_CELL_DONE:
                    _block_done = True   # stair cells not done — keep exploring

        if cov >= done_threshold and not _block_done:
            # Zone complete. If robot is inside a stair building and there are
            # other incomplete zones in the same cluster with relay coverage,
            # jump directly to the next one rather than waiting for CBBA.
            if (is_shadow_zone
                    and self.radio_shadow[r.pos[0], r.pos[1]]
                    and self._shadow_zone_type.get(r.task_zone) == 'stair'):
                my_cluster = self._shadow_cluster_id.get(r.task_zone)
                cluster_zones = [z for z, c in self._shadow_cluster_id.items()
                                 if c == my_cluster]
                next_z = None; best_uf = 0.0
                for z in cluster_zones:
                    if z == r.task_zone: continue
                    if not self._relay_ok_flood.get(z, False): continue
                    # Gate on stair-cell incompleteness, not total zone uf.
                    # Total uf includes surrounding open terrain which CBBA handles —
                    # this switch is only for moving to the next BUILDING in the cluster.
                    zx_c, zy_c = z
                    x0_c = zx_c*self.zone_w_cells; x1_c = min(x0_c+self.zone_w_cells, GRID_W)
                    y0_c = zy_c*self.zone_h_cells; y1_c = min(y0_c+self.zone_h_cells, GRID_H)
                    sm_c = self._world_stair_arr[x0_c:x1_c, y0_c:y1_c]
                    n_s_c = int(np.sum(sm_c))
                    if n_s_c == 0: continue   # no stair cells — not a building zone
                    unk_s_c = int(np.sum(
                        (self.union_belief[x0_c:x1_c, y0_c:y1_c] == T_UNKNOWN) & sm_c))
                    stair_uf_c = unk_s_c / n_s_c
                    if stair_uf_c < 0.05: continue   # building already ≥95% done
                    fronts = self.zone_frontiers_for(r, z)
                    if fronts and stair_uf_c > best_uf:
                        best_uf = stair_uf_c; next_z = z
                if next_z is not None:
                    # Direct switch — bypass CBBA, claim the zone immediately
                    self._release_zone(r, "complete_switch_cluster")
                    r.task_zone = next_z
                    r.zone_lease_until = self.timestep + LEASE_T
                    r.task_no_progress = 0; r.task_last_known = 0
                    t2 = self.zone_tasks[next_z]
                    if r.name not in t2.owners:
                        t2.owners.append(r.name)
                    t2.status = "held"; t2.expires_at = self.timestep + LEASE_T
                    if next_z not in r.bundle: r.bundle.append(next_z)
                    if next_z not in r.assigned_zones: r.assigned_zones.append(next_z)
                    return
            self._release_zone(r, "complete"); return
        if not self.zone_feasible(r, stats, zone=r.task_zone):
            self._release_zone(r, "unsuitable"); return

        # No-progress blacklist: robot gives up if it makes no scan progress.
        # For shadow (stair/disc) zones, never blacklist while the robot is
        # physically inside the shadow — being inside and moving IS progress,
        # even if the integer coverage counter hasn't ticked yet.
        # Use a longer limit for shadow zones: the last few cells of a building
        # are hard to reach and take many ticks of slow corner-crawling.
        robot_inside_shadow = (is_shadow_zone
                               and self.radio_shadow[r.pos[0], r.pos[1]])
        if is_shadow_zone:
            # Full NO_PROGRESS_K patience while inside the building.
            # Only halve it while waiting outside for relay coverage.
            no_progress_limit = NO_PROGRESS_K if robot_inside_shadow else NO_PROGRESS_K // 2
        else:
            no_progress_limit = NO_PROGRESS_K

        if (r.role != Role.RELAY
                and not lease_active
                and not robot_inside_shadow          # never blacklist while inside
                and r.task_no_progress >= no_progress_limit):
            self._blacklist_zone(r, r.task_zone, "no_progress"); return

        # Waiting for relay: if the zone needs relay and robot is outside, freeze counter
        waiting_for_relay = (is_shadow_zone
                             and not self.radio_shadow[r.pos[0], r.pos[1]]
                             and not self._relay_ok_flood.get(r.task_zone, False))

        # update progress counter using local coverage
        known_now = int(local_cov * 10000)
        if r.task_last_known == 0: r.task_last_known = known_now
        if known_now > r.task_last_known:
            r.task_no_progress = 0; r.task_last_known = known_now
        else:
            # Freeze counter when: inside shadow (actively scanning) or
            # waiting outside for relay coverage to open.
            if robot_inside_shadow or waiting_for_relay:
                pass  # freeze: not truly stuck
            else:
                zx2, zy2 = r.task_zone
                zx2_min = zx2 * self.zone_w_cells; zx2_max = zx2_min + self.zone_w_cells
                zy2_min = zy2 * self.zone_h_cells; zy2_max = zy2_min + self.zone_h_cells
                rx2, ry2 = r.pos
                dist_to_zone = max(0, zx2_min - rx2, rx2 - zx2_max + 1,
                                   zy2_min - ry2, ry2 - zy2_max + 1)
                if dist_to_zone <= r.terrain_R:
                    r.task_no_progress += 1

        # update frontier signal
        fronts = self.zone_frontiers_for(r, r.task_zone)
        r.zone_frontier_count  = len(fronts)
        r.zone_frontier_signal = min(1.0, len(fronts)/25.0)

    def _move_robot(self, r, occupied):
        """Goal selection and movement for non-relay active robots."""
        # pick new goal if needed
        # NOTE: also replan when path is empty but goal exists — this happens
        # after a sidestep clears the path; without this the robot freezes
        # because move_step exits immediately on empty path without incrementing
        # stuck_steps, so the (goal_commit==0 and stuck_steps>5) branch never fires.
        # Path-empty replan: path ran out but robot hasn't arrived at goal.
        # Replan to the SAME goal — avoids goal-switching instability.
        if not r.path and r.goal is not None and r.pos != r.goal:
            # Before replanning, check for 2-cycle deadlock: another robot is
            # at our goal moving toward our current position. If so, wait 1 tick.
            my_x, my_y = r.pos
            gx, gy = r.goal
            deadlock = any(
                rr is not r and rr.active and rr.pos == (gx, gy)
                and rr.goal == (my_x, my_y)
                for rr in self.robots
            )
            if deadlock:
                # Jitter: pick a random adjacent free cell for 1 tick
                import random as _rnd
                neighbours = [(my_x+dx, my_y+dy) for dx,dy in ((1,0),(-1,0),(0,1),(0,-1))
                              if 0<=my_x+dx<GRID_W and 0<=my_y+dy<GRID_H
                              and self.union_belief[my_x+dx,my_y+dy] not in (T_OBS,T_WATER)
                              and not self.radio_shadow[my_x+dx,my_y+dy]]
                if neighbours:
                    wait_cell = _rnd.choice(neighbours)
                    r.path = [wait_cell]
                    r.move_step(occupied)
                    return
            if not r.set_goal(r.goal):
                r.goal = None
            else:
                r.move_step(occupied)
                return

        # Late-game docility: be more patient before abandoning a goal as the map
        # nears complete, damping goal-churn on sparse far frontier.  Affects only
        # re-selection timing, never capability/zone assignment.
        _cov = getattr(self, 'global_cov', 0.0)
        _patience = 5 + int(25 * max(0.0, _cov - 0.85) / 0.15)   # 5 early → 30 near 100%
        need_new = (r.goal is None or r.pos == r.goal or
                    (r.goal_commit == 0 and r.stuck_steps > _patience))
        if need_new:
            # Fast bundle exhaustion check: if every zone in the bundle has no
            # accessible frontiers, clear immediately and force a CBBA rebid.
            # This prevents boats / non-stair robots sitting idle for IDLE_RESCUE_K
            # ticks when their water zones are fully scanned.
            if r.bundle and r.task_zone is None:
                all_exhausted = all(
                    not self.zone_has_frontiers(r, z)
                    for z in r.bundle
                    if not r.blacklist.get(z, 0) > self.timestep
                )
                if all_exhausted:
                    r.bundle = []; r.assigned_zones = []; r.blacklist = {}
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
            reservations = getattr(self, '_goal_reservations', {})
            reservations[tgt] = r.caps_mask
            if not r.set_goal(tgt):
                r.goal = None
                return  # path failed, try next tick

        r.move_step(occupied)

# ─────────────────────────────────────────────────────────────────────────────
# Rendering helpers
# ─────────────────────────────────────────────────────────────────────────────
def build_grid_surface(sim, show_map, show_survivors, show_risk,
                       union_belief, union_T, union_R):
    """Vectorised grid renderer using pygame.surfarray — no per-cell Python loops."""
    world = sim.world
    W, H = world.w, world.h

    # Build colour array (W, H, 3) using numpy
    colour = np.zeros((W, H, 3), dtype=np.uint8)

    # Terrain colours — map terrain code -> RGB using lookup table
    _TC = np.array([
        TERRAIN_COLOUR_CODE[T_UNKNOWN],
        TERRAIN_COLOUR_CODE[T_FREE],
        TERRAIN_COLOUR_CODE[T_OBS],
        TERRAIN_COLOUR_CODE[T_STAIRS],
        TERRAIN_COLOUR_CODE[T_WATER],
        TERRAIN_COLOUR_CODE[T_BRIDGE],
    ], dtype=np.uint8)  # shape (6, 3)

    if show_map:
        # Build world terrain array once (cached on sim if not present)
        if not hasattr(sim, '_world_terrain_arr'):
            sim._world_terrain_arr = np.array(
                [[world.grid[x][y]["t"] for y in range(H)] for x in range(W)],
                dtype=np.uint8)
        tb_arr = sim._world_terrain_arr
    else:
        tb_arr = union_belief  # uint8, same codes

    colour = _TC[tb_arr]  # (W, H, 3) via fancy indexing

    if show_risk:
        # Build risk colour array vectorised
        if show_map:
            if not hasattr(sim, '_world_temp_arr'):
                sim._world_temp_arr = np.array([[world.grid[x][y]["temp"] for y in range(H)] for x in range(W)], dtype=np.float32)
                sim._world_rad_arr  = np.array([[world.grid[x][y]["rad"]  for y in range(H)] for x in range(W)], dtype=np.float32)
            risk_raw = np.maximum(sim._world_temp_arr, sim._world_rad_arr)
            visible_mask = np.ones((W, H), dtype=bool)
        else:
            t_safe = np.where(np.isnan(union_T), 0.0, union_T)
            r_safe = np.where(np.isnan(union_R), 0.0, union_R)
            risk_raw = np.maximum(t_safe, r_safe)
            visible_mask = (union_belief != T_UNKNOWN)

        max_risk = float(risk_raw.max()) or 1e-9
        n = np.clip(risk_raw / max_risk, 0, 1)
        risk_colour = np.stack([
            (n * 255).astype(np.uint8),
            ((1 - n) * 255).astype(np.uint8),
            np.zeros((W, H), dtype=np.uint8)
        ], axis=2)
        grey = np.full((W, H, 3), 180, dtype=np.uint8)
        colour = np.where(visible_mask[:, :, None], risk_colour, grey)

    # Survivor/found overlay
    red = np.array([255, 0, 0], dtype=np.uint8)
    for pos in sim.found:
        colour[pos[0], pos[1]] = red
    if show_survivors:
        for pos in sim.survivors:
            if pos not in sim.found:
                colour[pos[0], pos[1]] = red

    # Scale up by CELL_SIZE using kron
    ones = np.ones((CELL_SIZE, CELL_SIZE), dtype=np.uint8)
    surf = pygame.Surface((W * CELL_SIZE, H * CELL_SIZE))
    px = pygame.surfarray.pixels3d(surf)
    for c in range(3):
        px[:, :, c] = np.kron(colour[:, :, c], ones)
    del px

    # Faint grid lines (thin, fast)
    gl = (170, 170, 170)
    for px in range(0, W * CELL_SIZE, 4 * CELL_SIZE):
        pygame.draw.line(surf, gl, (px, 0), (px, H * CELL_SIZE), 1)
    for py in range(0, H * CELL_SIZE, 4 * CELL_SIZE):
        pygame.draw.line(surf, gl, (0, py), (W * CELL_SIZE, py), 1)

    return surf


def build_shadow_surface(sim):
    """Vectorised shadow surface using surfarray."""
    # Scale shadow mask up using kron (faster than repeat+repeat)
    big = np.kron(sim.radio_shadow.astype(np.uint8),
                  np.ones((CELL_SIZE, CELL_SIZE), dtype=np.uint8))
    surf = pygame.Surface((GRID_W * CELL_SIZE, GRID_H * CELL_SIZE), pygame.SRCALPHA)
    # Write RGB channels via pixels3d, alpha via pixels_alpha
    rgb = pygame.surfarray.pixels3d(surf)
    rgb[big == 1] = [40, 40, 40]
    del rgb
    alpha = pygame.surfarray.pixels_alpha(surf)
    alpha[big == 1] = 110
    del alpha
    return surf


def draw_shadow_coverage(screen, sim):
    """Vectorised shadow coverage overlay using surfarray."""
    # Build per-cell RGB and alpha arrays at grid resolution
    rgb_arr   = np.zeros((GRID_W, GRID_H, 3), dtype=np.uint8)
    alpha_arr = np.zeros((GRID_W, GRID_H),    dtype=np.uint8)

    # Build covered/uncovered masks
    covered_mask   = np.zeros((GRID_W, GRID_H), dtype=bool)
    uncovered_mask = np.zeros((GRID_W, GRID_H), dtype=bool)
    zw = sim.zone_w_cells; zh = sim.zone_h_cells
    for zx in range(sim.zone_nx):
        for zy in range(sim.zone_ny):
            z = (zx, zy)
            if sim._shadow_zone_type.get(z, 'none') == 'none': continue
            x0 = zx*zw; x1 = min(x0+zw, GRID_W)
            y0 = zy*zh; y1 = min(y0+zh, GRID_H)
            if sim._relay_ok_flood.get(z, False):
                covered_mask[x0:x1, y0:y1] = True
            else:
                uncovered_mask[x0:x1, y0:y1] = True

    shd = sim.radio_shadow
    m_cov = shd & covered_mask
    m_unc = shd & uncovered_mask

    rgb_arr[m_cov]   = [0,   200, 80]
    alpha_arr[m_cov] = 55
    rgb_arr[m_unc]   = [220, 40,  40]
    alpha_arr[m_unc] = 65

    # White border: non-shadow cells adjacent to covered shadow
    shd_i = shd.astype(np.uint8)
    nbr = ((np.roll(shd_i,1,0)|np.roll(shd_i,-1,0)|
             np.roll(shd_i,1,1)|np.roll(shd_i,-1,1)) > 0)
    border = (~shd) & nbr & covered_mask
    rgb_arr[border]   = [255, 255, 255]
    alpha_arr[border] = 120

    # Scale up to screen resolution using kron
    ones = np.ones((CELL_SIZE, CELL_SIZE), dtype=np.uint8)
    big_alpha = np.kron(alpha_arr, ones)

    surf = pygame.Surface((GRID_W*CELL_SIZE, GRID_H*CELL_SIZE), pygame.SRCALPHA)
    px_rgb   = pygame.surfarray.pixels3d(surf)
    px_alpha = pygame.surfarray.pixels_alpha(surf)

    # Scale RGB channels
    for c in range(3):
        px_rgb[:, :, c] = np.kron(rgb_arr[:, :, c], ones)
    px_alpha[:] = big_alpha

    del px_rgb, px_alpha
    screen.blit(surf, (0, 0))


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
            if not r.active:
                continue   # dead robots have no plan to show
            clr = tuple(max(0,c//2) for c in ROBOT_COLOUR[robot_type(r.name)])
            for (px,py) in r.path:
                screen.fill(clr,(px*CELL_SIZE,py*CELL_SIZE,CELL_SIZE,CELL_SIZE))
            if r.goal:
                gx,gy = r.goal
                pygame.draw.rect(screen,ROBOT_COLOUR[robot_type(r.name)],
                                 (gx*CELL_SIZE,gy*CELL_SIZE,CELL_SIZE,CELL_SIZE),2)

    for r in robots:
        x,y = r.pos
        px, py = x*CELL_SIZE, y*CELL_SIZE

        if not r.active:
            # Draw a faded grey tombstone marker so dead robots are visible but
            # clearly out of service.  Dark grey fill + small X cross.
            dead_clr = (60, 60, 60)
            screen.fill(dead_clr, (px, py, CELL_SIZE, CELL_SIZE))
            lc = (120, 120, 120)
            cx, cy = px + CELL_SIZE//2, py + CELL_SIZE//2
            d = max(1, CELL_SIZE//3)
            pygame.draw.line(screen, lc, (cx-d, cy-d), (cx+d, cy+d), 1)
            pygame.draw.line(screen, lc, (cx+d, cy-d), (cx-d, cy+d), 1)
            continue

        clr = ROBOT_COLOUR[robot_type(r.name)]
        screen.fill(clr,(px, py, CELL_SIZE, CELL_SIZE))
        if getattr(r, 'role', None) == Role.RELAY:
            cx, cy = px + CELL_SIZE//2, py + CELL_SIZE//2
            pygame.draw.circle(screen, (255, 220, 80), (cx, cy), 10, 1)
            # Inner ring: relay physically at border — use zone_has_outside_relay
            # if available (FleetSim), otherwise fall back to _relay_ok_flood check
            task = getattr(r, 'task_zone', None)
            if task is not None:
                sim_obj = getattr(r, 'sim', None)
                if sim_obj is not None:
                    if hasattr(sim_obj, 'zone_has_outside_relay'):
                        at_border = sim_obj.zone_has_outside_relay(task)
                    else:
                        at_border = sim_obj._relay_ok_flood.get(task, False)
                    if at_border:
                        pygame.draw.circle(screen, (255, 255, 0), (cx, cy), 7, 2)


# ─────────────────────────────────────────────────────────────────────────────
# GUI loop
# ─────────────────────────────────────────────────────────────────────────────
def gui_loop():
    pygame.init()
    screen = pygame.display.set_mode((GRID_W*CELL_SIZE+SIDEBAR_WIDTH, GRID_H*CELL_SIZE))
    pygame.display.set_caption("Heterogeneous Robot Fleet Simulator")
    font   = pygame.font.SysFont(None, 24)
    random.seed(0); np.random.seed(0)   # fixed seed — reproducible world
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
        if show_shadow:
            screen.blit(shadow_surf,(0,0))
            draw_shadow_coverage(screen, sim)   # live relay coverage overlay
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


def verify_exact_potential(seeds=(0, 1, 2), warmup=40, snapshots=4, spacing=30,
                           cycles_per_snapshot=1500, tol_frac=1e-3):
    """
    Numerical Monderer & Shapley (1996) exact-potential test for the role game.

    A finite game is an EXACT potential game iff the sum of the deviating players'
    utility changes around every 4-cycle of two-player deviations is zero:

        D = [u_i(b_i,a_j) - u_i(a_i,a_j)]
          + [u_j(b_i,b_j) - u_j(b_i,a_j)]
          + [u_i(a_i,b_j) - u_i(b_i,b_j)]
          + [u_j(a_i,a_j) - u_j(a_i,b_j)]   ==  0      (all other robots fixed)

    We sample many such cycles from the REAL _role_utility_pg on real snapshots and
    report the residual |D|, normalised by the typical utility magnitude so it is
    interpretable.  Near-zero everywhere => empirically exact potential; a nonzero
    tail => at best an ordinal potential (still has the finite-improvement property
    and pure-NE existence), which is what should then be claimed.  The relay/non-
    relay split localises any residual to the public-good term.
    """
    import numpy as _np, random as _r
    _ROLES = [Role.SCOUT, Role.SCAN, Role.LOITER, Role.RELAY]
    all_abs = []; relay_abs = []; nonrelay_abs = []; umag = []
    for seed in seeds:
        _r.seed(seed); _np.random.seed(seed)
        sim = FleetSim()
        step = 0; nxt = warmup; taken = 0
        while taken < snapshots and step < warmup + spacing * snapshots + 5:
            sim.step(); step += 1
            if step < nxt:
                continue
            nxt += spacing; taken += 1
            active = [r for r in sim.robots if r.active]
            if len(active) < 2:
                continue
            clusters = getattr(sim, '_last_clusters', []) or []
            cinfo = {cid: (cl, any(sim._relay_ok_flood.get(z, False) for z in cl))
                     for cid, cl in enumerate(clusters)}

            def U(rob):
                return float(sim._role_utility_pg(rob, rob.role, active, rob.task_zone, cinfo))

            saved = {r.name: r.role for r in active}
            for _ in range(cycles_per_snapshot):
                i, j = _r.sample(active, 2)
                a_i, b_i = _r.sample(_ROLES, 2)
                a_j, b_j = _r.sample(_ROLES, 2)
                i.role, j.role = a_i, a_j; ui_aa = U(i); uj_aa = U(j)
                i.role, j.role = b_i, a_j; ui_ba = U(i); uj_ba = U(j)
                i.role, j.role = b_i, b_j; ui_bb = U(i); uj_bb = U(j)
                i.role, j.role = a_i, b_j; ui_ab = U(i); uj_ab = U(j)
                D = (ui_ba - ui_aa) + (uj_bb - uj_ba) + (ui_ab - ui_bb) + (uj_aa - uj_ab)
                ad = abs(D)
                all_abs.append(ad)
                umag.append(abs(ui_aa)); umag.append(abs(uj_aa))
                (relay_abs if Role.RELAY in (a_i, b_i, a_j, b_j) else nonrelay_abs).append(ad)
            for r in active:
                r.role = saved[r.name]

    if not all_abs:
        print("verify_exact_potential: no cycles sampled (instance too small).")
        return
    aa = _np.array(all_abs); scale = max(1e-9, float(_np.median(umag)))
    tol = tol_frac * scale
    print(f"\nExact-potential test (Monderer-Shapley 4-cycle)  —  {len(aa)} cycles, "
          f"{len(seeds)} seeds")
    print(f"  typical |utility| (median)            : {scale:.3f}")
    print(f"  residual |D|   mean={aa.mean():.4g}  median={_np.median(aa):.4g}  "
          f"max={aa.max():.4g}")
    print(f"  residual /scale mean={aa.mean()/scale:.4g}  max={aa.max()/scale:.4g}")
    print(f"  fraction of cycles with |D| <= {tol_frac:g}×scale : "
          f"{100*float(_np.mean(aa <= tol)):.1f}%")
    if relay_abs and nonrelay_abs:
        ra = _np.array(relay_abs); na = _np.array(nonrelay_abs)
        print(f"  cycles WITHOUT relay : mean|D|={na.mean():.4g}  max={na.max():.4g}")
        print(f"  cycles WITH    relay : mean|D|={ra.mean():.4g}  max={ra.max():.4g}")
    if aa.max() <= tol:
        print("  VERDICT: residuals ~0 everywhere -> EMPIRICALLY EXACT potential game.")
    elif _np.median(aa) <= tol:
        print("  VERDICT: ~0 in the median with a nonzero tail -> near-exact; the game")
        print("           is at best an ORDINAL potential. Claim finite-improvement +")
        print("           pure-NE existence (still gives convergence), not exact-Phi.")
    else:
        print("  VERDICT: substantial residuals -> NOT an exact potential game as-is.")
        print("           Safe claims: ordinal potential (FIP + pure NE) or 'potential-")
        print("           game-inspired'. Use the relay/non-relay split to localise it.")


if __name__ == "__main__":
    import sys as _sys
    if "--verify-potential" in _sys.argv:
        verify_exact_potential()
    else:
        gui_loop()