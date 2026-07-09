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
  #Local game where only role occupancy is considered.
  #CBBA algorithm
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
N_HOTSPOTS_TEMP  = 18;  TEMP_AMP_N = 30; TEMP_AMP_P = 0.35; TEMP_AMP_SCALE = 10.0
N_HOTSPOTS_RAD   = 18;  RAD_AMP_N  = 28; RAD_AMP_P  = 0.30; RAD_AMP_SCALE  = 12.0
SIGMA_MIN = 2.5;  SIGMA_MAX = 9.0
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
CBBA_ITERS     = 3
MAX_BUNDLE     = 4
ZONE_CAPACITY  = 2
LEASE_T        = 50
COOLDOWN_T     = 80
NO_PROGRESS_K  = 30
IDLE_RESCUE_K  = 15   # ticks idle with no goal before forcing a CBBA rebid

# Relay
RELAY_MIN_HOLD     = 150       # ticks a relay must stay before being demoted
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

        # ── river system: exactly one river splitting the map horizontally ──
        self._carve_river(base_width=random.randint(5,7), n_ctrl=4)

        # Clean up river immediately after carving (before anything else touches the grid)
        self._fill_river_islands()

        # ── place bridges: exactly one guaranteed crossing ──
        self._place_bridges(n_bridges=1, bridge_w=5, min_spacing=40)

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
               traffic_u16, traffic_w,
               union_T=None, union_R=None):

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
                    # Exception: if the cell is also adjacent to a known bridge, allow it —
                    # the robot is approaching a crossing and should not be pushed back.
                    if has_land and not has_water and not has_air:
                        water_adjacent = False
                        bridge_adjacent = False
                        for ddx, ddy in NBR4:
                            ax, ay = nx+ddx, ny+ddy
                            if 0<=ax<W and 0<=ay<H:
                                nt = int(terrain_u8[ax,ay])
                                if nt == T_WATER:  water_adjacent = True
                                if nt == T_BRIDGE: bridge_adjacent = True
                        if water_adjacent and not bridge_adjacent: continue
                    # Boat robots: unknown cells are only passable if adjacent to known
                    # water or bridge — dry land is contiguous, the unknown is probably
                    # land too.  This mirrors the land halo above and prevents boats from
                    # pathfinding across undiscovered terrain.
                    if has_water and not has_air:
                        water_or_bridge_adjacent = False
                        for ddx, ddy in NBR4:
                            ax, ay = nx+ddx, ny+ddy
                            if 0<=ax<W and 0<=ay<H:
                                nt = int(terrain_u8[ax,ay])
                                if nt in (T_WATER, T_BRIDGE):
                                    water_or_bridge_adjacent = True; break
                        if not water_or_bridge_adjacent: continue
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
                    # Known cell — hard block if lethal according to best available reading
                    # Use union (shared) belief if available, else personal belief
                    t_ = float(union_T[nx, ny]) if (union_T is not None and not np.isnan(union_T[nx, ny])) else float(temp_f32[nx, ny])
                    r_ = float(union_R[nx, ny]) if (union_R is not None and not np.isnan(union_R[nx, ny])) else float(rad_f32[nx, ny])
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
        self.terrain_R              = 4   # sensor radius: terrain reveal AND survivor detection

        if self._reveal_all(): self._recompute_chunked()

    def _reveal_all(self):
        """
        Sensor scan: reveal terrain within terrain_R using ray casting.

        Each cell in the disc of radius terrain_R is revealed only if there is
        line-of-sight from the robot's position — i.e. the Bresenham path to
        that cell is not blocked by T_OBS walls (or T_STAIRS when the robot is
        outside a building).  This matches physical sensor behaviour: a robot
        cannot see through solid walls.

        Survivor detection in step() uses the same radius and the same LOS test,
        so survivors behind walls are not detected until the robot has a clear
        sightline.
        """
        x0, y0 = self.pos
        R   = self.terrain_R
        now = self.sim.timestep
        W, H = self.world.w, self.world.h
        robot_inside_building = (self.world.grid[x0][y0]["t"] == T_STAIRS)

        newly_scanned = []
        new_data = False
        for dx in range(-R, R + 1):
            for dy in range(-R, R + 1):
                if dx*dx + dy*dy > R*R:
                    continue
                nx, ny = x0 + dx, y0 + dy
                if not (0 <= nx < W and 0 <= ny < H):
                    continue
                # Ray cast: skip cell if wall blocks LOS
                if dx != 0 or dy != 0:
                    if not self.sim._has_los(x0, y0, nx, ny, robot_inside_building):
                        continue
                # Cell is visible — update belief
                self.personally_scanned[nx, ny] = True
                self.scan_age[nx, ny] = 0
                self.confidence[nx, ny] = 1.0
                new_cell = not self.known_mask[nx, ny]
                if new_cell:
                    self.known_mask[nx, ny] = True
                    self.terrain_belief[nx, ny] = self.world.grid[nx][ny]["t"]
                    self.temp_belief[nx, ny]    = self.world.grid[nx][ny]["temp"]
                    self.rad_belief[nx, ny]     = self.world.grid[nx][ny]["rad"]
                newly_scanned.append((nx, ny))
                if new_cell: new_data = True
                # else: re-scan — refresh age/confidence, still queue for broadcast

        # Queue all visible cells into outbox for comms propagation
        for (nx, ny) in newly_scanned:
            self.outbox.append({
                'x': nx, 'y': ny,
                'terrain': int(self.terrain_belief[nx, ny]),
                'temp':    float(self.temp_belief[nx, ny]),
                'rad':     float(self.rad_belief[nx, ny]),
                'ts':      now,
            })
        return new_data  # True only if genuinely new cells revealed → caller recomputes chunked

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
        Only accept data fresher than what we already have (by timestamp).
        """
        if not self.inbox: return
        dirty = False
        now = self.sim.timestep
        scan_age = self.scan_age
        terrain_belief = self.terrain_belief
        known_mask = self.known_mask
        temp_belief = self.temp_belief
        rad_belief = self.rad_belief
        confidence = self.confidence
        _exp = math.exp
        _CONF_TAU = CONF_TAU
        for msg in self.inbox:
            x, y = msg['x'], msg['y']
            msg_age = now - msg['ts']
            if msg_age < int(scan_age[x, y]):
                terrain_belief[x, y] = msg['terrain']
                known_mask[x, y]     = True
                t_val = msg['temp']
                if t_val == t_val: temp_belief[x, y] = t_val   # NaN check: NaN != NaN
                r_val = msg['rad']
                if r_val == r_val: rad_belief[x, y]  = r_val
                capped = msg_age if msg_age < 32767 else 32767
                scan_age[x, y]   = capped
                confidence[x, y] = _exp(-capped / _CONF_TAU)
                dirty = True
        self.inbox.clear()
        if dirty:
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
                relay_ok_fn=lambda z: self.sim._relay_ok_flood.get(z, False),
                cell_to_zone_fn=self.sim.cell_to_zone,
                global_cov=self.sim.global_cov,
                unk_pen=unk_pen, info_w=info_w, unk_prior=UNK_PRIOR,
                alpha_mult=a_mult, beta_mult=b_mult, soft_frac=SOFT_FRAC,
                traffic_u16=self.sim.traffic_u16, traffic_w=TRAFFIC_W,
                union_T=self.sim.union_T, union_R=self.sim.union_R,
            )
            if not new_path:
                self.failed_goals[self.goal] = self.sim.timestep + 80
                self.goal = None; self.path = []; self.goal_commit = 0
                return False
            self.path = new_path

        next_cell = self.path[0]

        # ── collision: cooperative yielding (WHCA*-lite) ──
        if next_cell in occupied:
            # Try a 1-step sidestep: find any free adjacent cell that isn't
            # next_cell, isn't the cell we came from (avoid ping-pong),
            # and is passable. Step there for 1 tick then replan.
            # This breaks head-ons and corridor deadlocks without waiting.
            yielded = False
            tb_arr = self.terrain_belief
            for nx, ny in self.world.neighbours(self.pos):
                if (nx, ny) in occupied: continue
                if (nx, ny) == next_cell: continue
                tb = int(tb_arr[nx, ny])
                if not traversable_code(tb, self.caps_mask): continue
                if self.sim.radio_shadow[nx, ny]:
                    z2 = self.sim.cell_to_zone(nx, ny)
                    if not self.sim.relay_ok_extended(z2): continue
                # Step sideways
                occupied.discard(self.pos); occupied.add((nx, ny))
                self.pos = (nx, ny)
                self.path = []      # force full replan next tick
                self.goal_commit = 0
                self.stuck_steps = 0
                if self._reveal_all(): self._scan_dirty = True
                drain = {"Legged":1.0,"Drone":2.0,"Boat":2.0,"Rover":0.4}
                self.battery -= drain.get(robot_type(self.name),1.0) *                     {Role.SCOUT:1.4,Role.SCAN:1.0,Role.RELAY:1.1,Role.LOITER:0.6}[self.role]
                yielded = True
                break
            if not yielded:
                # No sidestep available — wait, but clear goal after 3 ticks
                # so a genuinely new path is found rather than replanning same
                self.stuck_steps += 1
                if self.stuck_steps > 3:
                    self.path = []; self.goal = None; self.goal_commit = 0
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
        # Mark stair cells (building interiors)
        for x in range(GRID_W):
            for y in range(GRID_H):
                if self.world.grid[x][y]["t"] == T_STAIRS:
                    rs[x,y] = True
        # Dilate stair shadow by 1 cell so the relay border falls outside the
        # OBS wall lip — otherwise the nearest reachable cell is across the wall
        stair_only = rs.astype(np.uint8)
        stair_dilated = (np.roll(stair_only,1,0)|np.roll(stair_only,-1,0)|
                         np.roll(stair_only,1,1)|np.roll(stair_only,-1,1)).astype(bool)
        rs |= stair_dilated
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
        # Tag each shadow zone as 'stair' or 'disc'.
        # Zones with actual T_STAIRS cells are 'stair'.
        # Zones whose shadow is entirely dilation artefacts (T_FREE/T_OBS bled out
        # from a neighbouring stair building) inherit 'stair' from that neighbour —
        # they form the door approach corridor and must be in the same relay cluster.
        # All other shadow zones are 'disc' (open-area signal loss).
        self._shadow_zone_type = {}  # zone -> 'stair' | 'disc' | 'none'

        # First pass: classify by actual cell terrain
        stair_cell_counts = {}
        for zx in range(self.zone_nx):
            for zy in range(self.zone_ny):
                z = (zx, zy)
                xs, ys = self.zone_cells(z)
                stair_count = sum(1 for x in xs for y in ys
                                  if rs[x,y] and self.world.grid[x][y]["t"] == T_STAIRS)
                disc_count  = sum(1 for x in xs for y in ys
                                  if rs[x,y] and self.world.grid[x][y]["t"] != T_STAIRS)
                stair_cell_counts[z] = stair_count
                if stair_count + disc_count == 0:
                    self._shadow_zone_type[z] = 'none'
                elif stair_count >= disc_count:
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
        unknown = int(np.sum(union == T_UNKNOWN))
        known   = total - unknown

        # Terrain counts — only among known cells
        known_mask = (union != T_UNKNOWN)
        n_water  = int(np.sum(union == T_WATER))
        n_stairs = int(np.sum(union == T_STAIRS))
        n_free   = int(np.sum(union == T_FREE))

        # Hazard averages — only over known, non-nan cells
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

        # Use world-truth stair fraction so the critical bonus is non-zero even
        # before robots have scanned the building interior. Without this, f_stairs=0
        # for all unexplored stair zones, the CBBA critical bonus is always 0, and
        # capable robots see no utility advantage over open terrain.
        fs_world = self._zone_stair_frac.get(zone, fs)
        if fs_world > fs:
            fs = fs_world

        # Shadow frac uses precomputed static count
        shadow_count = self._zone_shadow_count.get(zone, 0)
        sf = shadow_count / total if total > 0 else 0.0

        cx = zx*self.zone_w_cells + self.zone_w_cells//2
        cy = zy*self.zone_h_cells + self.zone_h_cells//2
        stats = dict(unknown_frac=unknown/total if total>0 else 0.0,
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
            union_T=self.union_T,
            union_R=self.union_R,
        )

    # ── CBBA ──────────────────────────────────────────────────────────────────
    def _zone_capacity(self, zone):
        """
        Effective explorer capacity for a zone.
        Stair zones with an active relay get capacity 3 so the relay robot
        (which holds an owner slot but parks outside) doesn't crowd out the
        two real explorers that need to enter the building.
        All other zones use the global ZONE_CAPACITY=2.
        """
        if (self._shadow_zone_type.get(zone) == 'stair'
                and self._relay_ok_flood.get(zone, False)):
            return 3
        return ZONE_CAPACITY

    def _assign_zones_cbba(self):
        """
        CBBA zone assignment.
        Key fix: consensus loop correctly operates per-zone.
        Key fix: only reset bundles for robots NOT currently mid-task.
        """
        # Precompute dead-robot zone map once per CBBA call (used in _zone_utility)
        self._dead_in_zone_cache = {}
        for r in self.robots:
            if not r.active:
                z = self.cell_to_zone(r.pos[0], r.pos[1])
                if z: self._dead_in_zone_cache[z] = self._dead_in_zone_cache.get(z, 0) + 1

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
                if not r.active or r.battery <= 0: continue
                for z in r.bundle:
                    u = self._zone_utility(r, z, bundle_counts)
                    if u is not None:
                        zone_claims.setdefault(z, []).append((r.name, u))

            # resolve each zone independently
            for z, claims in zone_claims.items():
                if len(claims) <= 1: continue
                claims.sort(key=lambda t: t[1], reverse=True)

                # enforce heterogeneity: one robot per locomotion type per zone.
                # Exception: stair zones with active relay can accept multiple Legged
                # robots since buildings are large and Legged is the only ground type
                # that can enter — the one-per-type rule would permanently cap them at 1.
                is_open_stair = (self._shadow_zone_type.get(z) == 'stair'
                                 and self._relay_ok_flood.get(z, False))
                winners = []; used_types = set()
                for nm, u in claims:
                    rr = next((x for x in self.robots if x.name==nm), None)
                    if rr is None: continue
                    rt = robot_type(nm)
                    if not is_open_stair and rt in used_types: continue
                    winners.append(nm); used_types.add(rt)
                    if len(winners) >= self._zone_capacity(z): break

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
                if r.name not in t.owners and len(t.owners) < self._zone_capacity(z):
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
        shadow_frac = self._shadow_frac_for_zone(zone)
        relay_needed = shadow_frac > 0.05 and not self._relay_ok_flood.get(zone, False)
        relay_active = shadow_frac > 0.05 and self._relay_ok_flood.get(zone, False)

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
            # Only pull explorers in when relay is PHYSICALLY active at border.
            # relay_needed_bonus removed — it was drawing robots into uncovered shadow.
            # Stair-capable robots get a larger bonus on stair zones so they outbid
            # Rovers/Boats who can't actually enter the building interior.
            zone_type = self._shadow_zone_type.get(zone, 'none')
            can_enter_shadow = bool(r.caps_mask & (CAP_STAIRS | CAP_AIR))
            if relay_active and zone_type == 'stair' and can_enter_shadow:
                relay_explorer_bonus = 8.0 * sf * max(uf, 0.5)
            else:
                relay_explorer_bonus = (4.0 * sf * max(uf, 0.5)) if relay_active else 0.0
            dead_bonus = 1.5 * self._dead_in_zone_cache.get(zone, 0)

            u = (1.0*uf + risk_term + terrain_term - 0.05*travel
                 + critical + shadow_bonus + relay_explorer_bonus + dead_bonus - load)

        else:
            # ── UNKNOWN ZONE: bid on information-gain potential only ──────────
            if local_unknown_frac < 0.05: return None   # nothing to learn here

            # Feasibility check applies even for unknown zones — a boat that
            # already knows it's landlocked should not bid on dry zones it can't reach
            stats_for_feasibility = self.zone_stats(zone)
            if not self.zone_feasible(r, stats_for_feasibility, zone=zone): return None

            # Boats avoid zones unlikely to have water (far from map edge/centre water)
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
                terrain_affinity = 2.0 * fs_world - 0.5 * fw_world
            elif r.name.startswith("Drone"):
                terrain_affinity = 1.0 * fs_world
            elif r.name.startswith("Boat"):
                terrain_affinity = 2.0 * fw_world - 0.5 * fs_world
            # Critical-access bonus: stair zones are only reachable by Legged/Drone
            # so they have asymmetric value — count it even in the unknown branch
            can_enter = bool(r.caps_mask & (CAP_STAIRS | CAP_AIR))
            critical_bonus = 1.5 * fs_world if can_enter else 0.0

            # Pull explorers into bubble only when relay is physically active.
            relay_explorer_bonus = (3.0 * shadow_frac * local_unknown_frac) if relay_active else 0.0

            # Slight diversity bonus: prefer zones far from current task
            diversity = 0.1 * travel

            dead_bonus = 1.5 * self._dead_in_zone_cache.get(zone, 0)

            u = (1.5 * info_gain + terrain_affinity + critical_bonus
                 - 0.08*travel + relay_explorer_bonus - diversity + dead_bonus - load)

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
        Pure potential-game role assignment over {SCOUT, SCAN, LOITER, RELAY}.

        Every active robot chooses the role that maximises its marginal-contribution
        utility.  Best-response iteration converges to a pure Nash equilibrium
        because the underlying game is an exact potential game with private costs
        (Monderer & Shapley 1996 — see potential_game_proof.docx).

        The only imperative safety rule retained: a relay that has physically
        drifted inside the radio shadow is immediately demoted to SCAN, because
        it cannot provide comms coverage from inside the shadow.
        """
        t      = self.timestep
        active = [r for r in self.robots if r.active and r.battery > 0]
        shadow = self.radio_shadow

        # ── 1. Build shadow clusters (connected components of shadow zones) ────
        shadow_zones = [
            (zx, zy)
            for zx in range(self.zone_nx)
            for zy in range(self.zone_ny)
            if self._shadow_frac_for_zone((zx, zy)) > 0.05
        ]
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
        self._last_clusters = clusters

        # ── 2. Safety: demote any relay that drifted into shadow ──────────────
        for r in active:
            if r.role == Role.RELAY and shadow[r.pos[0], r.pos[1]]:
                r.role = Role.SCAN
                r.relay_hold_until  = 0
                r.role_locked_until = t + 10

        # ── 3. Potential-game best-response over ALL four roles ───────────────
        self._pg_best_response_roles(active, clusters)

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

    # ── Potential Game constants ───────────────────────────────────────────────
    # Effort weights w(a): marginal exploration value of one robot in role a.
    # Ordering w(SCOUT) > w(SCAN) > w(LOITER) > 0 proven in potential_game_proof.docx.
    _W = {Role.SCOUT: 2.4, Role.SCAN: 1.2, Role.LOITER: 0.1}

    # Congestion coefficients γ(a): cost of each additional robot choosing role a.
    # γ(RELAY) must satisfy:
    #   relay_val_large − γ_R  > best_effort_u   (robot self-elects when nearest)
    #   relay_val_large − 2γ_R < 0               (no double-cover)
    # With relay_val_large ≈ 6–8 and best_effort_u ≈ 0.64: γ_R ∈ (2.68, 4.0).
    # Chosen γ_R = 3.5 — see §4 of potential_game_proof.docx.
    _GAMMA = {Role.SCOUT: 0.8, Role.SCAN: 0.5, Role.LOITER: 0.01, Role.RELAY: 0.8}

    # Private distance cost weight for relay (Lemma 2.7 — preserves potential).
    _RELAY_TRAVEL_W = 3.0

    def _relay_val(self, cluster):
        """
        Value unlocked by placing a relay at the border of `cluster`.
        Base: Σ_z uf(z) · sf(z) · zone_size, normalised by zone_size.
        Urgency ×2 when active explorers are already inside.
        Additional +dead_bonus per dead robot inside — their last position marks
        high-value cells that likely contain survivors or unexplored territory.
        """
        zone_size = self.zone_w_cells * self.zone_h_cells
        val = 0.0
        x0_min = self.world.w; x1_max = 0
        y0_min = self.world.h; y1_max = 0
        cluster_zones = set(cluster)
        for z in cluster:
            uf = self.zone_stats(z)["unknown_frac"]
            sf = self._shadow_frac_for_zone(z)
            val += uf * sf * zone_size
            zx, zy = z
            x0 = zx*self.zone_w_cells; x1 = min(x0+self.zone_w_cells, self.world.w)
            y0 = zy*self.zone_h_cells; y1 = min(y0+self.zone_h_cells, self.world.h)
            x0_min = min(x0_min, x0); x1_max = max(x1_max, x1)
            y0_min = min(y0_min, y0); y1_max = max(y1_max, y1)
        val /= max(1, zone_size)
        # If ANY active explorer is inside this shadow cluster, relay is life-or-death.
        # Use a massive flat bonus that dominates all other role utilities — the game
        # must always elect a relay when robots are at risk of dying inside shadow.
        explorers_inside = sum(
            1 for r in self.robots
            if r.active and r.role != Role.RELAY
            and x0_min <= r.pos[0] < x1_max
            and y0_min <= r.pos[1] < y1_max
            and self.radio_shadow[r.pos[0], r.pos[1]]
        )
        if explorers_inside:
            val += 50.0 * explorers_inside   # overwhelms any effort-role utility
        # Dead robot bonus: each robot that died inside this cluster adds urgency —
        # their last position signals high-value unexplored territory
        dead_inside = sum(
            1 for r in self.robots
            if not r.active
            and self.cell_to_zone(r.pos[0], r.pos[1]) in cluster_zones
            and self.radio_shadow[r.pos[0], r.pos[1]]
        )
        val += dead_inside * 1.5
        return val

    def _role_utility_pg(self, robot, role, s_minus_i_counts, zone, cluster_info):
        """
        Exact potential game marginal-contribution utility for `robot` choosing `role`.

        Parameters
        ----------
        robot           : Robot instance (robot i)
        role            : Role being evaluated
        s_minus_i_counts: dict {Role -> int} — counts of OTHER robots' current roles
                          (excludes robot i itself — the s_{-i} convention from proof)
        zone            : robot's current task_zone (may be None)
        cluster_info    : dict {cluster_id -> (cluster_list, already_covered_bool)}
                          built once per tick in _pg_best_response_roles

        Returns
        -------
        float utility
        """
        n_minus_i = s_minus_i_counts.get(role, 0)   # count in this role excl. robot i

        if role == Role.RELAY:
            # ── Robot-specific relay utility (public good + private distance cost) ─
            # U_i(RELAY) = V(c*) · 1{c* uncovered}
            #              − (n_{−i,RELAY}+1) · γ_RELAY          [public congestion]
            #              − (dist_i / D_NORM) · RELAY_TRAVEL_W   [private cost]
            #
            # c* = best uncovered cluster reachable by this robot.
            # The public-good term V(c)·1{covered} is symmetric (robot-independent),
            # preserving the Rosenthal potential Φ.  The private distance term cancels
            # from ΔΦ = U_i(s_i, s_{-i}) − U_i(s'_i, s_{-i}) and does not affect
            # convergence (Monderer & Shapley 1996, Lemma 2.7 — see proof §4).
            D_NORM = float(self.world.w + self.world.h)
            best_val  = 0.0
            best_dist = D_NORM
            rx, ry    = robot.pos
            for cid, (cl, covered) in cluster_info.items():
                if covered:
                    continue             # 2nd relay adds nothing to public good
                # Skip clusters this robot physically can't serve:
                # boats can't reach land building borders, land robots can't reach water
                cluster_type = self._shadow_zone_type.get(cl[0], 'none') if cl else 'none'
                is_boat = bool(robot.caps_mask & CAP_WATER) and not bool(robot.caps_mask & CAP_AIR)
                if cluster_type == 'stair' and is_boat:
                    continue   # boat can't reach a building's land border
                rv = self._relay_val(cl)
                if rv <= 0:
                    continue
                min_d = min(
                    abs(rx - (z[0]*self.zone_w_cells + self.zone_w_cells//2))
                    + abs(ry - (z[1]*self.zone_h_cells + self.zone_h_cells//2))
                    for z in cl
                )
                net = rv - (min_d / D_NORM) * self._RELAY_TRAVEL_W
                best_net = best_val - (best_dist / D_NORM) * self._RELAY_TRAVEL_W
                if net > best_net:
                    best_val  = rv
                    best_dist = min_d
            # Congestion: only penalise other relays already assigned to THIS cluster,
            # not the total robot count — otherwise 12 robots makes relay utility always negative
            n_relays_this_cluster = sum(
                1 for r2 in self.robots
                if r2 is not robot and r2.active and r2.role == Role.RELAY
                and r2.task_zone is not None
                and any(r2.task_zone in cl for _, (cl, _) in cluster_info.items())
            )
            congestion = (n_relays_this_cluster + 1) * self._GAMMA[Role.RELAY]
            travel     = (best_dist / D_NORM) * self._RELAY_TRAVEL_W

            # Explorer-opportunity cost: a stair-capable robot assigned to an open
            # stair zone pays a large penalty for choosing RELAY, because it is the
            # only type that can enter the building and should not be diverted.
            # Scale penalty with proximity: the closer to the zone, the larger the
            # penalty (robot has committed to that approach corridor).
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
                # Penalty grows as robot gets closer: max penalty at zone boundary,
                # fades to zero at ~4× sensor radius away so far robots can still relay.
                fade = max(0.0, 1.0 - dist_to_zone / (robot.terrain_R * 4))
                explorer_penalty = 6.0 * fade

            return best_val - congestion - travel - explorer_penalty

        # ── Effort roles: SCOUT / SCAN / LOITER ──
        # Use the robot's current task_zone uf if it has productive work.
        # If the zone is done or absent, look at the best available zone in the fleet
        # (relay-covered shadow zones included) so robots don't all collapse to LOITER
        # just because their personal zone is exhausted.
        if zone is not None:
            uf = self.zone_stats(zone)["unknown_frac"]
        else:
            uf = 0.0

        if uf < 0.05:
            # Current zone is exhausted — find the best available unexplored zone.
            # Shadow zones WITHOUT relay coverage still count toward uf: they represent
            # real remaining work (unknown cells, potential survivors) and must prevent
            # robots from collapsing to LOITER just because those zones are currently
            # unreachable.  Relay election is a separate decision; effort roles should
            # reflect that work exists, not whether access has been arranged yet.
            best_uf = 0.0
            for zx in range(self.zone_nx):
                for zy in range(self.zone_ny):
                    z_uf = self.zone_stats((zx, zy))["unknown_frac"]
                    if z_uf > best_uf:
                        best_uf = z_uf
            uf = best_uf

        w = self._W[role]
        exploration_gain = uf * w
        congestion = (n_minus_i + 1) * self._GAMMA[role]

        # LOITER penalty: when unexplored area exists in the fleet, LOITER is wasteful.
        # Uses the global uf (best available zone) rather than local zone uf so
        # a robot in a done zone can't cheaply LOITER while the map is largely unseen.
        # Penalty is role-specific (not robot-specific) so the Rosenthal potential
        # game form still holds — Φ(s) = ... - γ_r * n_r*(n_r+1)/2 still separates.
        loiter_penalty = 0.0
        if role == Role.LOITER and uf > 0.05:
            loiter_penalty = 1.2 * uf   # strong penalty; LOITER only wins when uf≈0

        # Additional LOITER penalty for stair-capable robots assigned to an open
        # stair zone. These robots must not idle — the building window is time-limited
        # and LOITER has near-zero congestion cost so it wins by default when all
        # effort roles are crowded. This penalty overrides that.
        if role == Role.LOITER and zone is not None:
            if (self._shadow_zone_type.get(zone) == 'stair'
                    and self._relay_ok_flood.get(zone, False)
                    and bool(robot.caps_mask & (CAP_STAIRS | CAP_AIR))):
                loiter_penalty += 4.0

        return exploration_gain - congestion - loiter_penalty

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

        # cluster_info: {cid -> (cluster_list, covered_bool)}
        # "covered" = at least one robot has chosen RELAY with this cluster as
        # task_zone AND that robot's hold is not about to expire.
        # A relay whose hold expires within RELAY_HANDOFF_WINDOW ticks is treated
        # as uncovered so: (a) it re-evaluates whether to stay, and (b) a nearby
        # replacement robot sees high relay utility and can take over.
        RELAY_HANDOFF_WINDOW = 15   # ticks before hold expiry to trigger handoff
        cluster_info = {}
        for cid, cl in enumerate(clusters):
            cl_set   = set(cl)
            # Relay physically at border AND hold not expiring soon
            covered = (
                any(self.zone_has_outside_relay(z) for z in cl) or
                any(r.role == Role.RELAY
                    and r.task_zone in cl_set
                    and r.relay_hold_until > t + RELAY_HANDOFF_WINDOW
                    for r in active)
            )
            cluster_info[cid] = (cl, covered)

        all_roles = [Role.SCOUT, Role.SCAN, Role.LOITER, Role.RELAY]

        # Best-response loop — terminates because each step strictly increases Φ
        # and the joint action space is finite.
        max_iters = len(eligible) + 1
        for _iter in range(max_iters):
            changed = False
            random.shuffle(eligible)   # avoid systematic ordering bias

            for robot in eligible:
                # Skip robots that became locked during this BR iteration
                if t < robot.role_locked_until:
                    continue

                # s_{-i}: role counts of all OTHER active robots
                s_minus_i = {}
                for r2 in active:
                    if r2 is robot: continue
                    s_minus_i[r2.role] = s_minus_i.get(r2.role, 0) + 1

                # Evaluate every role and pick the best
                best_role = robot.role
                best_u    = self._role_utility_pg(
                    robot, robot.role, s_minus_i, robot.task_zone, cluster_info)

                for role in all_roles:
                    if role == robot.role:
                        continue
                    u = self._role_utility_pg(
                        robot, role, s_minus_i, robot.task_zone, cluster_info)
                    if u > best_u + 1e-9:
                        best_u = u; best_role = role

                if best_role == robot.role:
                    continue

                # Role change
                old_role  = robot.role
                robot.role = best_role
                changed    = True

                if best_role == Role.RELAY:
                    # Initialise relay navigation state for _move_relay
                    # Target: most-valuable uncovered cluster nearest to this robot
                    rx, ry = robot.pos
                    D_NORM = float(self.world.w + self.world.h)
                    best_cl = None; best_net = -1e9
                    for cid, (cl, covered) in cluster_info.items():
                        if covered: continue
                        # Boats can't serve stair (building) clusters — land-only border
                        cluster_type = self._shadow_zone_type.get(cl[0], 'none') if cl else 'none'
                        is_boat = bool(robot.caps_mask & CAP_WATER) and not bool(robot.caps_mask & CAP_AIR)
                        if cluster_type == 'stair' and is_boat: continue
                        # Prefer not to elect a stair-capable robot as relay for a stair
                        # cluster if there is at least one other stair/air capable robot
                        # free to explore. Relay duty wastes unique building access.
                        can_enter = bool(robot.caps_mask & (CAP_STAIRS | CAP_AIR))
                        if cluster_type == 'stair' and can_enter:
                            free_explorers = sum(
                                1 for rr in active
                                if rr is not robot
                                and rr.role != Role.RELAY
                                and bool(rr.caps_mask & (CAP_STAIRS | CAP_AIR))
                            )
                            if free_explorers == 0:
                                pass   # no alternative — allow it
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
                        net = rv - (min_d / D_NORM) * self._RELAY_TRAVEL_W
                        if net > best_net:
                            best_net = net; best_cl = cl
                    if best_cl is not None:
                        robot.task_zone = min(
                            best_cl,
                            key=lambda z: self.zone_coverage(self.union_belief, z))
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
                        robot.relay_hold_until   = t + RELAY_MIN_HOLD
                        robot.role_locked_until  = t + RELAY_MIN_HOLD
                        robot.relay_last_occupied = t
                        robot.relay_anchor       = None
                    else:
                        # No uncovered cluster to serve — don't become relay, no lock
                        robot.role = old_role
                        changed = (old_role != robot.role)

                elif old_role == Role.RELAY:
                    # Relay stepped down — clear navigation state and invalidate
                    # all reachable caches so robots stop treating shadow as accessible
                    robot.relay_anchor      = None
                    robot.relay_anchor_zone = None
                    robot.relay_hold_until  = 0
                    robot.role_locked_until = 0   # don't carry relay lock into new role
                    for rr in self.robots:
                        rr._reachable_cache = None

            if not changed:
                break   # Nash equilibrium reached

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

        # Task zone centre (target area)
        zx, zy = r.task_zone
        cx = zx * self.zone_w_cells + self.zone_w_cells // 2
        cy = zy * self.zone_h_cells + self.zone_h_cells // 2

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
                # Auto-extend hold while explorers are still active inside this cluster.
                # Without this the relay demotes at RELAY_MIN_HOLD even if a Legged is
                # still clearing the building interior — abandoning them mid-mission.
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
                if explorers_inside:
                    # Keep extending hold so the relay never demotes mid-clearance
                    r.relay_hold_until  = max(r.relay_hold_until,  self.timestep + RELAY_MIN_HOLD)
                    r.role_locked_until = max(r.role_locked_until, self.timestep + RELAY_MIN_HOLD)
                if r._reveal_all(): r._recompute_chunked()
                return
            # Wrong cluster border — clear anchor and reselect
            r.relay_anchor = None; r.relay_anchor_zone = None

        # Recompute anchor only if needed (zone changed or lost anchor)
        if r.relay_anchor_zone != r.task_zone or r.relay_anchor is None:
            zone_type = self._shadow_zone_type.get(r.task_zone, 'none')

            # Build candidate border cells: only those adjacent to our cluster's shadow
            # Filter _shadow_border_cells_arr to cells whose shadow neighbour is
            # in the same cluster and same type as task_zone
            cluster_border = []
            for pt in self._shadow_border_cells_arr:
                bx, by = int(pt[0]), int(pt[1])
                # Skip cells the relay has already tried and failed to reach
                if (bx, by) in getattr(r, '_relay_anchor_blacklist', set()):
                    continue
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

            # Pick closest border cell to zone centre
            best_border = min(cluster_border, key=lambda p: abs(p[0]-cx)+abs(p[1]-cy))
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
                union_T=self.union_T, union_R=self.union_R,
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
            r.relay_failed_path_count = 0   # successful replan, reset counter

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
            if r._reveal_all(): r._recompute_chunked()
            drain = {"Legged": 1.0, "Drone": 2.0, "Boat": 2.0, "Rover": 0.4}
            r.battery -= drain.get(robot_type(r.name), 1.0) * 1.1

    # ── choose exploration goal ───────────────────────────────────────────────
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
            best = None; best_d = 1e9
            x0 = max(0, zx*self.zone_w_cells - 4)
            x1 = min(self.world.w, (zx+1)*self.zone_w_cells + 4)
            y0 = max(0, zy*self.zone_h_cells - 4)
            y1 = min(self.world.h, (zy+1)*self.zone_h_cells + 4)
            for x in range(x0, x1):
                for y in range(y0, y1):
                    if self.radio_shadow[x, y]: continue
                    if union[x, y] in (T_OBS, T_UNKNOWN): continue
                    if reach_arr is not None and not reach_arr[x, y]: continue
                    d = abs(x-cx) + abs(y-cy)
                    if d < best_d:
                        best_d = d; best = (x, y)
            if best is not None:
                return best
            # No reachable cell near zone — fall through to normal goal selection

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
                # Hard filter: never pick a goal in uncovered shadow
                if (self.radio_shadow[x, y]
                        and not self._relay_ok_flood.get(self.cell_to_zone(x, y), False)):
                    continue
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
                          and not (self.radio_shadow[int(p[0]),int(p[1])]
                                   and not self._relay_ok_flood.get(
                                       self.cell_to_zone(int(p[0]),int(p[1])), False))
                          and any(union[nx,ny]==T_UNKNOWN
                                  for nx,ny in self.world.neighbours((int(p[0]),int(p[1]))))]

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

        # Crowd penalty: count active robots with goals within Manhattan dist 10
        crowd = np.zeros(N, dtype=np.float32)
        for rr in self.robots:
            if rr is r or not rr.active or rr.goal is None: continue
            gx2, gy2 = rr.goal
            crowd += (np.abs(cx_a - gx2) + np.abs(cy_a - gy2) < 10).astype(np.float32)

        # Shadow-pull bonus: when relay is active for a stair zone and this robot
        # can enter stairs, strongly prefer shadow-interior cells over edge frontiers.
        # Without this, robots with a stair task_zone oscillate at the zone boundary
        # picking whichever non-shadow frontier is momentarily closest.
        shadow_pull = np.zeros(N, dtype=np.float32)
        if (r.task_zone is not None
                and self._shadow_zone_type.get(r.task_zone) == 'stair'
                and self._relay_ok_flood.get(r.task_zone, False)
                and bool(r.caps_mask & (CAP_STAIRS | CAP_AIR))):
            in_shadow = np.array(
                [self.radio_shadow[cx_a[i], cy_a[i]] for i in range(N)],
                dtype=np.float32)
            # Subtract a large constant from shadow candidates so they sort first.
            # Use dist as the tiebreaker so the robot enters via the shortest path.
            shadow_pull = in_shadow * 60.0

        scores = dist.astype(np.float32) - 2.5*info + risk_term + 10.0*crowd - shadow_pull
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

        # ── CBBA reassignment every 50 ticks ──
        if self.timestep % 50 == 0:
            self._assign_zones_cbba()

        # ── rebuild per-tick caches ──
        self._frontier_cache.clear()
        self._reachable_by_mask.clear()
        self._zone_stats_cache_tick = -1  # force zone stats refresh
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
            for msg in r.outbox:
                msg['deliver_at'] = deliver_at
            self._pending_msgs.extend(r.outbox)
            r.outbox.clear()

        # Phase 5: deliver matured messages to all comms-capable robots
        still_pending = []
        deliverable = []
        for item in self._pending_msgs:
            if t >= item['deliver_at']:
                deliverable.append(item)
            else:
                still_pending.append(item)
        self._pending_msgs = still_pending

        if deliverable:
            for r in active_robots:
                if comms_ok(r):
                    r.inbox.extend(deliverable)

        # Phase 6: each robot processes inbox
        for r in active_robots:
            r._process_inbox()
        # Recompute chunked hazard maps once per robot per tick (deferred from move_step)
        for r in active_robots:
            if r._inbox_dirty or r._scan_dirty:
                r._recompute_chunked()
                r._inbox_dirty = False
                r._scan_dirty  = False

        # ── rebuild union belief ──────────────────────────────────────────────
        self.union_belief = self._union_terrain()
        self.union_T      = self._union_temp()
        self.union_R      = self._union_rad()

        # ── survivor detection ──────────────────────────────────────────────────
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
        # If any active explorer is inside relay-covered shadow, skip role changes
        # this tick — demoting the relay while robots depend on it kills them.
        explorers_in_covered_shadow = any(
            r.active and r.role != Role.RELAY
            and self.radio_shadow[r.pos[0], r.pos[1]]
            and self.relay_ok_extended(self.cell_to_zone(r.pos[0], r.pos[1]))
            for r in self.robots
        )
        if not explorers_in_covered_shadow:
            self._decide_roles()

        # ── occupation set (collision reservation) ──
        occupied = {r.pos for r in self.robots if r.active and r.battery>0}

        # ── PHASE 1: move relays first, then update relay_ok ──
        for r in self.robots:
            if not r.active: continue
            if r.role == Role.RELAY:
                self._move_relay(r, occupied)

        # update relay_ok AFTER relays have settled
        # Compute efficiently: for each relay at a shadow border, mark its zone covered
        # instead of checking all 64 zones × all robots × all neighbours
        self.relay_ok = {(zx,zy): False
                         for zx in range(self.zone_nx) for zy in range(self.zone_ny)}
        for r in self.robots:
            if not r.active or r.role != Role.RELAY: continue
            if r.task_zone is None: continue
            rx, ry = r.pos
            if self.radio_shadow[rx, ry]: continue  # relay inside shadow doesn't count
            zone_type = self._shadow_zone_type.get(r.task_zone, 'none')
            if zone_type == 'none': continue
            # Check if any neighbour is a shadow cell in the same cluster
            for nx, ny in self.world.neighbours((rx, ry)):
                if not self.radio_shadow[nx, ny]: continue
                nz = self.cell_to_zone(nx, ny)
                if nz is None: continue
                if self._shadow_zone_type.get(nz, 'none') != zone_type: continue
                if not self._same_shadow_cluster(nz, r.task_zone): continue
                # Mark the zone the relay is touching as having outside relay
                self.relay_ok[nz] = True
                break
        self._compute_relay_flood()  # propagate through connected zone cluster

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
                    # Don't interrupt a robot that is already en route to a shadow goal
                    if r.goal and self.radio_shadow[r.goal[0], r.goal[1]]: continue
                    # Release current task_zone if it isn't a stair zone
                    if (r.task_zone is not None
                            and self._shadow_zone_type.get(r.task_zone) != 'stair'):
                        self._release_zone(r, "relay_opened_stair")
                    r.bundle = [z for z in r.bundle
                                if self._shadow_zone_type.get(z) == 'stair']
                self._assign_zones_cbba()

            self._relay_ok_prev = dict(self._relay_ok_flood)

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
        # Before processing, check if any explorer is in uncovered shadow.
        # If so, run an emergency re-election + relay move so the fleet gets
        # one extra chance to cover them before the kill fires this tick.
        shadow_at_risk = [
            r for r in self.robots if r.active and r.role != Role.RELAY
            and r.battery > 0
            and self.radio_shadow[r.pos[0], r.pos[1]]
            and not self.relay_ok_extended(self.cell_to_zone(r.pos[0], r.pos[1]))
        ]
        if shadow_at_risk:
            self._decide_roles()
            for r in self.robots:
                if r.active and r.role == Role.RELAY:
                    self._move_relay(r, occupied)
            self.relay_ok = {(zx,zy): False
                             for zx in range(self.zone_nx) for zy in range(self.zone_ny)}
            for r2 in self.robots:
                if not r2.active or r2.role != Role.RELAY or r2.task_zone is None: continue
                r2x, r2y = r2.pos
                if self.radio_shadow[r2x, r2y]: continue
                zt2 = self._shadow_zone_type.get(r2.task_zone, 'none')
                if zt2 == 'none': continue
                for nx2, ny2 in self.world.neighbours((r2x, r2y)):
                    if not self.radio_shadow[nx2, ny2]: continue
                    nz2 = self.cell_to_zone(nx2, ny2)
                    if nz2 and self._shadow_zone_type.get(nz2,'none')==zt2 and self._same_shadow_cluster(nz2,r2.task_zone):
                        self.relay_ok[nz2] = True; break
            self._compute_relay_flood()

        for r in self.robots:
            if not r.active: continue
            if r.role == Role.RELAY: continue
            if r.battery <= 0:
                if r.active:
                    r.active = False; r.death_reason = "battery depleted"
                    self.dead_robots.append((r.name, r.death_reason))
                continue

            # ── Shadow kill: robot in shadow with no relay coverage dies ──
            if (self.radio_shadow[r.pos[0], r.pos[1]]
                    and not self.relay_ok_extended(self.cell_to_zone(r.pos[0], r.pos[1]))):
                r.active = False
                r.death_reason = "lost comms — relay dropped"
                self.dead_robots.append((r.name, r.death_reason))
                occupied.discard(r.pos)
                continue

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
            can_enter_stairs = bool(r.caps_mask & (CAP_STAIRS | CAP_AIR))
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
            # Only count no_progress while the robot is actually near the zone.
            # A robot travelling across the map toward a distant stair zone will
            # never scan that zone en-route, so counting ticks during transit
            # causes premature blacklisting before the robot arrives.
            # Use zone boundary distance: if the robot is outside sensor range of
            # the zone, freeze the counter rather than incrementing it.
            zx2, zy2 = r.task_zone
            zx2_min = zx2 * self.zone_w_cells
            zx2_max = zx2_min + self.zone_w_cells
            zy2_min = zy2 * self.zone_h_cells
            zy2_max = zy2_min + self.zone_h_cells
            rx2, ry2 = r.pos
            dist_to_zone = max(0, zx2_min - rx2, rx2 - zx2_max + 1,
                               zy2_min - ry2, ry2 - zy2_max + 1)
            if dist_to_zone <= r.terrain_R:
                r.task_no_progress += 1
            # else: robot is still en route — don't penalise

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
        need_new = (r.goal is None or r.pos == r.goal or
                    (not r.path and r.goal is not None) or
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
        if r.role == Role.RELAY:
            cx, cy = px + CELL_SIZE//2, py + CELL_SIZE//2
            # Outer ring (thin, semi-transparent yellow) — relay elected, en-route or holding
            pygame.draw.circle(screen, (255, 220, 80), (cx, cy), 10, 1)
            # Inner ring (thicker, bright yellow) — relay physically at shadow border
            # zone_has_outside_relay checks physical position, not just task_zone assignment
            if r.sim.zone_has_outside_relay(r.task_zone) if r.task_zone else False:
                pygame.draw.circle(screen, (255, 255, 0), (cx, cy), 7, 2)


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


if __name__ == "__main__":
    gui_loop()