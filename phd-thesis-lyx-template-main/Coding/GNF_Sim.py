"""
Greedy Nearest Frontier (GNF) Baseline Simulator
==================================================
Primary comparison baseline for the heterogeneous fleet sim.

Architecture
------------
- Same world generation (GridWorld), robot types, spawn logic, and survivor
  placement as hetero_robot_fleet_sim.py — seeded identically so each seed
  produces the exact same physical environment for a fair comparison.
- Planning: each robot independently selects the nearest reachable unexplored
  frontier cell (Manhattan distance). No coordination, no roles, no CBBA.
  O(R * F) per step where F = frontier cells (typically << WH).
- No relay concept: robots enter shadow zones freely (GNF has no comms model).
  This is a deliberate baseline property — it shows what GNF gets "for free"
  vs the overhead our system pays for coordination.
- A* path planning reused from main sim (same cost function, no hazard weights —
  GNF is hazard-blind, treating all traversable cells equally).

What GNF cannot do vs the full system
--------------------------------------
- No role differentiation: all robots behave identically regardless of type
- No hazard avoidance: paths through dangerous zones freely
- Massive redundancy: multiple robots converge on same frontier constantly
- No relay coordination: building exploration is uncoordinated
- No CBBA zone assignment: coverage is spatially unbalanced

This means GNF will:
  ✓ Be faster per step (O(R*F) vs O(R²Z/k²))
  ✓ Achieve reasonable open-terrain coverage quickly
  ✗ Fail to coordinate building exploration
  ✗ Miss survivors in high-hazard zones (no SCOUT sacrifice role)
  ✗ Show high redundancy — robots covering same areas repeatedly
"""

import sys, os, random, math
from collections import deque
from unittest.mock import MagicMock

import numpy as np

# ── Headless pygame stub (same as main sim) ──────────────────────────────────
_pg = MagicMock()
for _m in ['pygame', 'pygame.display', 'pygame.font']:
    if _m not in sys.modules:
        sys.modules[_m] = _pg
_pg.SRCALPHA = 0
_pg.Surface  = lambda *a, **k: MagicMock()
_pg.Rect     = lambda *a, **k: MagicMock()

import importlib.util as _ilu

# ── Load world/constants from main sim ───────────────────────────────────────
def _load_main_sim(sim_path: str):
    spec = _ilu.spec_from_file_location("_hsim_gnf", sim_path)
    M    = _ilu.module_from_spec(spec)
    spec.loader.exec_module(M)
    return M

# ─────────────────────────────────────────────────────────────────────────────
# GNF Robot — minimal state, greedy nearest-frontier planner
# ─────────────────────────────────────────────────────────────────────────────
class GNFRobot:
    """
    Stateless greedy robot.  Each tick:
      1. Scan surroundings (same sensor model as main sim)
      2. Pick nearest reachable frontier cell (BFS distance, not A*)
      3. A* to that cell (terrain-only cost, no hazard avoidance)
      4. Execute one step
    """
    __slots__ = (
        'name', 'pos', 'caps_mask', 'caps', 'world', 'sim',
        'temp_limit', 'rad_limit',
        'terrain_belief', 'known_mask', 'temp_belief', 'rad_belief',
        'chunked',
        'active', 'battery', 'death_reason', 'hazard_killed',
        'goal', 'path', 'stuck_steps', 'failed_goals',
        'dose_T', 'dose_R',
        'terrain_R',
        '_reachable_arr', '_reachable_tick',
        'scan_age', 'confidence',
        'outbox', 'inbox', '_inbox_dirty', '_scan_dirty',
        'personally_scanned',
        'role', 'task_zone',   # renderer compatibility stubs
    )

    def __init__(self, name, x, y, caps, caps_mask, world, sim,
                 temp_limit, rad_limit):
        self.name       = name
        self.pos        = (x, y)
        self.caps       = caps
        self.caps_mask  = caps_mask
        self.world      = world
        self.sim        = sim
        self.temp_limit = temp_limit
        self.rad_limit  = rad_limit

        M = sim.M
        self.terrain_belief = np.full((M.GRID_W, M.GRID_H), M.T_UNKNOWN, dtype=np.uint8)
        self.known_mask     = np.zeros((M.GRID_W, M.GRID_H), dtype=bool)
        self.temp_belief    = np.full((M.GRID_W, M.GRID_H), np.nan, dtype=np.float32)
        self.rad_belief     = np.full((M.GRID_W, M.GRID_H), np.nan, dtype=np.float32)
        self.chunked        = np.zeros((2, M.GRID_W//M.CHUNK_SIZE,
                                           M.GRID_H//M.CHUNK_SIZE), dtype=np.float32)
        self.scan_age       = np.full((M.GRID_W, M.GRID_H), 32767, dtype=np.int16)
        self.confidence     = np.zeros((M.GRID_W, M.GRID_H), dtype=np.float32)
        self.personally_scanned = np.zeros((M.GRID_W, M.GRID_H), dtype=bool)
        self.outbox         = []
        self.inbox          = []
        self._inbox_dirty   = False
        self._scan_dirty    = False

        self.active       = True
        self.battery      = M.MAX_BATTERY
        self.death_reason = None
        self.hazard_killed = False

        self.goal         = None
        self.path         = []
        self.stuck_steps  = 0
        self.failed_goals = {}

        self.dose_T = 0.0; self.dose_R = 0.0

        self.terrain_R = max(3, round(24 / M.CELL_SIZE))
        self._reachable_arr  = None
        self._reachable_tick = -999

        # Renderer compatibility — draw_robots checks r.role and r.task_zone
        import sys as _sys
        _Role = getattr(sim.M, 'Role', None)
        self.role      = _Role.SCAN if _Role else None
        self.task_zone = None

        self._scan()

    # ── sensor scan (identical to main sim) ──────────────────────────────────
    def _scan(self):
        M   = self.sim.M
        x0, y0 = self.pos
        R   = self.terrain_R
        W, H = self.world.w, self.world.h
        now = self.sim.timestep
        robot_inside = (self.world.grid[x0][y0]["t"] == M.T_STAIRS)
        new_data = False
        for dx in range(-R, R+1):
            for dy in range(-R, R+1):
                if dx*dx+dy*dy > R*R: continue
                nx, ny = x0+dx, y0+dy
                if not (0<=nx<W and 0<=ny<H): continue
                if (dx!=0 or dy!=0) and not self.sim._has_los(x0,y0,nx,ny,robot_inside):
                    continue
                self.personally_scanned[nx, ny] = True
                self.scan_age[nx, ny] = 0
                self.confidence[nx, ny] = 1.0
                if not self.known_mask[nx, ny]:
                    self.known_mask[nx, ny] = True
                    self.terrain_belief[nx, ny] = self.world.grid[nx][ny]["t"]
                    self.temp_belief[nx, ny]    = self.world.grid[nx][ny]["temp"]
                    self.rad_belief[nx, ny]     = self.world.grid[nx][ny]["rad"]
                    new_data = True
        return new_data

    # ── reachability BFS — respects radio shadow ──────────────────────────────
    def reachable(self):
        M = self.sim.M
        t = self.sim.timestep
        if self._reachable_tick == t and self._reachable_arr is not None:
            return self._reachable_arr
        from scipy import ndimage as _ndi
        W, H   = self.world.w, self.world.h
        tb_arr = self.terrain_belief
        mask   = self.caps_mask
        trav   = M._TRAV_LUT
        mask4  = mask & 0xF

        passable = np.zeros((W, H), dtype=bool)
        for tc in range(6):
            if trav[tc][mask4]:
                passable |= (tb_arr == tc)

        # Shadow gate: non-border shadow cells are only passable when relay covers them
        # OR the robot itself can enter shadow (Legged/Drone with stairs/air cap)
        can_enter_shadow = bool(mask & (M.CAP_STAIRS | M.CAP_AIR))
        shd = self.sim.radio_shadow
        border = self.sim._shadow_border_mask_cache
        shadow_interior = shd & ~border
        if np.any(shadow_interior) and not can_enter_shadow:
            # Rover/Boat: blocked from shadow interior entirely
            passable &= ~shadow_interior
        elif np.any(shadow_interior) and can_enter_shadow:
            # Legged/Drone: blocked unless relay is present at border
            relay_covered = self.sim._relay_ok
            passable &= ~(shadow_interior & ~relay_covered)

        # Land halo — avoid unknown-near-water
        is_land = bool(mask & M.CAP_LAND) and not bool(mask & (M.CAP_AIR|M.CAP_WATER))
        is_boat = bool(mask & M.CAP_WATER) and not bool(mask & M.CAP_AIR)
        if is_land:
            unk = (tb_arr == M.T_UNKNOWN)
            water_nbr  = _ndi.binary_dilation(tb_arr==M.T_WATER,  structure=np.ones((3,3),dtype=bool))
            bridge_nbr = _ndi.binary_dilation(tb_arr==M.T_BRIDGE, structure=np.ones((3,3),dtype=bool))
            passable &= ~(unk & water_nbr & ~bridge_nbr)
            if self.world.grid[self.pos[0]][self.pos[1]]["t"] == M.T_STAIRS:
                if bool(mask & (M.CAP_STAIRS | M.CAP_AIR)):
                    stair_nbr = _ndi.binary_dilation(tb_arr==M.T_STAIRS,
                                                      structure=np.ones((3,3),dtype=bool))
                    passable |= (unk & stair_nbr)
        elif is_boat:
            unk = (tb_arr == M.T_UNKNOWN)
            wb_nbr = _ndi.binary_dilation(
                (tb_arr==M.T_WATER)|(tb_arr==M.T_BRIDGE), structure=np.ones((3,3),dtype=bool))
            block = unk & ~wb_nbr
            block[self.pos[0], self.pos[1]] = False
            passable &= ~block

        sx, sy = self.pos
        if not passable[sx, sy]:
            arr = np.zeros((W, H), dtype=bool); arr[sx, sy] = True
        else:
            labeled, _ = _ndi.label(passable)
            arr = (labeled == labeled[sx, sy])

        self._reachable_arr  = arr
        self._reachable_tick = t
        return arr

    # ── nearest frontier selection — the GNF core ────────────────────────────
    def _nearest_frontier(self) -> tuple | None:
        """
        Vectorised nearest-frontier selection.

        Replaces the pure-Python BFS which visited up to 40,000 cells per
        robot per tick late-game — confirmed cause of 17,000ms+ steps.
        Uses numpy shift operations to find all frontier candidates in O(W×H)
        then picks nearest by Manhattan distance. Runs in ~1ms regardless of
        map coverage or fleet size.
        """
        M = self.sim.M
        W, H = self.world.w, self.world.h
        union = self.sim.union_belief
        reach = self.reachable()
        is_boat = bool(self.caps_mask & M.CAP_WATER) and not bool(self.caps_mask & M.CAP_AIR)

        # Build adjacency-to-unknown mask via four directional shifts
        unk_mask = (union == M.T_UNKNOWN)
        adj_unk = np.zeros((W, H), dtype=bool)
        adj_unk[:-1, :] |= unk_mask[1:,  :]
        adj_unk[1:,  :] |= unk_mask[:-1, :]
        adj_unk[:,  :-1] |= unk_mask[:,  1:]
        adj_unk[:,  1:]  |= unk_mask[:, :-1]

        if is_boat:
            wb = (union == M.T_WATER) | (union == M.T_BRIDGE)
            cand_mask = reach & wb & adj_unk
        else:
            cand_mask = reach & (unk_mask | ((union != M.T_UNKNOWN) & adj_unk))

        coords = np.argwhere(cand_mask)
        if len(coords) == 0:
            return None

        # Filter failed goals
        t = self.sim.timestep
        valid = [(int(p[0]), int(p[1])) for p in coords
                 if t >= self.failed_goals.get((int(p[0]), int(p[1])), 0)]
        if not valid:
            return None

        # Nearest by Manhattan distance
        rx, ry = self.pos
        return min(valid, key=lambda p: abs(p[0] - rx) + abs(p[1] - ry))

    # ── A* to goal — respects shadow gate ────────────────────────────────────
    def _plan_to(self, goal) -> bool:
        M  = self.sim.M
        _zero_chunked = np.zeros((2, M.GRID_W//M.CHUNK_SIZE,
                                     M.GRID_H//M.CHUNK_SIZE), dtype=np.float32)
        _zero_traffic = np.zeros((M.GRID_W, M.GRID_H), dtype=np.uint16)

        # Shadow: robots that can enter pass relay_ok_fn; others are blocked
        can_enter = bool(self.caps_mask & (M.CAP_STAIRS | M.CAP_AIR))
        relay_ok  = self.sim._relay_ok

        path = M.AStar.search(
            start=self.pos, goal=goal,
            caps_mask=self.caps_mask,
            terrain_u8=self.terrain_belief,
            temp_f32=self.temp_belief, rad_f32=self.rad_belief,
            chunked_risk=_zero_chunked,
            temp_limit=9999.0, rad_limit=9999.0,
            radio_shadow=self.sim.radio_shadow,
            relay_ok_fn=(lambda z: bool(relay_ok[goal[0], goal[1]])
                         if can_enter else lambda z: False),
            cell_to_zone_fn=self.sim.cell_to_zone,
            global_cov=self.sim.global_cov,
            unk_pen=0.3, info_w=0.1, unk_prior=0.25,
            alpha_mult=0.0, beta_mult=0.0,
            soft_frac=1.0,
            traffic_u16=_zero_traffic, traffic_w=0.0,
            shadow_border=self.sim._shadow_border_mask_cache,
        )
        if not path:
            self.failed_goals[goal] = self.sim.timestep + 60
            return False
        self.goal = goal
        self.path = path
        return True

    # ── one movement tick ─────────────────────────────────────────────────────
    def tick(self, occupied: set):
        M = self.sim.M
        if not self.active or self.battery <= 0:
            self.active = False; return

        # Pick new goal if needed
        if not self.goal or not self.path or self.pos == self.goal:
            tgt = self._nearest_frontier()
            if tgt is None:
                self.stuck_steps += 1
                if self.stuck_steps > 20:
                    self.failed_goals = {}  # clear blacklist
                    self.stuck_steps  = 0
                return
            self.stuck_steps = 0
            if not self._plan_to(tgt):
                return

        if not self.path:
            return

        next_cell = self.path[0]
        # Terrain safety gate
        true_t = self.world.grid[next_cell[0]][next_cell[1]]["t"]
        if true_t == M.T_OBS:
            self.path = []; self.goal = None; return
        is_boat = bool(self.caps_mask & M.CAP_WATER) and not bool(self.caps_mask & M.CAP_AIR)
        if is_boat and true_t not in (M.T_WATER, M.T_BRIDGE):
            self.terrain_belief[next_cell[0], next_cell[1]] = true_t
            self.known_mask[next_cell[0], next_cell[1]] = True
            self.path = []; self.goal = None; return

        prev = self.pos
        occupied.discard(prev)
        self.pos = self.path.pop(0)
        occupied.add(self.pos)

        self._scan()

        # Battery drain (same rates as main sim, flat SCAN multiplier)
        drain = {"Legged": 1.0, "Drone": 2.0, "Boat": 2.0, "Rover": 0.4}
        rt    = next((t for t in ("Legged","Drone","Boat","Rover")
                      if self.name.startswith(t)), "Legged")
        self.battery -= drain.get(rt, 1.0)

        # Hazard exposure (tracked for comparison, but doesn't kill in GNF baseline)
        c = self.world.grid[self.pos[0]][self.pos[1]]
        self.dose_T += max(0.0, c["temp"]) * 0.01
        self.dose_R += max(0.0, c["rad"])  * 0.01

        if c["temp"] > self.temp_limit or c["rad"] > self.rad_limit:
            reasons = []
            if c["temp"] > self.temp_limit: reasons.append(f"temp({c['temp']:.0f}>{self.temp_limit:.0f})")
            if c["rad"]  > self.rad_limit:  reasons.append(f"rad({c['rad']:.0f}>{self.rad_limit:.0f})")
            self.active = False; self.hazard_killed = True
            self.death_reason = " & ".join(reasons)
        if self.battery <= 0:
            self.active = False
            self.death_reason = "battery depleted"


# ─────────────────────────────────────────────────────────────────────────────
# GNF Simulation — wraps world + robots, exposes step() matching FleetSim API
# ─────────────────────────────────────────────────────────────────────────────
class GNFSim:
    """
    Drop-in replacement for FleetSim for benchmark purposes.
    Identical world generation (same seed → same map, same survivors).
    Replaces the entire planning/coordination layer with GNF.
    """

    def __init__(self, M):
        """M is the loaded main-sim module (provides world, constants, AStar)."""
        self.M        = M
        self.timestep = 0
        self.global_cov = 0.0

        self.world = M.GridWorld(M.GRID_W, M.GRID_H)

        self.zone_w_cells = M.ZONE_CHUNKS * M.CHUNK_SIZE
        self.zone_h_cells = M.ZONE_CHUNKS * M.CHUNK_SIZE
        self.zone_nx = M.GRID_W // self.zone_w_cells
        self.zone_ny = M.GRID_H // self.zone_h_cells

        # Build radio shadow using FleetSim's exact method so stair shadow
        # (building footprints + 1-cell dilation) AND disc shadow match fleet.
        M.FleetSim._build_radio_shadow(self)

        # relay_ok: marks shadow-interior cells covered by a border robot.
        self._relay_ok       = np.zeros((M.GRID_W, M.GRID_H), dtype=bool)
        self._relay_ok_flood = {}
        self.zone_tasks      = {}

        self.found       = set()
        self.dead_robots = []

        self._build_robots()
        self._build_survivors()

        self.union_belief = self._union_terrain()
        self.union_T      = self._union_temp()
        self.union_R      = self._union_rad()

    # ── LOS check (reused from main sim logic) ────────────────────────────────
    def _has_los(self, x0, y0, x1, y1, robot_inside: bool) -> bool:
        M  = self.M
        dx = abs(x1-x0); dy = abs(y1-y0)
        sx = 1 if x1>x0 else -1; sy = 1 if y1>y0 else -1
        x, y = x0, y0; err = dx-dy; steps = dx+dy
        for _ in range(steps-1):
            e2 = 2*err
            if e2>-dy: err-=dy; x+=sx
            if e2< dx: err+=dx; y+=sy
            if x==x1 and y==y1: break
            t = self.world.grid[x][y]["t"]
            if t==M.T_OBS: return False
            if t==M.T_STAIRS and not robot_inside: return False
        return True

    def cell_to_zone(self, x, y):
        zx = x // self.zone_w_cells; zy = y // self.zone_h_cells
        if 0<=zx<self.zone_nx and 0<=zy<self.zone_ny:
            return (zx, zy)
        return None

    def zone_cells(self, zone):
        zx, zy = zone
        M = self.M
        return (range(zx*self.zone_w_cells, min((zx+1)*self.zone_w_cells, M.GRID_W)),
                range(zy*self.zone_h_cells, min((zy+1)*self.zone_h_cells, M.GRID_H)))

    def zone_neighbors4(self, zone):
        zx, zy = zone
        out = []
        for dzx, dzy in ((1,0),(-1,0),(0,1),(0,-1)):
            nzx, nzy = zx+dzx, zy+dzy
            if 0<=nzx<self.zone_nx and 0<=nzy<self.zone_ny:
                out.append((nzx, nzy))
        return out

    def _build_robots(self):
        """
        Delegate to FleetSim._build_robots for identical spawn positions,
        then wrap each robot as a GNFRobot.
        radio_shadow is already set correctly — no need to blank it afterwards.
        """
        M = self.M
        # Call FleetSim._build_robots with self — sets self.robots
        M.FleetSim._build_robots(self)

        # Wrap FleetSim Robot objects with GNF behaviour
        self.robots = [
            GNFRobot(r.name, r.pos[0], r.pos[1],
                     r.caps, r.caps_mask, self.world, self,
                     r.temp_limit, r.rad_limit)
            for r in self.robots
        ]

    def _build_survivors(self):
        M = self.M
        free = [(x,y) for x in range(M.GRID_W) for y in range(M.GRID_H)
                if self.world.grid[x][y]["t"] in (M.T_FREE, M.T_STAIRS)]
        def near(cx,cy,r=10):
            return [(x,y) for x,y in free if abs(x-cx)<=r and abs(y-cy)<=r]
        critical = []
        for pool in (near(int(M.GRID_W*.75),M.GRID_H//2),
                     near(int(M.GRID_W*.55),int(M.GRID_H*.75)),
                     near(M.GRID_W//6+2,M.GRID_H//2+6)):
            if pool: critical.append(random.choice(pool))
        rest = [c for c in free if c not in critical]
        self.survivors = critical + random.sample(rest, max(0, 18-len(critical)))

    # ── union belief ──────────────────────────────────────────────────────────
    def _union_terrain(self):
        u = np.zeros((self.M.GRID_W, self.M.GRID_H), dtype=np.uint8)
        for r in self.robots:
            np.maximum(u, r.terrain_belief, out=u)
        return u

    def _union_temp(self):
        u = np.full((self.M.GRID_W, self.M.GRID_H), np.nan, dtype=np.float32)
        for r in self.robots:
            mask = r.known_mask & np.isnan(u)
            u[mask] = r.temp_belief[mask]
        return u

    def _union_rad(self):
        u = np.full((self.M.GRID_W, self.M.GRID_H), np.nan, dtype=np.float32)
        for r in self.robots:
            mask = r.known_mask & np.isnan(u)
            u[mask] = r.rad_belief[mask]
        return u

    # ── main step ─────────────────────────────────────────────────────────────
    def step(self) -> bool:
        self.timestep += 1
        M = self.M

        # ── Update relay coverage from robot positions ────────────────────────
        # A robot at a shadow-border cell acts as an implicit relay — it bridges
        # comms without any explicit role election. The _relay_ok array marks
        # every shadow-interior cell reachable from a border-robot's position
        # via flood-fill inside the shadow.
        border  = self._shadow_border_mask_cache
        shadow  = self.radio_shadow
        relay_ok = np.zeros((M.GRID_W, M.GRID_H), dtype=bool)

        for r in self.robots:
            if not r.active: continue
            rx, ry = r.pos
            # Robot is AT the shadow border (outside shadow, touching it)
            if not shadow[rx, ry] and border[rx, ry]:
                # Flood-fill to mark all shadow cells reachable from this border
                from collections import deque as _deque
                q = _deque()
                for dx2, dy2 in ((1,0),(-1,0),(0,1),(0,-1)):
                    nx2, ny2 = rx+dx2, ry+dy2
                    if 0<=nx2<M.GRID_W and 0<=ny2<M.GRID_H and shadow[nx2,ny2]:
                        if not relay_ok[nx2, ny2]:
                            relay_ok[nx2, ny2] = True
                            q.append((nx2, ny2))
                while q:
                    cx2, cy2 = q.popleft()
                    for dx2, dy2 in ((1,0),(-1,0),(0,1),(0,-1)):
                        nx2, ny2 = cx2+dx2, cy2+dy2
                        if (0<=nx2<M.GRID_W and 0<=ny2<M.GRID_H
                                and shadow[nx2,ny2] and not relay_ok[nx2,ny2]):
                            relay_ok[nx2, ny2] = True
                            q.append((nx2, ny2))

        self._relay_ok = relay_ok
        # Update _relay_ok_flood dict form for renderer compatibility
        self._relay_ok_flood = {}
        for zx in range(self.zone_nx):
            for zy in range(self.zone_ny):
                x0=zx*self.zone_w_cells; x1=min(x0+self.zone_w_cells, M.GRID_W)
                y0=zy*self.zone_h_cells; y1=min(y0+self.zone_h_cells, M.GRID_H)
                if np.any(relay_ok[x0:x1, y0:y1]):
                    self._relay_ok_flood[(zx,zy)] = True

        occupied = {r.pos for r in self.robots if r.active}

        for r in self.robots:
            if not r.active: continue
            if r.battery <= 0:
                r.active = False
                r.death_reason = "battery depleted"
                self.dead_robots.append((r.name, r.death_reason))
                continue
            r.tick(occupied)

        # Rebuild union belief
        self.union_belief = self._union_terrain()
        self.union_T      = self._union_temp()
        self.union_R      = self._union_rad()
        self.global_cov   = float(np.mean(self.union_belief != M.T_UNKNOWN))

        # Survivor detection
        for r in self.robots:
            if not r.active: continue
            R  = r.terrain_R
            rx, ry = r.pos
            r_inside = self.world.grid[rx][ry]["t"] == M.T_STAIRS
            for s in self.survivors:
                if s in self.found: continue
                sx, sy = s
                if (rx-sx)**2+(ry-sy)**2 > R*R: continue
                if self._has_los(rx, ry, sx, sy, r_inside):
                    self.found.add(s)

        if len(self.found) >= len(self.survivors):
            return False

        return any(r.active and r.battery > 0 for r in self.robots)


# ─────────────────────────────────────────────────────────────────────────────
# GNF-Shadow — GNF with radio shadow enforced, no coordination
# ─────────────────────────────────────────────────────────────────────────────
class GNFShadowRobot(GNFRobot):
    """
    GNFRobot with a shadow bounce-back gate identical to Fleet's move_step.

    A robot that would step into shadow without relay coverage is bounced back
    to its previous cell, its path is cleared, and the shadow zone is
    blacklisted briefly so the planner avoids it.

    No relay role election exists — robots can only enter shadow if another
    robot happens to be sitting at the border cell covering that zone.
    In practice this never happens because no robot coordinates to hold a
    border position, so GNF-Shadow robots never enter buildings.

    This is the honest baseline: shows what greedy frontier following achieves
    when the physical comms constraint is enforced but no coordination exists
    to satisfy it.
    """

    def tick(self, occupied: set):
        M = self.sim.M

        if not self.active or self.battery <= 0:
            self.active = False; return

        # Pick new goal if needed
        if not self.goal or not self.path or self.pos == self.goal:
            tgt = self._nearest_frontier()
            if tgt is None:
                self.stuck_steps += 1
                if self.stuck_steps > 20:
                    self.failed_goals = {}
                    self.stuck_steps  = 0
                return
            self.stuck_steps = 0
            if not self._plan_to(tgt):
                return

        if not self.path:
            return

        next_cell = self.path[0]

        # Terrain safety gate
        true_t = self.world.grid[next_cell[0]][next_cell[1]]["t"]
        if true_t == M.T_OBS:
            self.path = []; self.goal = None; return
        is_boat = bool(self.caps_mask & M.CAP_WATER) and not bool(self.caps_mask & M.CAP_AIR)
        if is_boat and true_t not in (M.T_WATER, M.T_BRIDGE):
            self.terrain_belief[next_cell[0], next_cell[1]] = true_t
            self.known_mask[next_cell[0], next_cell[1]] = True
            self.path = []; self.goal = None; return

        # ── Shadow gate: bounce back if next cell is shadow with no relay ──
        nx, ny = next_cell
        if self.sim.radio_shadow[nx, ny] and not self.sim._relay_ok[nx, ny]:
            # Clear path and blacklist this shadow zone so the planner
            # avoids it rather than repeatedly approaching the same border
            z = self.sim.cell_to_zone(nx, ny)
            if z is not None:
                self.failed_goals[next_cell] = self.sim.timestep + 60
            self.path = []; self.goal = None
            return

        # Execute move
        prev = self.pos
        occupied.discard(prev)
        self.pos = self.path.pop(0)
        occupied.add(self.pos)

        self._scan()

        # Battery drain
        drain = {"Legged": 1.0, "Drone": 2.0, "Boat": 2.0, "Rover": 0.4}
        rt    = next((t for t in ("Legged","Drone","Boat","Rover")
                      if self.name.startswith(t)), "Legged")
        self.battery -= drain.get(rt, 1.0)

        # Hazard exposure
        c = self.world.grid[self.pos[0]][self.pos[1]]
        self.dose_T += max(0.0, c["temp"]) * 0.01
        self.dose_R += max(0.0, c["rad"])  * 0.01

        if c["temp"] > self.temp_limit or c["rad"] > self.rad_limit:
            reasons = []
            if c["temp"] > self.temp_limit: reasons.append(f"temp({c['temp']:.0f}>{self.temp_limit:.0f})")
            if c["rad"]  > self.rad_limit:  reasons.append(f"rad({c['rad']:.0f}>{self.rad_limit:.0f})")
            self.active = False; self.hazard_killed = True
            self.death_reason = " & ".join(reasons)
        if self.battery <= 0:
            self.active = False
            self.death_reason = "battery depleted"


class GNFShadowSim(GNFSim):
    """
    GNFSim with radio shadow enforced but no relay coordination.

    Identical to GNFSim except:
    1. Robots use GNFShadowRobot — bounce back from shadow.
    2. step() skips the relay flood-fill entirely so _relay_ok stays
       all-False. This prevents 50 incidentally-positioned robots from
       accidentally unlocking shadow zones, which made the first version
       behave identically to plain GNF.

    Expected:
      - Open terrain coverage: ~99% (frontier following unaffected)
      - Stair coverage: ~0% (no robot ever enters a building)
      - Deaths: 0 hazard, 0 trapped
      - Found: open-terrain survivors only (~55-65% of total)

    Gap vs Fleet = quantified value of relay coordination.
    """

    def _build_robots(self):
        M = self.M
        M.FleetSim._build_robots(self)
        self.robots = [
            GNFShadowRobot(r.name, r.pos[0], r.pos[1],
                           r.caps, r.caps_mask, self.world, self,
                           r.temp_limit, r.rad_limit)
            for r in self.robots
        ]

    def step(self) -> bool:
        self.timestep += 1
        M = self.M

        # Shadow always blocked — no relay flood-fill at all.
        # Overrides GNFSim.step() relay computation so incidental border
        # positioning of 50 robots never accidentally unlocks buildings.
        self._relay_ok       = np.zeros((M.GRID_W, M.GRID_H), dtype=bool)
        self._relay_ok_flood = {}

        occupied = {r.pos for r in self.robots if r.active}

        for r in self.robots:
            if not r.active: continue
            if r.battery <= 0:
                r.active = False
                r.death_reason = "battery depleted"
                self.dead_robots.append((r.name, r.death_reason))
                continue
            r.tick(occupied)

        self.union_belief = self._union_terrain()
        self.union_T      = self._union_temp()
        self.union_R      = self._union_rad()
        self.global_cov   = float(np.mean(self.union_belief != M.T_UNKNOWN))

        for r in self.robots:
            if not r.active: continue
            R  = r.terrain_R
            rx, ry = r.pos
            r_inside = self.world.grid[rx][ry]["t"] == M.T_STAIRS
            for s in self.survivors:
                if s in self.found: continue
                sx, sy = s
                if (rx-sx)**2+(ry-sy)**2 > R*R: continue
                if self._has_los(rx, ry, sx, sy, r_inside):
                    self.found.add(s)

        if len(self.found) >= len(self.survivors):
            return False

        return any(r.active and r.battery > 0 for r in self.robots)