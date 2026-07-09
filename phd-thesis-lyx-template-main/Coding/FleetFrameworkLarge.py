"""
Large-Scale GUI Runner — with all performance patches
=====================================================
Launches the fleet sim visualiser with the 1km² / 50-robot configuration
and all benchmark performance patches applied.

Run from the Coding folder:
    python run_large_gui.py [seed] [display_px]

Examples:
    python run_large_gui.py          # seed=3, 4px/cell
    python run_large_gui.py 3        # seed=3
    python run_large_gui.py 0 5      # seed=0, 5px/cell (1000×1000 window)

Controls:
    SPACE   — pause / resume
    S       — step one tick while paused
    P       — toggle path display
    Q / ESC — quit
"""

import sys, os, random, importlib.util
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SIM_FILE = os.path.join(BASE_DIR, "2DFleetFrameworkJ.py")
if not os.path.exists(SIM_FILE):
    SIM_FILE = os.path.join(BASE_DIR, "hetero_robot_fleet_sim.py")

spec = importlib.util.spec_from_file_location("hsim", SIM_FILE)
M    = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)

# ── Large-scale config ────────────────────────────────────────────────────────
LARGE_GRID_W      = 200
LARGE_GRID_H      = 200
LARGE_CELL_SIZE   = 5
LARGE_ZONE_CHUNKS = 10
LARGE_MAX_BATTERY = 1500
LARGE_MAX_BUNDLE  = 8
LARGE_N_SURVIVORS = 45
LARGE_ROBOTS = {"Legged": 13, "Drone": 17, "Boat": 8, "Rover": 12}

M.GRID_W      = LARGE_GRID_W
M.GRID_H      = LARGE_GRID_H
M.CELL_SIZE   = LARGE_CELL_SIZE
M.ZONE_CHUNKS = LARGE_ZONE_CHUNKS
M.MAX_BATTERY = LARGE_MAX_BATTERY
M.MAX_BUNDLE  = LARGE_MAX_BUNDLE

# ── Large-map scaling patches ─────────────────────────────────────────────────
M.RELAY_MIN_HOLD = 300
if hasattr(M, 'ALLOCATION_CADENCE'):
    M.ALLOCATION_CADENCE = 100
M.COOLDOWN_T = 120
M.CBBA_ITERS = 1

# ── Performance Fix 1: goal_commit ───────────────────────────────────────────
_orig_robot_init = M.Robot.__init__
def _patched_robot_init(self, *args, **kwargs):
    _orig_robot_init(self, *args, **kwargs)
    self.goal_commit = 60
M.Robot.__init__ = _patched_robot_init

# ── Performance Fix 2: shared reachable arrays ───────────────────────────────
_orig_reachable = M.Robot.reachable
def _patched_reachable(self):
    t = self.sim.timestep
    if self._reachable_tick == t and self._reachable_arr is not None:
        return self._reachable_arr
    mask = self.caps_mask
    if not hasattr(self.sim, '_reachable_mask_tick'):
        self.sim._reachable_mask_tick = {}
    tick_cache = self.sim._reachable_mask_tick
    if tick_cache.get(mask) == t and mask in self.sim._reachable_by_mask:
        self._reachable_arr   = self.sim._reachable_by_mask[mask]
        self._reachable_cache = None
        self._reachable_tick  = t
        return self._reachable_arr
    result = _orig_reachable(self)
    self.sim._reachable_by_mask[mask] = result
    tick_cache[mask] = t
    return result
M.Robot.reachable = _patched_reachable

# ── Performance Fix 3: lock CBBA cadence to 50 ticks minimum ─────────────────
_orig_step = M.FleetSim.step
def _patched_step(self):
    if not hasattr(self, '_union_cov_cache') or self._union_cov_tick != self.timestep:
        self._union_cov_cache = float(np.mean(self.union_belief != M.T_UNKNOWN))
        self._union_cov_tick  = self.timestep
    _real_cov = self._union_cov_cache
    self._union_cov_cache = min(self._union_cov_cache, 0.69)
    result = _orig_step(self)
    self._union_cov_cache = _real_cov
    return result
M.FleetSim.step = _patched_step

# ── Performance Fix 4: gate _decide_roles on change signal ───────────────────
_orig_decide_roles = M.FleetSim._decide_roles
def _patched_decide_roles(self):
    t = self.timestep
    relay_flood = getattr(self, '_relay_ok_flood', {})
    shadow_type = getattr(self, '_shadow_zone_type', {})
    relay_sig = frozenset(z for z, ok in relay_flood.items() if ok)
    n_active  = sum(1 for r in self.robots if r.active)
    cbba_tick = getattr(self, '_cbba_last_tick', -999)   # correct attribute name
    stair_sig = frozenset(z for z in relay_flood if shadow_type.get(z, 'none') == 'stair')
    changed = (
        relay_sig != getattr(self, '_pr_relay_sig',  None) or
        n_active  != getattr(self, '_pr_n_active',   -1)   or
        cbba_tick != getattr(self, '_pr_cbba_tick',  -999) or
        stair_sig != getattr(self, '_pr_stair_sig',  None)
    )
    last_run = getattr(self, '_pr_last_run', -999)
    if not changed and (t - last_run) < 5:
        return
    self._pr_relay_sig = relay_sig
    self._pr_n_active  = n_active
    self._pr_cbba_tick = cbba_tick
    self._pr_stair_sig = stair_sig
    self._pr_last_run  = t
    _orig_decide_roles(self)
M.FleetSim._decide_roles = _patched_decide_roles

# ── Robot spawn patch ─────────────────────────────────────────────────────────
def _patched_build_robots(self):
    templates = [
        ("Legged", {M.Capability.LAND, M.Capability.STAIRS},
         np.array([10., 10.]), (M.TEMP_LIMIT, M.RAD_LIMIT)),
        ("Drone",  {M.Capability.AIR},
         np.array([10., 10.]), (M.TEMP_LIMIT, M.RAD_LIMIT)),
        ("Boat",   {M.Capability.WATER},
         np.array([0.,  0.]),  (9999., 9999.)),
        ("Rover",  {M.Capability.LAND},
         np.array([-2., -2.]), (9999., 9999.)),
    ]
    tpl = {n: (n, c, w, l) for n, c, w, l in templates}
    spawn = []
    for t, n in LARGE_ROBOTS.items():
        spawn += [t] * n
    random.shuffle(spawn)

    W, H = M.GRID_W, M.GRID_H
    clusters = [
        (W//6,   H//6),   (W//2, H//6),   (5*W//6, H//6),
        (W//6,   H//2),   (W//2, H//2),   (5*W//6, H//2),
        (W//6, 5*H//6),   (W//2, 5*H//6), (5*W//6, 5*H//6),
    ]
    water_cells = [(x, y) for x in range(W) for y in range(H)
                   if self.world.grid[x][y]["t"] == M.T_WATER]
    water_by_quad = {(qx, qy): [(x, y) for x, y in water_cells
                                 if (x < W//2) == (qx == 0) and (y < H//2) == (qy == 0)]
                     for qx in (0, 1) for qy in (0, 1)}

    self.robots = []
    for i, tname in enumerate(spawn):
        _, caps, weights, (tlim, rlim) = tpl[tname]
        name   = f"{tname}{i}"
        center = clusters[i % len(clusters)]
        qx = 0 if center[0] < W//2 else 1
        qy = 0 if center[1] < H//2 else 1

        if tname == "Boat" and water_cells:
            pool = water_by_quad.get((qx, qy), []) or water_cells
            ns   = [c for c in pool if not self.radio_shadow[c[0], c[1]]]
            sx, sy = random.choice(ns or pool)
        else:
            sx, sy = center
            for _ in range(50):
                cx = max(1, min(W-2, center[0] + random.randint(-12, 12)))
                cy = max(1, min(H-2, center[1] + random.randint(-12, 12)))
                tt = self.world.grid[cx][cy]["t"]
                if (tt == M.T_FREE or
                        (tt == M.T_STAIRS and M.Capability.STAIRS in caps)) \
                        and not self.radio_shadow[cx, cy]:
                    sx, sy = cx, cy; break
        self.robots.append(
            M.Robot(name, sx, sy, caps, self.world, self, weights, tlim, rlim))

M.FleetSim._build_robots = _patched_build_robots

# ── Survivor patch ────────────────────────────────────────────────────────────
def _patched_build_survivors(self):
    W, H = M.GRID_W, M.GRID_H
    free_open  = [(x, y) for x in range(W) for y in range(H)
                  if self.world.grid[x][y]["t"] == M.T_FREE]
    free_stair = [(x, y) for x in range(W) for y in range(H)
                  if self.world.grid[x][y]["t"] == M.T_STAIRS]

    n_total    = LARGE_N_SURVIVORS
    n_building = max(3, round(n_total * 0.40))
    n_open     = n_total - n_building

    building_survivors = random.sample(free_stair, min(n_building, len(free_stair)))

    grid_k = round(n_open ** 0.5)
    cell_w = W // grid_k; cell_h = H // grid_k
    open_survivors = []
    for gx in range(grid_k):
        for gy in range(grid_k):
            x0 = gx * cell_w; x1 = min(W, x0 + cell_w)
            y0 = gy * cell_h; y1 = min(H, y0 + cell_h)
            pool = [(x, y) for x, y in free_open if x0 <= x < x1 and y0 <= y < y1]
            if pool:
                open_survivors.append(random.choice(pool))
            if len(open_survivors) >= n_open:
                break
        if len(open_survivors) >= n_open:
            break

    self.survivors = building_survivors + open_survivors

M.FleetSim._build_survivors = _patched_build_survivors

# ── Patch gui_loop to accept an external seed ─────────────────────────────────
# gui_loop() hardcodes random.seed(0) — monkey-patch it to use our seed
_orig_gui_loop = M.gui_loop
def _patched_gui_loop(seed=3):
    import types
    # Temporarily replace FleetSim.__init__ to inject the seed
    _orig_init = M.FleetSim.__init__
    def _seeded_init(self_sim):
        random.seed(11); np.random.seed(4)
        _orig_init(self_sim)
    M.FleetSim.__init__ = _seeded_init
    try:
        _orig_gui_loop()
    finally:
        M.FleetSim.__init__ = _orig_init

# ── Launch ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    seed       = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    display_px = int(sys.argv[2]) if len(sys.argv) > 2 else 4

    M.CELL_SIZE = display_px
    w_px = LARGE_GRID_W * display_px
    h_px = LARGE_GRID_H * display_px

    print(f"Large-Scale Fleet Sim  (all performance patches active)")
    print(f"  Grid    : {LARGE_GRID_W}×{LARGE_GRID_H} @ {LARGE_CELL_SIZE}m = 1km²")
    print(f"  Robots  : {sum(LARGE_ROBOTS.values())} "
          f"({', '.join(f'{n} {t}' for t, n in LARGE_ROBOTS.items())})")
    print(f"  Battery : {LARGE_MAX_BATTERY}  CBBA_ITERS={M.CBBA_ITERS}  goal_commit=60")
    print(f"  Seed    : {seed}  Window: {w_px}×{h_px}px")
    print(f"\nBuilding world... (may take a few seconds)")
    print(f"Controls: SPACE=pause  S=step  P=paths  Q=quit\n")

    _patched_gui_loop(seed=seed)