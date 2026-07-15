"""
Large-Scale Five-Way Comparison Benchmark
==========================================
Compares: Fleet (Framework N, relay radius R=30) | GNF | GNF-Shadow | GNF-Risk |
          ACHORD-insp | ACHORD-Risk | CARA-2022 | CARA-EL-Base | CARA-EL-Dyn
(Greedy-Oracle retired: oracle terrain reads + hazard blindness made it a
pure casualty generator under the graded dose model — zero insight per run.
GNF-Risk replaces it: the same explorer, belief-only, risk-aware pathing.)
(The K-12/K-20/K-30 coverage-radius sweep is retired: R=30 was established as
best and is baked into the flagship 'Fleet' entry.)
on a 200×200 grid (1 km²) with 50 robots and 45 survivors.

Usage:
    python ComparisonBenchLarge.py [--steps N] [--seeds 0,1,2] [--out ./results]

Outputs
-------
  - Terminal table (per-seed + averaged summary)
  - PNG plots saved to --out directory
  - summary_large.json
"""

import sys, os, random, time, argparse, json, csv
from unittest.mock import MagicMock
from collections import defaultdict

_pg = MagicMock(); _pg.SRCALPHA = 0
for _m in ['pygame', 'pygame.display', 'pygame.font']:
    if _m not in sys.modules:
        sys.modules[_m] = _pg

import numpy as np
import importlib.util
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── File discovery ─────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))

_SEARCH_DIRS = [
    _HERE,
    os.path.join(_HERE, '..', 'outputs'),
    os.path.join(_HERE, '..'),
    '/mnt/user-data/outputs',
    '/home/claude',
]

def _find(*names):
    for name in names:
        for folder in _SEARCH_DIRS:
            p = os.path.join(folder, name)
            if os.path.exists(p):
                return os.path.abspath(p)
    print("\nERROR: Cannot find any of:", list(names))
    print("Searched folders:")
    for d in _SEARCH_DIRS:
        print("  " + d)
    sys.exit(1)

def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M); return M

# Load Framework J and Framework K as separate modules.
# J = bounded-relay baseline (the previous "Fleet"); K = walking-disk / disk-radius.
# Both share world construction so we keep M as an alias for the constants-patch
# target below; that section (grid size, cell size, etc.) applies to both.
# Single framework: Framework N (relay radius R=30 built in). The J-bounded
# predecessor is retired from the comparison; external baselines carry the
# cross-model story. Falls back to Framework M if N is absent.
M_K = _load(_find('2DFleetFrameworkN.py', '2DFleetFrameworkM.py'), 'fleet_N')
M   = M_K    # baselines and scenario patches all target this module
M_gnf  = _load(_find('gnf_sim.py', 'GNF_Sim.py', 'GNF_sim.py'),           'gnf')
# GNFShadowSim is defined in the same file — pull it out after loading
GNFShadowSim = getattr(M_gnf, 'GNFShadowSim', None)
M_cara = _load(_find('cara_sim.py', 'Cara_sim.py'),                                        'cara')
M_cara_paper = _load(_find('cara_paper.py', 'Cara_paper.py'),                              'cara_paper')
M_achord = _load(_find('ACHORD.py', 'Achord_sim.py', 'achord_sim.py'),                       'achord')
try:
    M_ritags = _load(_find('Ritags_sim.py', 'ritags_sim.py', 'RITAGS.py'), 'ritags')
except FileNotFoundError:
    M_ritags = None

# ── A* timing instrumentation ──────────────────────────────────────────────────
# In a real deployment the ground station runs the coordination (potential-game
# role selection, CBBA, waypoint assignment) and each robot plans its OWN path
# locally, on its own processor, in PARALLEL. Summing every robot's A* onto one
# CPU timeline is a simulation artifact, not a real central cost. We therefore
# split each step's wall time into:
#   • ground-station coordination  = total_step - sum(A* calls)   [central]
#   • per-robot onboard planning    = max single A* call this step [parallel/local]
# and report a realistic distributed per-step latency = ground + max-single-A*.
_ASTAR = {'sum': 0.0, 'max': 0.0, 'n': 0}
def _reset_astar():
    _ASTAR['sum'] = 0.0; _ASTAR['max'] = 0.0; _ASTAR['n'] = 0
def _wrap_astar(cls):
    _orig = cls.search
    def _timed(*a, **k):
        _t0 = time.perf_counter()
        _r = _orig(*a, **k)
        _dt = (time.perf_counter() - _t0) * 1000.0
        _ASTAR['sum'] += _dt
        if _dt > _ASTAR['max']: _ASTAR['max'] = _dt
        _ASTAR['n'] += 1
        return _r
    cls.search = staticmethod(_timed)
# Wrap every distinct AStar class across the loaded sims so the split is fair
# for all of them (shared classes are wrapped once via the id() guard).
_wrapped_astar = set()
for _mod in (M_K, M_gnf, M_cara):
    _ac = getattr(_mod, 'AStar', None)
    if _ac is not None and id(_ac) not in _wrapped_astar and hasattr(_ac, 'search'):
        _wrapped_astar.add(id(_ac))
        _wrap_astar(_ac)

# ── Large-scale configuration ──────────────────────────────────────────────────
LARGE_GRID_W      = 200
LARGE_GRID_H      = 200
LARGE_CELL_SIZE   = 5
LARGE_ZONE_CHUNKS = 10
LARGE_MAX_BATTERY = 2000
LARGE_MAX_BUNDLE  = 8
LARGE_N_SURVIVORS = 45
LARGE_ROBOTS = {"Legged": 13, "Drone": 17, "Boat": 8, "Rover": 12}

# ── Patch module constants BEFORE any sim is instantiated ──────────────────────
for _mod in (M_K,):
    _mod.GRID_W      = LARGE_GRID_W
    _mod.GRID_H      = LARGE_GRID_H
    _mod.CELL_SIZE   = LARGE_CELL_SIZE
    _mod.ZONE_CHUNKS = LARGE_ZONE_CHUNKS
    _mod.MAX_BATTERY = LARGE_MAX_BATTERY
    _mod.MAX_BUNDLE  = LARGE_MAX_BUNDLE

# Backward-compat aliases (existing scenario-patch code below writes M.X = ...)
M.GRID_W      = LARGE_GRID_W
M.GRID_H      = LARGE_GRID_H
M.CELL_SIZE   = LARGE_CELL_SIZE
M.ZONE_CHUNKS = LARGE_ZONE_CHUNKS
M.MAX_BATTERY = LARGE_MAX_BATTERY
M.MAX_BUNDLE  = LARGE_MAX_BUNDLE


# ── Wrap the M-patching block in a function so we can apply patches to
#    BOTH Framework J and Framework K without duplicating the code.
def _apply_large_scale_patches(M, patch_baselines=False):
    # ── Large-map scaling patches ──────────────────────────────────────────────────
    # At 200×200 a Rover needs ~100 ticks to reach a building border from the map
    # centre — twice the 128×128 travel time.  Scale relay hold and CARA MILP
    # cadence accordingly so relays are not replaced before they arrive.
    M.RELAY_MIN_HOLD = 300          # was 150 — doubles to cover 200×200 travel time
    if hasattr(M, 'ALLOCATION_CADENCE'):
        M.ALLOCATION_CADENCE = 100  # was 50 — must exceed relay travel time to border
    M.COOLDOWN_T  = 120             # was 40  — prevents zone-exhaustion cycling at 50 robots
    M.CBBA_ITERS  = 1               # was 3   — one consensus pass sufficient at 50 robots/100 zones

    # ── Performance Fix 1: increase goal_commit so robots hold paths 3× longer ────
    # At 200×200 cross-map paths are 100+ steps. Replanning every 20 ticks wastes
    # ~300ms per A* call on paths that are still valid.
    _orig_robot_init = M.Robot.__init__
    def _patched_robot_init(self, *args, **kwargs):
        _orig_robot_init(self, *args, **kwargs)
        self.goal_commit = 60   # was 20
    M.Robot.__init__ = _patched_robot_init

    # ── Performance Fix 2: share reachable arrays across same-capability robots ────
    # With 50 robots there are only 4 distinct caps_masks (Legged/Drone/Boat/Rover).
    # No need to run scipy.ndimage.label 50 times per tick when 4 calls suffice.
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

    # ── Performance Fix 3: lock CBBA cadence to 50 ticks minimum ──────────────────
    # The late-game 20-tick cadence doubles cost at 50 robots. Lock it at 50.
    _orig_step = M.FleetSim.step
    def _patched_step(self):
        if not hasattr(self, '_union_cov_cache') or self._union_cov_tick != self.timestep:
            self._union_cov_cache = float(np.mean(self.union_belief != M.T_UNKNOWN))
            self._union_cov_tick  = self.timestep
        _real_cov = self._union_cov_cache
        self._union_cov_cache = min(self._union_cov_cache, 0.69)  # prevent 20-tick cadence
        result = _orig_step(self)
        self._union_cov_cache = _real_cov
        return result
    M.FleetSim.step = _patched_step

    # ── Performance Fix 4: gate _decide_roles on a change signal ──────────────────
    # _decide_roles runs the full BR loop (O(R²×roles) utility evals) every tick.
    # Roles are in Nash equilibrium between events — only re-run when something
    # actually changed: relay_ok, robot death, CBBA fired, or building opened.
    # Safety net: always re-run at least every 15 ticks.
    # Expected: 10-15x reduction in BR calls on complex seeds.
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
        # Safety net: 5 ticks (not 15) — idle robots need reassignment promptly
        if not changed and (t - last_run) < 5:
            return
        self._pr_relay_sig = relay_sig
        self._pr_n_active  = n_active
        self._pr_cbba_tick = cbba_tick
        self._pr_stair_sig = stair_sig
        self._pr_last_run  = t
        _orig_decide_roles(self)

    M.FleetSim._decide_roles = _patched_decide_roles

    # ── Patch FleetSim._build_robots — large robot counts and spawn clusters ───────
    def _patched_build_robots(self):
        # Graded hazard durability (shared by EVERY model in the comparison —
        # substrate parity with Framework M's ROBOT_HAZARD_PROFILE). Falls
        # back to the legacy flat limits for older framework builds.
        _hp = getattr(M, 'ROBOT_HAZARD_PROFILE', None) or {
            'Legged': dict(temp_limit=M.TEMP_LIMIT, rad_limit=M.RAD_LIMIT),
            'Drone':  dict(temp_limit=M.TEMP_LIMIT, rad_limit=M.RAD_LIMIT),
            'Boat':   dict(temp_limit=9999., rad_limit=9999.),
            'Rover':  dict(temp_limit=9999., rad_limit=9999.),
        }
        templates = [
            ("Legged", {M.Capability.LAND, M.Capability.STAIRS},
             np.array([10., 10.]), (_hp['Legged']['temp_limit'], _hp['Legged']['rad_limit'])),
            ("Drone",  {M.Capability.AIR},
             np.array([10., 10.]), (_hp['Drone']['temp_limit'], _hp['Drone']['rad_limit'])),
            ("Boat",   {M.Capability.WATER},
             np.array([0., 0.]),   (_hp['Boat']['temp_limit'], _hp['Boat']['rad_limit'])),
            ("Rover",  {M.Capability.LAND},
             np.array([-2., -2.]), (_hp['Rover']['temp_limit'], _hp['Rover']['rad_limit'])),
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
            name = f"{tname}{i}"
            center = clusters[i % len(clusters)]
            qx = 0 if center[0] < W//2 else 1
            qy = 0 if center[1] < H//2 else 1

            if tname == "Boat" and water_cells:
                pool = water_by_quad.get((qx, qy), []) or water_cells
                non_shadow_quad   = [c for c in pool       if not self.radio_shadow[c[0], c[1]]]
                non_shadow_global = [c for c in water_cells if not self.radio_shadow[c[0], c[1]]]
                chosen_pool = non_shadow_quad or non_shadow_global or pool
                sx, sy = random.choice(chosen_pool)
            else:
                sx, sy = center
                for _ in range(50):
                    cx = max(1, min(W-2, center[0] + random.randint(-12, 12)))
                    cy = max(1, min(H-2, center[1] + random.randint(-12, 12)))
                    tt = self.world.grid[cx][cy]["t"]
                    if (tt == M.T_FREE or (tt == M.T_STAIRS and M.Capability.STAIRS in caps)) \
                            and not self.radio_shadow[cx, cy]:
                        sx, sy = cx, cy; break
            self.robots.append(
                M.Robot(name, sx, sy, caps, self.world, self, weights, tlim, rlim))

    M.FleetSim._build_robots = _patched_build_robots

    # ── Shared large-scale survivor placement — patched onto Fleet AND GNF ─────────
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

        # ceil (not round) so the grid tiles ENOUGH cells for n_open. round() gave
        # grid_k=5 for n_open=27 -> a 5x5=25-cell grid -> at most 25 open survivors,
        # so 2 were silently dropped (18 building + 25 open = 43): the 43-vs-45 bug.
        grid_k = max(1, int(np.ceil(n_open ** 0.5)))
        cell_w = max(1, W // grid_k); cell_h = max(1, H // grid_k)
        open_survivors = []; _used = set(building_survivors)
        for gx in range(grid_k):
            for gy in range(grid_k):
                x0 = gx * cell_w; x1 = min(W, x0 + cell_w)
                y0 = gy * cell_h; y1 = min(H, y0 + cell_h)
                pool = [(x, y) for x, y in free_open
                        if x0 <= x < x1 and y0 <= y < y1 and (x, y) not in _used]
                if pool:
                    c = random.choice(pool); open_survivors.append(c); _used.add(c)
                if len(open_survivors) >= n_open:
                    break
            if len(open_survivors) >= n_open:
                break

        survivors = building_survivors + open_survivors
        # Guarantee EXACTLY n_total survivors are placed so the reported denominator
        # is real — fill any residual shortfall from unused free cells.
        if len(survivors) < n_total:
            _used = set(survivors)
            spare = [c for c in free_open if c not in _used]
            random.shuffle(spare)
            survivors += spare[:n_total - len(survivors)]
        self.survivors = survivors[:n_total]

    M.FleetSim._build_survivors = _patched_build_survivors
    if patch_baselines:
        M_gnf.GNFSim._build_survivors = _patched_build_survivors
        if GNFShadowSim is not None:
            GNFShadowSim._build_survivors = _patched_build_survivors

# Apply to both frameworks; baselines patched only on the first pass.
_apply_large_scale_patches(M_K, patch_baselines=True)

# ── Build sim classes (after patching) ────────────────────────────────────────
FleetKSim   = M_K.FleetSim
GNFSim      = M_gnf.GNFSim

try:
    CARABase    = M_cara.make_cara_sim(M_gnf, M, use_exec_layer=False)
    CARADynamic = M_cara.make_cara_sim(M_gnf, M, use_exec_layer=True)
    CARAPaper   = M_cara_paper.make_cara_paper_sim(M_gnf, M)
    ACHORDSim   = M_achord.make_achord_sim(M_gnf, M)
    ACHORDRisk  = (M_achord.make_achord_risk_sim(M_gnf, M)
                   if hasattr(M_achord, 'make_achord_risk_sim') else None)
    RitagsSim   = (M_ritags.make_ritags_sim(M_gnf, M)
                   if M_ritags is not None else None)
except TypeError:
    print("  [cara_sim.py: use_exec_layer not supported — update for Base vs Dynamic]")
    CARABase    = M_cara.make_cara_sim(M_gnf, M)
    CARADynamic = M_cara.make_cara_sim(M_gnf, M)

# ── K variants: same sim class, different coverage radius ───────────────────
# K uses module-level constant RELAY_COVERAGE_RADIUS_CELLS. We set it
# immediately before instantiation, so each K-N sim gets its own radius.
def _make_k_factory(R):
    def factory():
        M_K.RELAY_COVERAGE_RADIUS_CELLS = R
        return FleetKSim()
    return factory

# Coverage-radius sweep concluded: R=30 won (see the K-12/20/30 study data).
# The flagship 'Fleet' entry IS Framework N at R=30; the sweep entries are
# retired so runtime goes to the cross-model comparison instead.
SIMS = [
    ('Fleet',        _make_k_factory(30),          '#08306b', '-'),
    ('GNF',          lambda: GNFSim(M),            '#d6604d', '--'),
    ('GNF-Shadow',   lambda: GNFShadowSim(M),      '#a50026', ':'),
    ('GNF-Risk',     lambda: M_gnf.GNFRiskSim(M),   '#4dac26', '-.'),
    ('ACHORD-insp',  lambda: ACHORDSim(M),          '#8c564b', (0,(4,1))),
    ('ACHORD-Risk',  lambda: ACHORDRisk(M),         '#e377c2', (0,(4,2))),
    ('RITAGS-insp',  lambda: RitagsSim(M),          '#8c6d31', (0,(3,1,1,1))),
    ('CARA-2022',    lambda: CARAPaper(M),          '#17becf', (0,(1,1))),
    ('CARA-EL-Base', lambda: CARABase(M),           '#984ea3', (0,(5,2))),
    ('CARA-EL-Dyn',  lambda: CARADynamic(M),        '#ff7f00', (0,(3,1,1,1))),
]
if GNFShadowSim is None:
    print("  [WARNING] GNFShadowSim not found in gnf_sim.py — skipping GNF-Shadow")
    SIMS = [s for s in SIMS if s[0] != 'GNF-Shadow']
if ACHORDRisk is None:
    print("  [WARNING] make_achord_risk_sim not found in Achord_sim.py — skipping ACHORD-Risk")
    SIMS = [s for s in SIMS if s[0] != 'ACHORD-Risk']
if RitagsSim is None:
    print("  [WARNING] Ritags_sim.py not found — skipping RITAGS-insp")
    SIMS = [s for s in SIMS if s[0] != 'RITAGS-insp']

# Sims that get per-phase compute-breakdown instrumentation (the flagship) and
K_SIM_NAMES = {'Fleet'}

RECORD_EVERY = 25
BAR_WIDTH    = 32

def _progress(done, total, prefix='', suffix=''):
    filled = int(BAR_WIDTH * done / max(total, 1))
    bar    = '#' * filled + '-' * (BAR_WIDTH - filled)
    pct    = 100.0 * done / max(total, 1)
    sys.stdout.write(f'\r  {prefix}[{bar}] {pct:5.1f}%  {suffix:<35s}')
    sys.stdout.flush()

def _clear():
    sys.stdout.write('\r' + ' ' * 90 + '\r')
    sys.stdout.flush()

# ── Metric helpers ─────────────────────────────────────────────────────────────
def _stair_mask(sim):
    if hasattr(sim, '_world_stair_arr'): return sim._world_stair_arr
    sm = np.zeros((M.GRID_W, M.GRID_H), dtype=bool)
    for x in range(M.GRID_W):
        for y in range(M.GRID_H):
            if sim.world.grid[x][y]['t'] == M.T_STAIRS: sm[x, y] = True
    sim._world_stair_arr = sm; return sm

def _stair_cov(sim):
    sm = _stair_mask(sim); ns = int(np.sum(sm))
    if ns == 0: return 0.0
    return (ns - int(np.sum((sim.union_belief == M.T_UNKNOWN) & sm))) / ns * 100

def _relay_covered(sim, r):
    if hasattr(sim, '_relay_ok'): return bool(sim._relay_ok[r.pos[0], r.pos[1]])
    z = sim.cell_to_zone(r.pos[0], r.pos[1])
    return bool(sim.relay_ok_extended(z)) if z else False

def _buildings_entered(sim):
    sm = _stair_mask(sim)
    from scipy.ndimage import label as _lbl
    lab, n = _lbl(sm); entered = 0
    for i in range(1, n + 1):
        bld = (lab == i)
        if np.any(sim.union_belief[bld] != M.T_UNKNOWN): entered += 1
    return entered, n

def _redundancy(sim):
    active = [r for r in sim.robots if r.active]
    if not active: return 0.0
    pos_counts = defaultdict(int)
    for r in active: pos_counts[r.pos] += 1
    return sum(1 for r in active if pos_counts[r.pos] > 1) / len(active)

def _zone_redundancy(sim):
    active = [r for r in sim.robots if r.active]
    if not active: return 0.0
    zone_counts = defaultdict(int)
    for r in active:
        z = getattr(r, 'task_zone', None)
        if z is None:
            z = sim.cell_to_zone(r.pos[0], r.pos[1])
        if z is not None:
            zone_counts[z] += 1
    redundant = sum(1 for r in active
                    for z in [getattr(r, 'task_zone', None) or sim.cell_to_zone(r.pos[0], r.pos[1])]
                    if z and zone_counts[z] > 1)
    return redundant / len(active)

# ── Result container ───────────────────────────────────────────────────────────
class Result:
    def __init__(self, name, seed):
        self.name = name; self.seed = seed
        self.total_survivors = 0; self.buildings_total = 0
        self.completed = False; self.completion_step = None
        self.found_ts = []; self.cov_ts = []; self.stair_ts = []
        self.bldg_ts = []; self.redundancy_ts = []; self.zone_redundancy_ts = []
        self.step_ms_ts = []; self.flipflop_ts = []; self.idle_move_ts = []
        # compute split: central coordination vs distributed onboard planning
        self.ground_ms_ts = []   # ground-station coordination (non-A*) per step
        self.astar_max_ts = []   # max single A* call per step (per-robot parallel)
        self.dist_ms_ts   = []   # realistic distributed latency = ground + max-A*
        self.ground_avg = 0.0; self.ground_p50 = 0.0; self.ground_p90 = 0.0
        self.astar_max_avg = 0.0; self.dist_avg = 0.0; self.dist_p90 = 0.0
        self.milp_ms_ts = []
        self.final_found = 0; self.final_cov = 0.0; self.final_stair = 0.0
        self.deaths = 0; self.hazard_deaths = 0; self.battery_deaths = 0
        self.comms_deaths = 0                  # new: comms-loss deaths
        self.viol = 0; self.trapped = 0
        self.p50 = 0.0; self.p90 = 0.0; self.avg_ms = 0.0
        self.milp_avg = 0.0; self.milp_max = 0.0; self.milp_n = 0
        self.hold = 0; self.ejects = 0
        self.wall_s = 0.0
        self.stalled = False
        # ── New J-vs-K metrics ────────────────────────────────────────────
        self.peak_relays = 0                   # max concurrent relays
        self.relays_ts = []                    # (step, active_relay_count)
        self.t_first_found = -1                # tick when first survivor found
        self.t_half_found  = -1                # tick when N/2 survivors found
        self.t_all_found   = -1                # tick when all survivors found
        # ── K compute breakdown (K variants only, else empty) ─────────────
        # Seconds accumulated per phase across the whole run
        self.compute_breakdown = {}            # {'role', 'alloc', 'cover', 'other'}

# ── Runner ─────────────────────────────────────────────────────────────────────
STALL_WINDOW = 600   # was 300 — doubled for 200×200 map (relay cycle ~200 ticks)

def run_sim(sim_name, factory, seed, steps):
    random.seed(seed); np.random.seed(seed)
    sim = factory()
    res = Result(sim_name, seed)
    res.total_survivors = len(sim.survivors)
    _, res.buildings_total = _buildings_entered(sim)

    # ── K compute-breakdown instrumentation ────────────────────────────────
    # For K variants only, wrap the phase methods to accumulate wallclock
    # time per phase. Wrappers are cheap (perf_counter + dict add) and add
    # negligible cost themselves. Non-K sims: unchanged.
    breakdown = {'role': 0.0, 'alloc': 0.0, 'cover': 0.0, 'other': 0.0}
    if sim_name in K_SIM_NAMES:
        def _timed(method, key):
            def wrapped(*a, **k):
                t0 = time.perf_counter()
                try:    return method(*a, **k)
                finally: breakdown[key] += time.perf_counter() - t0
            return wrapped
        if hasattr(sim, '_pg_best_response_roles'):
            sim._pg_best_response_roles = _timed(sim._pg_best_response_roles, 'role')
        if hasattr(sim, '_assign_zones_cbba'):
            sim._assign_zones_cbba = _timed(sim._assign_zones_cbba, 'alloc')
        if hasattr(sim, '_active_relay_coverage_union'):
            sim._active_relay_coverage_union = _timed(sim._active_relay_coverage_union, 'cover')
        if hasattr(sim, '_disk_relay_coverage'):
            sim._disk_relay_coverage = _timed(sim._disk_relay_coverage, 'cover')

    step_acc = []; trapped = 0; prev_milp_n = 0
    ground_acc = []; astar_max_acc = []; dist_acc = []
    t_wall = time.time()
    from collections import deque as _deque
    _pos_hist = {r.name: _deque(maxlen=5) for r in sim.robots}
    _prev_union_count = int(np.sum(sim.union_belief != M.T_UNKNOWN))
    _prev_pos = {r.name: r.pos for r in sim.robots}
    _last_found_step = 0
    # Which module owns this sim's Role enum? Fall back to M for baselines.
    _mod_for_role = M
    _Role = getattr(_mod_for_role, 'Role', None)

    for step in range(1, steps + 1):
        t0 = time.perf_counter()
        _reset_astar()
        sim.step()
        ms = (time.perf_counter() - t0) * 1000
        step_acc.append(ms)
        # split this step into central coordination vs parallel onboard planning
        _astar_sum = _ASTAR['sum']; _astar_max = _ASTAR['max']
        _ground = max(0.0, ms - _astar_sum)        # ground station (non-A*)
        ground_acc.append(_ground)
        astar_max_acc.append(_astar_max)            # slowest single robot's plan
        dist_acc.append(_ground + _astar_max)       # realistic distributed latency

        active_robots = [r for r in sim.robots if r.active]

        # Flip-flop detection
        flip_count = 0
        for r in active_robots:
            hist = _pos_hist[r.name]
            if r.pos in hist: flip_count += 1
            hist.append(r.pos)
        flip_pct = flip_count / max(len(active_robots), 1) * 100

        # Idle movement
        new_union_count = int(np.sum(sim.union_belief != M.T_UNKNOWN))
        new_cells_this_step = new_union_count - _prev_union_count
        moved_robots = [r for r in active_robots if r.pos != _prev_pos.get(r.name)]
        idle_count = len(moved_robots) if new_cells_this_step == 0 and moved_robots else 0
        idle_pct = idle_count / max(len(active_robots), 1) * 100
        _prev_union_count = new_union_count
        _prev_pos = {r.name: r.pos for r in active_robots}

        # Trapped count
        for r in sim.robots:
            if r.active and sim.radio_shadow[r.pos[0], r.pos[1]] and not _relay_covered(sim, r):
                trapped += 1

        # CARA MILP timing
        if hasattr(sim, 'milp_solve_times'):
            cur_n = len(sim.milp_solve_times)
            if cur_n > prev_milp_n:
                for mt in sim.milp_solve_times[prev_milp_n:]:
                    res.milp_ms_ts.append((step, mt))
                prev_milp_n = cur_n

        # Snapshots
        if step % RECORD_EVERY == 0 or step == 1:
            cov = float(np.mean(sim.union_belief != M.T_UNKNOWN)) * 100
            ent, _ = _buildings_entered(sim)
            res.found_ts.append((step, len(sim.found)))
            res.cov_ts.append((step, cov))
            res.stair_ts.append((step, _stair_cov(sim)))
            res.bldg_ts.append((step, ent))
            res.redundancy_ts.append((step, _redundancy(sim) * 100))
            res.zone_redundancy_ts.append((step, _zone_redundancy(sim) * 100))
            res.flipflop_ts.append((step, flip_pct))
            res.idle_move_ts.append((step, idle_pct))
            res.step_ms_ts.append((step, float(np.mean(step_acc))))
            res.ground_ms_ts.append((step, float(np.mean(ground_acc))))
            res.astar_max_ts.append((step, float(np.mean(astar_max_acc))))
            res.dist_ms_ts.append((step, float(np.mean(dist_acc))))
            # Track concurrent relay count (0 for baselines lacking Role.RELAY)
            n_relays = 0
            if _Role is not None:
                n_relays = sum(1 for r in sim.robots
                               if r.active and getattr(r, 'role', None) == _Role.RELAY)
            res.relays_ts.append((step, n_relays))
            res.peak_relays = max(res.peak_relays, n_relays)
            step_acc = []; ground_acc = []; astar_max_acc = []; dist_acc = []

        # Time-to-milestone tracking (per-step so we catch the exact tick)
        n_found = len(sim.found)
        if n_found >= 1 and res.t_first_found < 0:
            res.t_first_found = step
        if n_found >= res.total_survivors // 2 and res.t_half_found < 0:
            res.t_half_found = step
        if n_found >= res.total_survivors and res.t_all_found < 0:
            res.t_all_found = step

        if n_found > 0 and (not res.found_ts or n_found > res.found_ts[-1][1]):
            _last_found_step = step

        if n_found >= res.total_survivors and not res.completed:
            res.completed = True; res.completion_step = step
            break

        # Exit if no active robots remain
        if not active_robots:
            break

        # Exit if stalled — no new survivors for STALL_WINDOW steps
        if step - _last_found_step > STALL_WINDOW and step > STALL_WINDOW:
            res.stalled = True
            break

        if step % 10 == 0 or step == steps:
            elapsed = time.time() - t_wall
            eta     = (steps - step) / max(step, 1) * elapsed
            _progress(step, steps,
                      prefix=f'{sim_name:<14} ',
                      suffix=f'step {step}/{steps}  ETA {int(eta//60):02d}:{int(eta%60):02d}')

    _clear()
    res.wall_s = time.time() - t_wall
    all_ms = [ms for _, ms in res.step_ms_ts]
    st = sorted(all_ms) if all_ms else [0]
    # compute-split aggregates
    g = sorted(v for _, v in res.ground_ms_ts) or [0]
    res.ground_avg = sum(g)/len(g); res.ground_p50 = g[len(g)//2]; res.ground_p90 = g[int(len(g)*0.9)]
    am = [v for _, v in res.astar_max_ts] or [0]
    res.astar_max_avg = sum(am)/len(am)
    d = sorted(v for _, v in res.dist_ms_ts) or [0]
    res.dist_avg = sum(d)/len(d); res.dist_p90 = d[int(len(d)*0.9)]

    res.final_found  = len(sim.found)
    res.final_cov    = float(np.mean(sim.union_belief != M.T_UNKNOWN)) * 100
    res.final_stair  = _stair_cov(sim)
    res.deaths         = sum(1 for r in sim.robots if not r.active)
    res.hazard_deaths  = sum(1 for r in sim.robots if not r.active and getattr(r, 'hazard_killed', False))
    # comms deaths: dead with death_reason containing 'comms' or 'signal' or 'stranded'
    res.comms_deaths = 0
    for r in sim.robots:
        if r.active: continue
        reason = str(getattr(r, 'death_reason', '') or '').lower()
        if any(k in reason for k in ('comms', 'signal', 'stranded', 'lost')):
            res.comms_deaths += 1
    res.battery_deaths = res.deaths - res.hazard_deaths - res.comms_deaths
    # Finalise K compute breakdown: 'other' = wallclock - sum(categorised)
    if sim_name in K_SIM_NAMES:
        wall = time.time() - t_wall
        categorised = breakdown['role'] + breakdown['alloc'] + breakdown['cover']
        breakdown['other'] = max(0.0, wall - categorised)
        res.compute_breakdown = dict(breakdown)
    res.viol    = sum(1 for r in sim.robots if r.active
                      and sim.radio_shadow[r.pos[0], r.pos[1]]
                      and not _relay_covered(sim, r))
    res.trapped = trapped
    n = len(st)
    res.p50 = st[n//2]; res.p90 = st[int(n * 0.9)]; res.avg_ms = sum(st) / n

    if hasattr(sim, 'milp_solve_times') and sim.milp_solve_times:
        mt = sim.milp_solve_times
        res.milp_avg = sum(mt)/len(mt); res.milp_max = max(mt); res.milp_n = len(mt)
    if hasattr(sim, 'hold_ticks'):  res.hold   = sum(sim.hold_ticks.values())
    if hasattr(sim, 'eject_events'): res.ejects = len(sim.eject_events)
    if hasattr(sim, 'relays_placed'): res.relays_placed = len(sim.relays_placed)
    if hasattr(sim, 'radios_dropped'): res.radios_dropped = len(sim.radios_dropped)
    if hasattr(sim, 'conn_ratio'): res.conn_ratio = float(sim.conn_ratio)
    if hasattr(sim, 'sprt_stats'):
        res.sprt_accepted = sim.sprt_stats.get('accepted', 0)
        res.sprt_redundancy = sim.sprt_stats.get('redundancy_adds', 0)
    return res

# ── Plotting helpers ───────────────────────────────────────────────────────────
ALPHA_BAND = 0.15

def _mean_std(results_by_name, name, getter):
    series = [getter(r) for r in results_by_name[name]]
    series = [s for s in series if s]   # drop empty
    if not series: return np.array([]), np.array([]), np.array([])
    # Use the longest series as the x-axis reference
    ref = max(series, key=len)
    xs  = np.array([s for s, _ in ref])
    # Pad shorter series by repeating their last value so all rows match length
    rows = []
    for s in series:
        vals = [v for _, v in s]
        if len(vals) < len(xs):
            vals = vals + [vals[-1]] * (len(xs) - len(vals))
        rows.append(vals[:len(xs)])
    arr = np.array(rows, dtype=float)
    return xs, arr.mean(0), arr.std(0)

def _plot_metric(ax, results_by_name, getter, ylabel, title, steps, ylim=None):
    for name, _, col, ls in SIMS:
        xs, m, s = _mean_std(results_by_name, name, getter)
        if not len(xs): continue
        ax.plot(xs, m, color=col, lw=2, ls=ls, label=name)
        ax.fill_between(xs, m-s, m+s, alpha=ALPHA_BAND, color=col)
    ax.set_xlabel('Timestep'); ax.set_ylabel(ylabel)
    ax.set_title(title); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_xlim(0, steps)
    if ylim: ax.set_ylim(*ylim)

def _save(fig, path, name):
    fig.tight_layout()
    fig.savefig(os.path.join(path, name), dpi=130)
    plt.close(fig)
    return name

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=3000)
    ap.add_argument('--seeds', type=str, default='21')
    ap.add_argument('--out',   type=str, default=None)
    ap.add_argument('--sims',  type=str, default='all',
                    help="comma list of sim names to run (e.g. 'GNF,CARA-2022'); "
                         "'all' runs the full suite. Note: the CARA-EL summary "
                         "block is skipped automatically when those sims are "
                         "filtered out.")
    args = ap.parse_args()
    global SIMS
    if args.sims != 'all':
        want = {s.strip() for s in args.sims.split(',')}
        known = {name for name, _, _, _ in SIMS}
        unknown = want - known
        if unknown:
            print('unknown sims:', sorted(unknown), '\nvalid:', sorted(known)); sys.exit(1)
        SIMS = [s for s in SIMS if s[0] in want]
    seeds = [int(s) for s in args.seeds.split(',')]
    steps = args.steps
    out_dir = args.out or os.path.join(_HERE, 'benchmark_large_comparison_RiTag')
    os.makedirs(out_dir, exist_ok=True)

    N_ROBOTS = sum(LARGE_ROBOTS.values())
    results_by_name = {name: [] for name, _, _, _ in SIMS}
    accs = {name: defaultdict(float) for name, _, _, _ in SIMS}
    _bench_t0 = time.time()
    def _fmt_hms(s):
        s = int(s); return f"{s//3600}:{(s%3600)//60:02d}:{s%60:02d}" 

    W = 128
    print(f"\n{'='*W}")
    print(f"  Large-Scale Five-Way Benchmark  —  "
          f"{LARGE_GRID_W}×{LARGE_GRID_H} grid @ {LARGE_CELL_SIZE}m  |  "
          f"{N_ROBOTS} robots  |  ~{LARGE_N_SURVIVORS} survivors")
    print(f"  Fleet(R=30) | GNF | GNF-Shadow | GNF-Risk | ACHORD-insp | ACHORD-Risk | RITAGS-insp | CARA-2022 | CARA-EL-Base | CARA-EL-Dyn")
    print(f"  {steps} steps  |  seeds: {seeds}  |  plots -> {out_dir}")
    print(f"{'='*W}")

    for seed in seeds:
        print(f"\n  seed={seed}:")
        print(f"  {'─'*124}")
        for run_idx, (name, factory, _, _) in enumerate(SIMS):
            overall = seeds.index(seed) * len(SIMS) + run_idx
            total_runs = len(seeds) * len(SIMS)
            _progress(overall, total_runs,
                      prefix='Overall  ',
                      suffix=f'seed={seed}  starting {name}...')
            res = run_sim(name, factory, seed, steps)
            results_by_name[name].append(res)
            is_cara = res.milp_n > 0
            stall_tag = ' [STALLED]' if res.stalled else ''
            base = (f"  {name:<14}"
                    f"  cov={res.final_cov:5.1f}%"
                    f"  stair={res.final_stair:5.1f}%"
                    f"  found={res.final_found}/{res.total_survivors}"
                    f"  deaths={res.deaths}(haz={res.hazard_deaths}/bat={res.battery_deaths})"
                    f"  trapped={res.trapped}"
                    f"  P50={res.p50:5.1f}ms  P90={res.p90:6.1f}ms  avg={res.avg_ms:5.1f}ms"
                    f"  wall={_fmt_hms(res.wall_s)}"
                    f"{stall_tag}")
            if is_cara:
                base += (f"  | milp_avg={res.milp_avg:.0f}ms"
                         f"  milp_max={res.milp_max:.0f}ms"
                         f"  n={res.milp_n}"
                         f"  hold={res.hold}  ejects={res.ejects}")
            print(base)
            for k in ('final_cov','final_stair','final_found','deaths','hazard_deaths','battery_deaths','comms_deaths','trapped',
                      'p50','p90','avg_ms','milp_avg','milp_max','milp_n','hold','ejects',
                      'ground_avg','ground_p50','ground_p90','astar_max_avg','dist_avg','dist_p90',
                      'peak_relays','relays_placed','radios_dropped',
                      'conn_ratio','sprt_accepted','sprt_redundancy'):
                accs[name][k] += getattr(res, k, 0)
            # Time-to-milestone: -1 sentinel means "never reached"; only average
            # over seeds where it was reached, else record 0.
            for k in ('t_first_found', 't_half_found', 't_all_found'):
                v = getattr(res, k, -1)
                if v is not None and v >= 0:
                    accs[name].setdefault(f'{k}_sum', 0)
                    accs[name].setdefault(f'{k}_n', 0)
                    accs[name][f'{k}_sum'] += v
                    accs[name][f'{k}_n'] += 1

        # ── Elapsed / overnight projection ─────────────────────────────────
        _elapsed = time.time() - _bench_t0
        _seeds_done = seeds.index(seed) + 1
        _per_seed = _elapsed / _seeds_done
        print(f"  elapsed {_fmt_hms(_elapsed)}  |  {_fmt_hms(_per_seed)}/seed"
              f"  ->  ~{int(8*3600/_per_seed)} seeds / 8h night,"
              f" ~{int(12*3600/_per_seed)} / 12h"
              f"  |  this run ends in ~{_fmt_hms(_per_seed*(len(seeds)-_seeds_done))}")
        print(f"  {'─'*124}")
        _ref_name = 'Fleet' if 'Fleet' in results_by_name else None
        if _ref_name is not None and results_by_name[_ref_name]:
            fm = results_by_name[_ref_name][-1]
            for name, _, _, _ in SIMS:
                if name == _ref_name: continue
                if name not in results_by_name or not results_by_name[name]: continue
                om = results_by_name[name][-1]
                dc = fm.final_cov   - om.final_cov
                ds = fm.final_stair - om.final_stair
                df = fm.final_found - om.final_found
                dt = om.avg_ms      - fm.avg_ms
                print(f"  {_ref_name} vs {name:<14}"
                      f"  cov={dc:+.1f}%  stair={ds:+.1f}%  found={df:+d}"
                      f"  speed={dt:+.1f}ms/step (+ = {_ref_name} faster)")

    # ── Averaged summary ───────────────────────────────────────────────────────
    n_seeds = len(seeds)
    # Exclude ONLY the per-milestone accumulator helpers (t_*_found_sum / _n),
    # which are handled separately below. Do NOT use a blanket endswith('_n')
    # filter — that also drops milp_n, which the CARA summary print needs.
    _milestone_helpers = {
        f'{k}_{suffix}'
        for k in ('t_first_found', 't_half_found', 't_all_found')
        for suffix in ('sum', 'n')
    }
    avgs = {name: {k: v/n_seeds for k, v in accs[name].items()
                    if k not in _milestone_helpers}
             for name in accs}
    # Time-to-milestone: mean of seeds that reached it (0 if none did)
    for name in accs:
        for k in ('t_first_found', 't_half_found', 't_all_found'):
            s = accs[name].get(f'{k}_sum', 0); c = accs[name].get(f'{k}_n', 0)
            avgs[name][k] = (s / c) if c > 0 else 0.0
    n_surv = LARGE_N_SURVIVORS

    if n_seeds > 1:
        print(f"\n{'='*W}")
        print(f"  Averages over {n_seeds} seeds:")
        print(f"  {'─'*124}")
        for name, _, _, _ in SIMS:
            d = avgs[name]; is_cara = d['milp_n'] > 0
            base = (f"  {name:<14}"
                    f"  cov={d['final_cov']:5.1f}%"
                    f"  stair={d['final_stair']:5.1f}%"
                    f"  found={d['final_found']:.1f}/{n_surv}"
                    f"  deaths={d['deaths']:.1f}(haz={d['hazard_deaths']:.1f}/bat={d['battery_deaths']:.1f})"
                    f"  trapped={d['trapped']:.0f}"
                    f"  avg={d['avg_ms']:5.1f}ms")
            if is_cara:
                base += (f"  | milp_avg={d['milp_avg']:.0f}ms"
                         f"  hold={d['hold']:.0f}  ejects={d['ejects']:.0f}")
            print(base)

    print(f"\n{'='*W}")
    print(f"  Summary  ({steps} steps, {n_seeds} seed{'s' if n_seeds>1 else ''},"
          f" {LARGE_GRID_W}×{LARGE_GRID_H} grid, {N_ROBOTS} robots)")
    hdr = (f"  {'Sim':<14}  {'Cov%':>6}  {'Stair%':>7}  {'Found':>6}"
           f"  {'Deaths':>7}  {'Trapped':>8}  {'Avg ms':>8}"
           f"  {'MILP avg':>9}  {'Hold':>6}  {'Ejects':>7}")
    print(f"  {'─'*124}\n{hdr}\n  {'─'*124}")
    for name, _, _, _ in SIMS:
        d = avgs[name]
        ms   = f"{d['milp_avg']:>8.0f}" if d['milp_avg'] > 0 else f"{'—':>8}"
        hold = f"{d['hold']:>6.0f}"     if d['hold']     > 0 else f"{'—':>6}"
        ej   = f"{d['ejects']:>7.0f}"   if d['ejects']   > 0 else f"{'—':>7}"
        print(f"  {name:<14}  {d['final_cov']:6.1f}  {d['final_stair']:7.1f}"
              f"  {d['final_found']:6.1f}  {d['deaths']:5.1f}(h={d['hazard_deaths']:.1f}/b={d['battery_deaths']:.1f})"
              f"  {d['trapped']:8.0f}  {d['avg_ms']:8.1f}"
              f"  {ms}  {hold}  {ej}")
    print(f"  {'─'*124}")
    print(f"  Compute split (ms/step) — the 'Centralized' column SUMS all robots' A* onto one")
    print(f"  timeline (simulation artifact). In deployment, path planning runs locally on each")
    print(f"  robot in parallel, so the real central load is 'Ground Stn' and a step's wall-clock")
    print(f"  latency is 'Distributed' = ground-station coordination + slowest single onboard plan.")
    print(f"  {'Sim':<14}  {'Centralized':>11}  {'GroundStn avg':>13}  {'GroundStn p90':>13}"
          f"  {'Onboard A* avg':>14}  {'Distrib avg':>11}  {'Distrib p90':>11}")
    print(f"  {'─'*124}")
    for name, _, _, _ in SIMS:
        d = avgs[name]
        print(f"  {name:<14}  {d['avg_ms']:11.1f}  {d['ground_avg']:13.1f}  {d['ground_p90']:13.1f}"
              f"  {d['astar_max_avg']:14.1f}  {d['dist_avg']:11.1f}  {d['dist_p90']:11.1f}")
    print(f"{'='*W}\n")

    # ── Save JSON ──────────────────────────────────────────────────────────────
    summary = {name: {k: round(float(v), 2) for k, v in avgs[name].items()}
               for name in avgs}
    summary['_config'] = {
        'grid': f'{LARGE_GRID_W}x{LARGE_GRID_H}',
        'cell_m': LARGE_CELL_SIZE,
        'n_robots': N_ROBOTS,
        'n_survivors': LARGE_N_SURVIVORS,
        'steps': steps,
        'seeds': seeds,
    }
    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # ── CSV export for later comparison ────────────────────────────────────────
    # Metric columns, in a stable order, shared by the per-run and summary CSVs.
    _CSV_METRICS = [
        'final_cov', 'final_stair', 'final_found', 'total_survivors',
        'completed', 'completion_step', 'stalled',
        'deaths', 'hazard_deaths', 'battery_deaths', 'comms_deaths',
        'trapped', 'peak_relays', 'relays_placed', 'radios_dropped',
        't_first_found', 't_half_found', 't_all_found',
        'p50', 'p90', 'avg_ms', 'ground_avg', 'ground_p90',
        'astar_max_avg', 'dist_avg', 'dist_p90',
        'milp_avg', 'milp_max', 'milp_n', 'hold', 'ejects', 'wall_s',
        'conn_ratio', 'sprt_accepted', 'sprt_redundancy',
    ]

    # (1) per-run: one row per (sim, seed) — the raw material for stats later.
    with open(os.path.join(out_dir, 'runs.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['sim', 'seed'] + _CSV_METRICS)
        for name, _, _, _ in SIMS:
            for r in sorted(results_by_name[name], key=lambda r: r.seed):
                w.writerow([name, r.seed] +
                           [getattr(r, m, '') for m in _CSV_METRICS])

    # (2) averaged summary: one row per sim (means over seeds).
    with open(os.path.join(out_dir, 'summary.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['sim', 'n_seeds'] + _CSV_METRICS)
        for name, _, _, _ in SIMS:
            d = avgs[name]
            w.writerow([name, n_seeds] +
                       [round(float(d.get(m, 0.0)), 3) for m in _CSV_METRICS])

    # (3) survivor-discovery timeseries (long format): sim, seed, step,
    #     found, found_pct — enough to re-plot discovery curves without a rerun.
    with open(os.path.join(out_dir, 'found_timeseries.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['sim', 'seed', 'step', 'found', 'found_pct'])
        for name, _, _, _ in SIMS:
            for r in sorted(results_by_name[name], key=lambda r: r.seed):
                tot = r.total_survivors or 1
                for step, val in r.found_ts:
                    w.writerow([name, r.seed, step, val,
                                round(val / tot * 100, 2)])

    print(f"  CSVs: runs.csv  summary.csv  found_timeseries.csv")

    # ── Plots ──────────────────────────────────────────────────────────────────
    print("Generating plots...", end=' ', flush=True)
    saved = []

    fig, ax = plt.subplots(figsize=(11, 5))
    _plot_metric(ax, results_by_name,
                 lambda r: [(s, v/r.total_survivors*100) for s, v in r.found_ts],
                 'Survivors found (%)', f'Survivor Discovery Rate — {LARGE_GRID_W}×{LARGE_GRID_H}',
                 steps, ylim=(0, 105))
    saved.append(_save(fig, out_dir, '01_survivors.png'))

    fig, ax = plt.subplots(figsize=(11, 5))
    _plot_metric(ax, results_by_name, lambda r: r.cov_ts,
                 'Coverage (%)', f'Map Coverage Over Time — {LARGE_GRID_W}×{LARGE_GRID_H}', steps)
    saved.append(_save(fig, out_dir, '02_coverage.png'))

    fig, ax = plt.subplots(figsize=(11, 5))
    _plot_metric(ax, results_by_name, lambda r: r.stair_ts,
                 'Stair coverage (%)',
                 f'Building (Stair) Coverage — {LARGE_GRID_W}×{LARGE_GRID_H}',
                 steps, ylim=(0, 105))
    saved.append(_save(fig, out_dir, '03_stair_coverage.png'))

    fig, ax = plt.subplots(figsize=(11, 5))
    _plot_metric(ax, results_by_name,
                 lambda r: [(s, v/max(r.buildings_total, 1)*100) for s, v in r.bldg_ts],
                 'Buildings entered (%)',
                 f'Building Exploration — {LARGE_GRID_W}×{LARGE_GRID_H}',
                 steps, ylim=(0, 105))
    saved.append(_save(fig, out_dir, '04_buildings.png'))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for name, _, col, _ in SIMS:
        comp = [r.completion_step for r in results_by_name[name] if r.completed]
        fail = sum(1 for r in results_by_name[name] if not r.completed)
        lbl  = f"{name} ({len(comp)}/{n_seeds}" + (f", {fail} fail)" if fail else ")")
        if comp: axes[0].hist(comp, bins=max(1, min(8, len(comp))), alpha=0.5,
                               color=col, label=lbl, edgecolor='white')
    axes[0].set_xlabel('Steps to completion'); axes[0].set_ylabel('Count')
    axes[0].set_title('Completion Time Distribution'); axes[0].legend(fontsize=7); axes[0].grid(alpha=0.3)
    x = np.arange(n_seeds); w = 0.15
    for i, (name, _, col, _) in enumerate(SIMS):
        vals = [r.completion_step if r.completed else steps
                for r in sorted(results_by_name[name], key=lambda r: r.seed)]
        axes[1].bar(x + (i - len(SIMS)//2)*w, vals, w, color=col, label=name, alpha=0.8)
    axes[1].axhline(steps, color='grey', lw=1, ls=':', label='timeout')
    axes[1].set_xticks(x); axes[1].set_xticklabels([str(s) for s in seeds])
    axes[1].set_xlabel('Seed'); axes[1].set_ylabel('Steps')
    axes[1].set_title('Completion Time per Seed'); axes[1].legend(fontsize=7); axes[1].grid(alpha=0.3, axis='y')
    fig.suptitle(f'Completion Time — {LARGE_GRID_W}×{LARGE_GRID_H}', fontsize=13)
    saved.append(_save(fig, out_dir, '05_completion_time.png'))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(n_seeds); w = 0.15
    for i, (name, _, col, _) in enumerate(SIMS):
        vals = [r.deaths for r in sorted(results_by_name[name], key=lambda r: r.seed)]
        axes[0].bar(x + (i - len(SIMS)//2)*w, vals, w, color=col, label=name, alpha=0.8)
    axes[0].set_xticks(x); axes[0].set_xticklabels([str(s) for s in seeds])
    axes[0].set_xlabel('Seed'); axes[0].set_ylabel('Deaths')
    axes[0].set_title('Robot Deaths per Seed'); axes[0].legend(fontsize=7); axes[0].grid(alpha=0.3, axis='y')
    for i, (name, _, col, _) in enumerate(SIMS):
        vals = [r.trapped for r in sorted(results_by_name[name], key=lambda r: r.seed)]
        axes[1].bar(x + (i - len(SIMS)//2)*w, vals, w, color=col, label=name, alpha=0.8)
    axes[1].set_xticks(x); axes[1].set_xticklabels([str(s) for s in seeds])
    axes[1].set_xlabel('Seed'); axes[1].set_ylabel('Robot-ticks in uncovered shadow')
    axes[1].set_title('Shadow Entrapment per Seed'); axes[1].legend(fontsize=7); axes[1].grid(alpha=0.3, axis='y')
    fig.suptitle(f'Safety Metrics — {LARGE_GRID_W}×{LARGE_GRID_H}', fontsize=13)
    saved.append(_save(fig, out_dir, '06_safety.png'))

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    _plot_metric(axes[0], results_by_name, lambda r: r.ground_ms_ts,
                 'ms / step', 'Ground-Station Coordination (central)', steps)
    _plot_metric(axes[1], results_by_name, lambda r: r.astar_max_ts,
                 'ms / step', 'Per-Robot Onboard Planning (parallel, max single A*)', steps)
    gtxt = '\n'.join(f"{name}: {avgs[name]['ground_avg']:.0f} ms" for name, _, _, _ in SIMS)
    atxt = '\n'.join(f"{name}: {avgs[name]['astar_max_avg']:.0f} ms" for name, _, _, _ in SIMS)
    axes[0].text(0.98, 0.95, gtxt, transform=axes[0].transAxes, ha='right', va='top',
                 fontsize=8, bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
    axes[1].text(0.98, 0.95, atxt, transform=axes[1].transAxes, ha='right', va='top',
                 fontsize=8, bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
    fig.suptitle(f'Computational Cost — {LARGE_GRID_W}×{LARGE_GRID_H}   '
                 f'(centralized sum is a sim artifact; real distributed ≈ ground + slowest onboard plan)',
                 fontsize=11)
    saved.append(_save(fig, out_dir, '07_compute.png'))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    _plot_metric(axes[0], results_by_name, lambda r: r.zone_redundancy_ts,
                 'Robots in shared zone (%)', 'Zone-Level Redundancy', steps, ylim=(0, None))
    _plot_metric(axes[1], results_by_name, lambda r: r.redundancy_ts,
                 'Robots sharing exact cell (%)', 'Cell-Level Redundancy', steps, ylim=(0, None))
    fig.suptitle(f'Coordination Redundancy — {LARGE_GRID_W}×{LARGE_GRID_H}', fontsize=13)
    saved.append(_save(fig, out_dir, '08_redundancy.png'))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for name, _, col, _ in [s for s in SIMS if 'CARA' in s[0]]:
        for res in results_by_name[name]:
            if res.milp_ms_ts:
                xs = [s for s, _ in res.milp_ms_ts]
                ys = [v for _, v in res.milp_ms_ts]
                axes[0].scatter(xs, ys, color=col, s=20, alpha=0.6, label=f"{name} s={res.seed}")
    axes[0].set_xlabel('Timestep'); axes[0].set_ylabel('MILP solve time (ms)')
    axes[0].set_title('MILP Solve Times per Trigger'); axes[0].legend(fontsize=7); axes[0].grid(alpha=0.3)
    cara_names = [s[0] for s in SIMS if 'CARA' in s[0]]
    cara_cols  = {s[0]: s[2] for s in SIMS if 'CARA' in s[0]}
    x2 = np.arange(n_seeds); w2 = 0.3
    for i, name in enumerate(cara_names):
        vals = [r.hold for r in sorted(results_by_name[name], key=lambda r: r.seed)]
        axes[1].bar(x2 + (i - 0.5)*w2, vals, w2, color=cara_cols[name], label=name, alpha=0.8)
    axes[1].set_xticks(x2); axes[1].set_xticklabels([str(s) for s in seeds])
    axes[1].set_xlabel('Seed'); axes[1].set_ylabel('Robot-ticks at hold gate')
    axes[1].set_title('Hold Gate Latency (CARA-Dynamic)'); axes[1].legend(fontsize=7); axes[1].grid(alpha=0.3, axis='y')
    fig.suptitle('CARA Execution Layer Metrics', fontsize=13)
    saved.append(_save(fig, out_dir, '09_cara_exec_layer.png'))

    fig, ax = plt.subplots(figsize=(11, 5))
    _plot_metric(ax, results_by_name, lambda r: r.flipflop_ts,
                 'Flip-flopping robots (%)', f'Flip-Flop Rate — {LARGE_GRID_W}×{LARGE_GRID_H}',
                 steps, ylim=(0, None))
    saved.append(_save(fig, out_dir, '10_flipflop.png'))

    fig, ax = plt.subplots(figsize=(11, 5))
    _plot_metric(ax, results_by_name, lambda r: r.idle_move_ts,
                 'Moving robots gaining no new cells (%)',
                 f'Idle Movement Rate — {LARGE_GRID_W}×{LARGE_GRID_H}',
                 steps, ylim=(0, None))
    saved.append(_save(fig, out_dir, '11_idle_movement.png'))

    # Summary dashboard
    fig = plt.figure(figsize=(18, 10))
    gs  = gridspec.GridSpec(2, 5, figure=fig, hspace=0.48, wspace=0.38)
    names = [s[0] for s in SIMS]
    cols  = [s[2] for s in SIMS]
    for i, (metric, label) in enumerate([
        ('final_cov',   'Coverage\n(%)'),
        ('final_stair', 'Stair\nCoverage (%)'),
        ('final_found', 'Survivors\nFound'),
        ('deaths',      'Robot\nDeaths'),
        ('trapped',     'Trapped\nTicks'),
    ]):
        ax = fig.add_subplot(gs[0, i])
        vals = [avgs[name][metric] for name in names]
        bars = ax.bar(names, vals, color=cols, alpha=0.85, edgecolor='white')
        ax.set_title(label, fontsize=9); ax.grid(alpha=0.3, axis='y')
        ax.tick_params(axis='x', labelsize=7, rotation=20)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2,
                    bar.get_height()+0.1, f'{val:.1f}',
                    ha='center', va='bottom', fontsize=7)

    ax_ms = fig.add_subplot(gs[1, 0])
    vals  = [avgs[name]['avg_ms'] for name in names]
    bars  = ax_ms.bar(names, vals, color=cols, alpha=0.85, edgecolor='white')
    ax_ms.set_title('Compute\n(ms/step)', fontsize=9); ax_ms.grid(alpha=0.3, axis='y')
    ax_ms.tick_params(axis='x', labelsize=7, rotation=20)
    for bar, val in zip(bars, vals):
        ax_ms.text(bar.get_x()+bar.get_width()/2,
                   bar.get_height()+0.05, f'{val:.1f}',
                   ha='center', va='bottom', fontsize=7)

    ax_surv = fig.add_subplot(gs[1, 1:4])
    _plot_metric(ax_surv, results_by_name,
                 lambda r: [(s, v/r.total_survivors*100) for s, v in r.found_ts],
                 'Survivors found (%)', 'Survivor Discovery Over Time', steps, ylim=(0, 105))

    ax_txt = fig.add_subplot(gs[1, 4]); ax_txt.axis('off')
    lines = [
        f'Large-Scale Summary',
        '─' * 30,
        f"Grid: {LARGE_GRID_W}×{LARGE_GRID_H} @ {LARGE_CELL_SIZE}m",
        f"Seeds: {n_seeds}  |  Steps: {steps}",
        f"Robots: {N_ROBOTS}  |  Survivors: ~{LARGE_N_SURVIVORS}",
        '',
        f"{'Sim':<14} {'Cov%':>5} {'Stair':>6} {'Found':>6} {'ms':>6}",
        '─' * 38,
    ]
    for name in names:
        d = avgs[name]
        lines.append(f"{name:<14} {d['final_cov']:5.1f} {d['final_stair']:6.1f}"
                     f" {d['final_found']:6.1f} {d['avg_ms']:6.1f}")
    if 'CARA-EL-Base' in avgs and 'CARA-EL-Dyn' in avgs:
        lines += ['', 'CARA-EL extras:',
                  f"  Base:    MILP {avgs['CARA-EL-Base']['milp_avg']:.0f}ms avg",
                  f"  Dynamic: {avgs['CARA-EL-Dyn']['hold']:.0f} hold-ticks",
                  f"           {avgs['CARA-EL-Dyn']['ejects']:.0f} eject events"]
    if 'CARA-2022' in avgs:
        lines += ['', 'CARA-2022 (faithful paper mechanism, see Cara_paper.py '
                      'docstring for declared adaptations):',
                  f"  relays placed: {avgs['CARA-2022'].get('relays_placed', 0):.1f} avg/run"]
    if 'ACHORD-insp' in avgs:
        lines += ['', 'ACHORD-inspired (droppable radios, see Achord_sim.py):',
                  f"  radios dropped: {avgs['ACHORD-insp'].get('radios_dropped', 0):.1f} avg/run"]
    ax_txt.text(0.03, 0.97, '\n'.join(lines), transform=ax_txt.transAxes,
                fontsize=8, va='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    fig.suptitle(
        f'Large-Scale Comparison ({LARGE_GRID_W}×{LARGE_GRID_H}, {N_ROBOTS} robots)  —  '
        f'Fleet vs GNF vs ACHORD vs CARA-2022 vs CARA-EL  —  '
        f'{n_seeds} Seed{"s" if n_seeds>1 else ""} × {steps} Steps',
        fontsize=11, fontweight='bold')
    saved.append(_save(fig, out_dir, '12_dashboard.png'))

    # ══════════════════════════════════════════════════════════════════════
    # 13: K compute breakdown by phase (K variants only)
    # ══════════════════════════════════════════════════════════════════════
    k_names_present = [n for n in names if n in K_SIM_NAMES]
    if k_names_present:
        fig, ax = plt.subplots(figsize=(10, 6))
        cats = ['role', 'alloc', 'cover', 'other']
        cat_cols = {'role': '#ff7f0e', 'alloc': '#2ca02c',
                     'cover': '#d62728', 'other': '#7f7f7f'}
        # Aggregate mean ms/step per phase across seeds
        agg = {}
        for kn in k_names_present:
            runs = results_by_name.get(kn, [])
            if not runs: continue
            per_cat = {c: [] for c in cats}
            for r in runs:
                steps_ran = max(1, len(r.step_ms_ts) * RECORD_EVERY)
                for c in cats:
                    per_cat[c].append(r.compute_breakdown.get(c, 0.0) * 1000.0 / steps_ran)
            agg[kn] = {c: float(np.mean(v)) if v else 0.0 for c, v in per_cat.items()}

        x = np.arange(len(k_names_present)); bot = np.zeros(len(k_names_present))
        for c in cats:
            vals = np.array([agg[n][c] for n in k_names_present])
            ax.bar(x, vals, bottom=bot, color=cat_cols[c], label=c, edgecolor='white')
            bot += vals
        ax.set_xticks(x); ax.set_xticklabels(k_names_present)
        ax.set_ylabel('ms per step'); ax.set_title(
            f'Fleet Compute Breakdown by Phase  —  '
            f'{LARGE_GRID_W}×{LARGE_GRID_H}  ({n_seeds} seed{"s" if n_seeds>1 else ""})',
            fontsize=11)
        ax.legend(fontsize=9, loc='upper left'); ax.grid(alpha=0.3, axis='y')
        saved.append(_save(fig, out_dir, '13_k_compute_breakdown.png'))

    # ══════════════════════════════════════════════════════════════════════
    # 14: J-vs-K story — deaths by cause, time-to-milestone, Pareto
    # ══════════════════════════════════════════════════════════════════════
    fig = plt.figure(figsize=(16, 5))
    gs2 = gridspec.GridSpec(1, 3, figure=fig, wspace=0.32)

    # (0) Deaths broken down by cause (stacked)
    ax = fig.add_subplot(gs2[0, 0])
    hz = np.array([avgs[n].get('hazard_deaths', 0) for n in names])
    bt = np.array([avgs[n].get('battery_deaths', 0) for n in names])
    cm = np.array([avgs[n].get('comms_deaths', 0) for n in names])
    xr = np.arange(len(names))
    ax.bar(xr, hz, color='#d62728', label='Hazard', edgecolor='white')
    ax.bar(xr, bt, bottom=hz, color='#ff7f0e', label='Battery', edgecolor='white')
    ax.bar(xr, cm, bottom=hz+bt, color='#8c564b', label='Comms', edgecolor='white')
    ax.set_xticks(xr); ax.set_xticklabels(names, rotation=25, fontsize=8)
    ax.set_ylabel('Deaths (avg)'); ax.set_title('Robot Deaths by Cause', fontsize=10)
    ax.legend(fontsize=8, loc='upper left'); ax.grid(alpha=0.3, axis='y')

    # (1) Time-to-milestone (grouped)
    ax = fig.add_subplot(gs2[0, 1])
    xr = np.arange(len(names)); w = 0.27
    first = [avgs[n].get('t_first_found', 0) for n in names]
    half  = [avgs[n].get('t_half_found', 0)  for n in names]
    all_  = [avgs[n].get('t_all_found', 0)   for n in names]
    ax.bar(xr - w, first, w, label='1st', color='#4dac26', alpha=0.85)
    ax.bar(xr,      half,  w, label='½',   color='#fdae61', alpha=0.85)
    ax.bar(xr + w,  all_,  w, label='All', color='#d73027', alpha=0.85)
    ax.set_xticks(xr); ax.set_xticklabels(names, rotation=25, fontsize=8)
    ax.set_ylabel('Tick'); ax.set_title(
        'Time-to-Milestone  (0 = not reached in run)', fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis='y')

    # (2) Compute-vs-Coverage Pareto
    ax = fig.add_subplot(gs2[0, 2])
    for name, _, colour, _ in SIMS:
        if name not in avgs: continue
        x = avgs[name].get('avg_ms', 0.0)
        y = avgs[name].get('final_cov', 0.0)
        ax.scatter([x], [y], s=200, color=colour, edgecolor='black',
                     linewidth=1, zorder=3)
        ax.annotate(name, (x, y), xytext=(5, 5),
                     textcoords='offset points', fontsize=8)
    ax.set_xlabel('Compute (ms/step)'); ax.set_ylabel('Coverage (%)')
    ax.set_title('Compute vs. Effectiveness  (up-and-left is better)',
                  fontsize=10)
    ax.grid(alpha=0.3)

    fig.suptitle(
        f'Mission Outcomes and Efficiency  ({n_seeds} seed{"s" if n_seeds>1 else ""})',
        fontsize=12, fontweight='bold')
    saved.append(_save(fig, out_dir, '14_jvsk_summary.png'))

    print("done")
    print(f"\nTotal bench time: {_fmt_hms(time.time() - _bench_t0)}"
          f"  ({len(seeds)} seed{'s' if len(seeds)>1 else ''} x {len(SIMS)} sims)")
    print(f"\nPlots saved to: {out_dir}/")
    for f in saved:
        print(f"  {f}")
    print(f"  summary.json")


if __name__ == '__main__':
    main()