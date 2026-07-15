#!/usr/bin/env python3
"""
AblationBench.py — component ablation study of Framework M (K).

Framework M stacks two coordination subsystems:
  HRBA  (Hierarchical Region Bundle Allocation) — WHICH ZONE each robot works
  PG    (heterogeneous potential game)          — WHICH ROLE each robot plays

This bench builds ablated VARIANTS of the framework at runtime — the source
file 2DFleetFrameworkM.py is never modified — runs every variant on the same
seeds, and exports graphs + tidy CSVs for later statistical analysis.

Variant construction is done two ways, both auditable:
  * SOURCE PATCHES  — tiny, exact string replacements applied to a fresh copy
    of the framework source before import.  Every anchor is verified to occur
    EXACTLY the expected number of times; if the framework changes and an
    anchor breaks, the bench refuses to run that variant rather than silently
    running the wrong experiment.
  * METHOD PATCHES  — whole-method monkeypatches applied after import (used
    where the ablation replaces an entire mechanism, e.g. greedy allocation
    in place of the HRBA auction).

CELLS (leave-one-out within each subsystem + interaction 2x2):

  A-axis — HRBA / CBBA layer
    A0-FULL   control (also serves as B0)
    A1-NOCLUS Layer 0 spatial clustering off (single global auction)
    A2-GREEDY whole auction replaced by nearest-feasible-zone greedy claim
    A3-NODIV  capability-diversity enforcement off (both consensus layers)
    A4-NOL2   inter-cluster conflict resolution (Layer 2) off
    A5-NORESC fallback + stair rescue + open-terrain rescue off

  B-axis — potential-game role selection
    B1-NOCONG congestion pricing off (gamma = 0 for all roles)
    B2-NOCAP  capability-yield term fixed at 1.0 (types interchangeable)
    B3-NOMF   mean-field signal off  (global coupling removed — the game
              decomposes into independent local games; tests whether the
              coupling that costs the exact-potential guarantee — leaving an
              ordinal potential game — buys anything in practice)
    B4-NAIVRV relay value = raw unknown-fraction (pre-mechanism-design form;
              life-safety override retained so safety is not confounded)
    B5-NOTRAV relay travel-cost weighting off (RELAY_TRAVEL_W = 0)
    A6-NOSWEEP residual-unknown sweep disabled (SWEEP_ENABLED = False)
    B6-NOSUB  radius-bounded election sub-clustering off (one election
              region per whole physical shadow cluster)
    B7-NOGAME potential game replaced by fixed heuristic (nearest eligible
              robot per uncovered cluster -> RELAY, everyone else -> SCAN)

  Interaction 2x2 (do the two subsystems need each other?)
    FULL       = A0        (full HRBA x full PG)
    PG-ONLY    = A2-GREEDY (greedy zones  x full PG)
    HRBA-ONLY  = B7-NOGAME (full HRBA    x fixed roles)
    NEITHER    = X-NEITHER (greedy zones x fixed roles)

Usage:
    python3 AblationBench.py --seeds 3 --steps 800
    python3 AblationBench.py --quick                  # control + 3 cells, tiny
    python3 AblationBench.py --cells A0-FULL,B3-NOMF,X-NEITHER
Outputs (to --out):
    ablation_runs.csv        one row per (variant, seed)  <- unit for stats
    ablation_timeseries.csv  long format (variant, seed, step, metric, value)
    ablation_summary.csv     means across seeds
    ablation_design.csv      what each cell removes / isolates
    ablation_deltas.png      outcome deltas vs control
    ablation_curves.png      coverage / found trajectories
    ablation_interaction.png the 2x2 HRBA x PG interaction
"""
import argparse, csv, importlib.util, os, random, sys, tempfile, time
from collections import defaultdict

import numpy as np

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

_HERE = os.path.dirname(os.path.abspath(__file__))
_SEARCH_DIRS = [_HERE, os.path.join(_HERE, '..'), '/mnt/user-data/outputs', '/home/claude/work']


def _find(*names):
    for name in names:
        for folder in _SEARCH_DIRS:
            p = os.path.join(folder, name)
            if os.path.exists(p):
                return os.path.abspath(p)
    print("ERROR: cannot find any of", names, "in", _SEARCH_DIRS)
    sys.exit(1)


FRAMEWORK_PATH = _find('2DFleetFrameworkM.py', 'FleetFrameworkM.py')

# ══════════════════════════════════════════════════════════════════════════════
# Large-scale environment (matches ComparisonBenchLarge.py exactly, so ablation
# results are directly comparable to the five-way benchmark's numbers).
#
# The small default environment (128x128, ~10 robots, 18 survivors) turned out
# too forgiving: a lucky exploration order alone got most ablated variants to
# 90%+ found, which barely differentiated the components under test. At 50
# robots / 45 survivors / 200x200 there is real congestion, real travel time
# to relay borders, and real zone contention — the conditions the HRBA auction
# and the potential game actually exist to handle — so ablating them should
# show up much more clearly than it did in the small environment.
# ══════════════════════════════════════════════════════════════════════════════
LARGE_ENV = True   # --small flips this off for quick iteration
HAZARD_ENV = True  # --plain-env flips this off (see apply_hazard_env below)

LARGE_GRID_W      = 200
LARGE_GRID_H      = 200
LARGE_CELL_SIZE   = 5
LARGE_ZONE_CHUNKS = 10
LARGE_MAX_BATTERY = 2000
LARGE_MAX_BUNDLE  = 8
LARGE_N_SURVIVORS = 45
LARGE_ROBOTS = {"Legged": 13, "Drone": 17, "Boat": 8, "Rover": 12}


def _apply_large_scale(M):
    """Everything ComparisonBenchLarge.py applies before instantiating FleetSim,
    minus the parts specific to comparing against other frameworks (J/GNF/CARA
    patching, baseline aliasing). Applied fresh to each variant's own module
    object, same as the ablation source/method patches."""
    M.GRID_W      = LARGE_GRID_W
    M.GRID_H      = LARGE_GRID_H
    M.CELL_SIZE   = LARGE_CELL_SIZE
    M.ZONE_CHUNKS = LARGE_ZONE_CHUNKS
    M.MAX_BATTERY = LARGE_MAX_BATTERY
    M.MAX_BUNDLE  = LARGE_MAX_BUNDLE
    M.RELAY_MIN_HOLD = 300
    if hasattr(M, 'ALLOCATION_CADENCE'):
        M.ALLOCATION_CADENCE = 100
    M.COOLDOWN_T = 120
    M.CBBA_ITERS = 1

    # goal_commit x3 — cross-map paths are 100+ steps at this scale
    _orig_robot_init = M.Robot.__init__
    def _patched_robot_init(self, *a, **k):
        _orig_robot_init(self, *a, **k)
        self.goal_commit = 60
    M.Robot.__init__ = _patched_robot_init

    # Share reachable arrays across same-capability robots (4 distinct masks,
    # not 50 separate scipy.ndimage.label calls per tick).
    _orig_reachable = M.Robot.reachable
    def _patched_reachable(self):
        t = self.sim.timestep
        if self._reachable_tick == t and self._reachable_arr is not None:
            return self._reachable_arr
        mask = self.caps_mask
        if not hasattr(self.sim, '_reachable_mask_tick'):
            self.sim._reachable_mask_tick = {}
        tick_cache = self.sim._reachable_mask_tick
        if tick_cache.get(mask) == t and mask in getattr(self.sim, '_reachable_by_mask', {}):
            self._reachable_arr   = self.sim._reachable_by_mask[mask]
            self._reachable_cache = None
            self._reachable_tick  = t
            return self._reachable_arr
        result = _orig_reachable(self)
        if not hasattr(self.sim, '_reachable_by_mask'):
            self.sim._reachable_by_mask = {}
        self.sim._reachable_by_mask[mask] = result
        tick_cache[mask] = t
        return result
    M.Robot.reachable = _patched_reachable

    # Lock CBBA cadence at 50 ticks minimum — the late-game 20-tick cadence
    # doubles allocation cost at 50 robots for no measurable outcome benefit.
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

    # Gate _decide_roles on a change signal — otherwise the full best-response
    # loop (O(R^2 x roles)) runs every tick regardless of whether anything
    # changed. Safety net: always re-run at least every 5 ticks.
    _orig_decide_roles = M.FleetSim._decide_roles
    def _patched_decide_roles(self):
        t = self.timestep
        relay_flood = getattr(self, '_relay_ok_flood', {})
        shadow_type = getattr(self, '_shadow_zone_type', {})
        relay_sig = frozenset(z for z, ok in relay_flood.items() if ok)
        n_active  = sum(1 for r in self.robots if r.active)
        cbba_tick = getattr(self, '_cbba_last_tick', -999)
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

    # 50-robot roster in 9 spawn clusters, boats seeded on non-shadow water.
    def _patched_build_robots(self):
        # Graded hazard durability (Drone fragile / Legged moderate / Rover
        # hardened / Boat immune). Falls back to the legacy shared limits if
        # the loaded framework predates ROBOT_HAZARD_PROFILE.
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

    # 45 survivors: 40% in buildings (stairs), 60% in the open, grid-spread so
    # they aren't clustered by chance — same placement ComparisonBenchLarge uses.
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
        if len(survivors) < n_total:
            _used = set(survivors)
            spare = [c for c in free_open if c not in _used]
            random.shuffle(spare)
            survivors += spare[:n_total - len(survivors)]
        self.survivors = survivors[:n_total]
    M.FleetSim._build_survivors = _patched_build_survivors


# ══════════════════════════════════════════════════════════════════════════════
# Source patches — (anchor, replacement, expected_count)
# ══════════════════════════════════════════════════════════════════════════════
SRC_PATCHES = {
    'A1-NOCLUS': [
        ("k = max(1, min(8, round(R ** 0.5)))",
         "k = 1  # [ABLATION A1] single global cluster — Layer 0 off", 1),
    ],
    'A3-NODIV': [
        ("if rr.caps_mask in used_caps: continue",
         "if False: continue  # [ABLATION A3] capability diversity off", 2),
    ],
    'A4-NOL2': [
        ("zone_claims_global.setdefault(z, []).append((r.name, u))",
         "pass  # [ABLATION A4] Layer-2 inter-cluster resolution off", 1),
    ],
    'A5-NORESC': [
        ("z = self._fallback_zone(r)",
         "z = None  # [ABLATION A5] fallback zone off", 1),
        ("if zt != 'stair': continue",
         "continue  # [ABLATION A5] stair-zone rescue off", 1),
        ("if self.zone_stats(z)['unknown_frac'] < 0.40: continue",
         "continue  # [ABLATION A5] open-terrain rescue off", 1),
    ],
    'B1-NOCONG': [
        ("_GAMMA = {Role.SCOUT: 1.5, Role.SCAN: 0.5, Role.LOITER: 0.4, Role.RELAY: 0.8}",
         "_GAMMA = {Role.SCOUT: 0.0, Role.SCAN: 0.0, Role.LOITER: 0.0, Role.RELAY: 0.0}"
         "  # [ABLATION B1]", 1),
    ],
    'A6-NOSWEEP': [
        # Endgame residual-unknown sweep off: robots whose hierarchy gives
        # them no goal stand down (pre-sweep behavior). Measures the sweep's
        # contribution to t_all / found@deadline as its own component.
        ("SWEEP_ENABLED = True",
         "SWEEP_ENABLED = False  # [ABLATION A6] endgame sweep disabled", 1),
    ],
    'B5-NOTRAV': [
        # NOTE: the pre-escalation n=16 dataset showed removal helping ~13%%;
        # a post-escalation spot-check (R=3+dose+water gate) showed the
        # opposite (40->35 found @280 on hazard seed 10). The direction is an
        # OPEN QUESTION this cell re-asks on the locked framework.
        ("_RELAY_TRAVEL_W = 3.0",
         "_RELAY_TRAVEL_W = 0.0  # [ABLATION B5] relay travel weighting off", 1),
    ],
    'B6-NOSUB': [
        ("subcl_map = getattr(self, '_shadow_subcluster_id', None) or {}",
         "subcl_map = getattr(self, '_shadow_cluster_id', None) or {}"
         "  # [ABLATION B6] whole-cluster election regions", 1),
    ],
}

# ══════════════════════════════════════════════════════════════════════════════
# Method patches — applied after import
# ══════════════════════════════════════════════════════════════════════════════

def _patch_no_capyield(M):
    """B2: all robot types treated as interchangeable in the role game."""
    M.FleetSim._capability_yield = lambda self, robot, zone: 1.0


def _patch_no_meanfield(M):
    """B3: global mean-field coupling removed — independent local games."""
    M.FleetSim._mean_field_signal = lambda self, robot, zone: 0.0


def _patch_naive_relay_val(M):
    """B4: relay value = raw unknown-fraction (pre-mechanism-design form).
    The life-safety override (explorers stranded inside) is RETAINED so the
    ablation measures the valuation refinement, not the safety property."""
    def _naive(self, cluster):
        cluster_zones = set(cluster)
        x0_min = self.world.w; x1_max = 0
        y0_min = self.world.h; y1_max = 0
        for z in cluster:
            zx, zy = z
            x0 = zx * self.zone_w_cells; x1 = min(x0 + self.zone_w_cells, self.world.w)
            y0 = zy * self.zone_h_cells; y1 = min(y0 + self.zone_h_cells, self.world.h)
            x0_min = min(x0_min, x0); x1_max = max(x1_max, x1)
            y0_min = min(y0_min, y0); y1_max = max(y1_max, y1)
        explorers_inside = sum(
            1 for r in self.robots
            if r.active and r.role != M.Role.RELAY
            and x0_min <= r.pos[0] < x1_max and y0_min <= r.pos[1] < y1_max
            and self.radio_shadow[r.pos[0], r.pos[1]])
        if explorers_inside:
            return 50.0 * explorers_inside
        zuf = self._zone_uf_cache
        val = sum(zuf.get(z, 0.0) * self._zone_shadow_frac.get(z, 0.0)
                  for z in cluster) * 1.5
        val = min(val, 4.0)
        dead_inside = sum(
            1 for r in self.robots
            if not r.active
            and self.cell_to_zone(r.pos[0], r.pos[1]) in cluster_zones
            and self.radio_shadow[r.pos[0], r.pos[1]])
        return max(0.2, val + dead_inside * 1.5)
    M.FleetSim._relay_val = _naive


def _patch_greedy_alloc(M):
    """A2: the whole HRBA auction replaced by nearest-feasible-zone greedy.
    No clustering, no bidding, no consensus, no diversity, bundle size 1.
    Keeps the once-per-tick guard and the zone-task release bookkeeping so
    the rest of the framework sees a well-formed allocation."""
    def _greedy(self):
        if (self.timestep > 0
                and getattr(self, '_cbba_last_tick', -1) == self.timestep):
            return
        self._cbba_last_tick = self.timestep
        for z, task in self.zone_tasks.items():
            task.progress = self.zone_coverage(self.union_belief, z)
            is_shz = self._zone_shadow_frac.get(z, 0.0) > 0.2
            done_thr = M.SHADOW_ZONE_DONE if is_shz else M.ZONE_DONE
            if task.progress >= done_thr and task.status != "blacklisted":
                task.owners = []; task.status = "released"; task.expires_at = 0
        explorers = []
        for r in self.robots:
            if not r.active or r.battery <= 0:
                continue
            if r.role == M.Role.RELAY:
                for z in r.bundle:
                    task = self.zone_tasks.get(z)
                    if task and r.name in task.owners:
                        task.owners.remove(r.name)
                        if not task.owners:
                            task.status = 'free'; task.expires_at = 0
                r.bundle = []; r.assigned_zones = []
                continue
            zone_done = (r.task_zone is None or
                         self.zone_coverage(self.union_belief, r.task_zone) >= M.ZONE_DONE)
            if zone_done:
                r.bundle = []; r.assigned_zones = []
            explorers.append(r)
        for z, t in self.zone_tasks.items():
            if t.status != "blacklisted":
                t.owners = [nm for nm in t.owners
                            if any(rr.name == nm and z in rr.bundle for rr in explorers)]
        zones = [(zx, zy) for zx in range(self.zone_nx) for zy in range(self.zone_ny)]
        random.shuffle(explorers)
        for r in explorers:
            if r.bundle:
                continue
            best = None; best_d = 1e18
            for z in zones:
                t = self.zone_tasks[z]
                if t.status == "blacklisted":
                    continue
                if len(t.owners) >= self._zone_capacity(z):
                    continue
                st = self.zone_stats(z)
                if st['unknown_frac'] < 0.05:
                    continue
                if not self.zone_feasible(r, st, zone=z):
                    continue
                zcx = z[0] * self.zone_w_cells + self.zone_w_cells // 2
                zcy = z[1] * self.zone_h_cells + self.zone_h_cells // 2
                d = abs(r.pos[0] - zcx) + abs(r.pos[1] - zcy)
                if d < best_d:
                    best_d = d; best = z
            if best is not None:
                r.bundle = [best]; r.assigned_zones = [best]
                t = self.zone_tasks[best]
                if r.name not in t.owners:
                    t.owners.append(r.name)
                t.status = "held"; t.expires_at = self.timestep + M.LEASE_T
    M.FleetSim._assign_zones_cbba = _greedy


def _patch_fixed_roles(M):
    """B7: potential game replaced by a fixed heuristic — for every uncovered
    shadow cluster with no assigned relay, the nearest eligible robot becomes
    RELAY; every other non-relay robot is SCAN. No SCOUT, no LOITER, no
    utilities, no best response."""
    def _fixed(self, active, clusters):
        t = self.timestep
        for r in active:
            if r.role != M.Role.RELAY and t >= r.role_locked_until:
                r.role = M.Role.SCAN
        for cl in clusters:
            if not cl:
                continue
            covered = any(self._relay_ok_flood.get(z, False) for z in cl)
            if covered:
                continue
            if any(rr.role == M.Role.RELAY and rr.task_zone in cl for rr in active):
                continue
            ctype = self._shadow_zone_type.get(cl[0], 'none')
            centres = [(z[0] * self.zone_w_cells + self.zone_w_cells // 2,
                        z[1] * self.zone_h_cells + self.zone_h_cells // 2) for z in cl]
            best = None; best_d = 1e18
            for rr in active:
                if rr.role == M.Role.RELAY or t < rr.role_locked_until:
                    continue
                if self.radio_shadow[rr.pos[0], rr.pos[1]]:
                    continue
                is_boat = bool(rr.caps_mask & M.CAP_WATER) and not bool(rr.caps_mask & M.CAP_AIR)
                is_land_only = (bool(rr.caps_mask & M.CAP_LAND)
                                and not bool(rr.caps_mask & (M.CAP_STAIRS | M.CAP_AIR | M.CAP_WATER)))
                if ctype == 'stair' and (is_boat or is_land_only):
                    continue
                if is_boat and not self._cluster_border_has_water.get(
                        self._shadow_cluster_id.get(cl[0]), False):
                    continue
                d = min(abs(rr.pos[0] - cx) + abs(rr.pos[1] - cy) for cx, cy in centres)
                if d < best_d:
                    best_d = d; best = rr
            if best is not None:
                for z in best.bundle:
                    task = self.zone_tasks.get(z)
                    if task and best.name in task.owners:
                        task.owners.remove(best.name)
                        if not task.owners:
                            task.status = 'free'; task.expires_at = 0
                best.bundle = []; best.assigned_zones = []
                best.role = M.Role.RELAY
                best.task_zone = max(cl, key=lambda z: self._zone_shadow_frac.get(z, 0.0))
                best.relay_hold_until = t + M.RELAY_MIN_HOLD
                best.role_locked_until = t + M.RELAY_MIN_HOLD
                best.relay_last_occupied = t
    M.FleetSim._pg_best_response_roles = _fixed


METHOD_PATCHES = {
    'A2-GREEDY': [_patch_greedy_alloc],
    'B2-NOCAP':  [_patch_no_capyield],
    'B3-NOMF':   [_patch_no_meanfield],
    'B4-NAIVRV': [_patch_naive_relay_val],
    'B7-NOGAME': [_patch_fixed_roles],
    'X-NEITHER': [_patch_greedy_alloc, _patch_fixed_roles],
}



# ══════════════════════════════════════════════════════════════════════════════
# Hazard-dense randomised environment (4-8 buildings; large lethal zones +
# many small hotspots for temp AND radiation). Solvability guardrails: no
# hotspot near spawn clusters; lethal cores kept off buildings so stair-
# capable robots are never sealed out. Randomised per run via the seeded
# global `random` module, so runs stay reproducible.
# ══════════════════════════════════════════════════════════════════════════════
import math
import random

import numpy as np


def apply_hazard_env(M,
                     extra_buildings=(0, 4),
                     temp_large=(3, 4), temp_large_sigma=(12.0, 18.0), temp_large_amp=(150.0, 260.0),
                     temp_small=(12, 20), temp_small_sigma=(2.5, 5.5), temp_small_amp=(90.0, 220.0),
                     rad_large=(2, 3),  rad_large_sigma=(12.0, 18.0),  rad_large_amp=(140.0, 240.0),
                     rad_small=(10, 16), rad_small_sigma=(2.5, 5.0),   rad_small_amp=(80.0, 200.0),
                     spawn_margin=14, bldg_margin_small=6):
    """Patch M.GridWorld generation in place. Call BEFORE FleetSim()."""

    # ── extra buildings on top of the original quadrant pass ────────────────
    _orig_generate = M.GridWorld._generate

    def _generate_hazard(self):
        _orig_generate(self)                      # river, bridges, 4 buildings
        n_extra = random.randint(*extra_buildings)
        for _ in range(n_extra):
            hw = random.randint(10, 18); hh = random.randint(10, 18)
            for _try in range(150):
                hx = random.randint(4, self.w - hw - 4)
                hy = random.randint(4, self.h - hh - 4)
                if self._rect_clear(hx, hy, hw, hh, pad=4):
                    self._stamp_house(hx, hy, hw, hh)
                    break
        self._fix_land_pinches(min_corridor=4)
    M.GridWorld._generate = _generate_hazard

    # ── shared hotspot machinery ─────────────────────────────────────────────
    def _building_rects(self):
        """Bounding boxes of stair blobs (post-generation, so includes extras)."""
        from scipy import ndimage as _ndi
        sm = np.zeros((self.w, self.h), dtype=bool)
        for x in range(self.w):
            for y in range(self.h):
                if self.grid[x][y]["t"] == M.T_STAIRS:
                    sm[x, y] = True
        lab, n = _ndi.label(sm)
        rects = []
        for sl in _ndi.find_objects(lab):
            if sl is not None:
                rects.append((sl[0].start, sl[1].start, sl[0].stop, sl[1].stop))
        return rects

    def _spawn_centres(self):
        W, H = self.w, self.h
        return [(W//6, H//6), (W//2, H//6), (5*W//6, H//6),
                (W//6, H//2), (W//2, H//2), (5*W//6, H//2),
                (W//6, 5*H//6), (W//2, 5*H//6), (5*W//6, 5*H//6)]

    def _sample_structured(self, n_rng, sigma_rng, amp_rng, bldg_margin):
        """Sample hotspots respecting spawn + building margins."""
        rects = getattr(self, '_hz_bldg_rects', None)
        if rects is None:
            rects = self._hz_bldg_rects = _building_rects(self)
        spawns = _spawn_centres(self)
        out = []
        n = random.randint(*n_rng)
        for _ in range(n):
            for _try in range(200):
                mx = random.randint(0, self.w - 1)
                my = random.randint(0, self.h - 1)
                sigma = random.uniform(*sigma_rng)
                if any((mx-sx)**2 + (my-sy)**2 < spawn_margin**2 for sx, sy in spawns):
                    continue
                margin = max(bldg_margin, sigma + 4) if sigma > 8 else bldg_margin
                if any(x0 - margin <= mx < x1 + margin and
                       y0 - margin <= my < y1 + margin
                       for x0, y0, x1, y1 in rects):
                    continue
                out.append(((mx, my), sigma, random.uniform(*amp_rng)))
                break
        return out

    def _field_from(self, hotspots, water_value):
        xs = np.arange(self.w, dtype=np.float64)[:, None]
        ys = np.arange(self.h, dtype=np.float64)[None, :]
        field = np.zeros((self.w, self.h), dtype=np.float64)
        for (mx, my), s, amp in hotspots:
            field += amp * np.exp(-((xs - mx)**2 + (ys - my)**2) / (2.0 * s * s))
        return field

    def _init_temperature_hazard(self):
        hs = (_sample_structured(self, temp_large, temp_large_sigma, temp_large_amp, bldg_margin_small)
              + _sample_structured(self, temp_small, temp_small_sigma, temp_small_amp, bldg_margin_small))
        self._hz_temp_hotspots = hs           # kept for visualisation/analysis
        field = _field_from(self, hs, 5.0)
        for x in range(self.w):
            row = field[x]
            for y in range(self.h):
                c = self.grid[x][y]
                c["temp"] = 5.0 if c["t"] in (M.T_WATER, M.T_BRIDGE) else float(row[y])
    M.GridWorld._init_temperature = _init_temperature_hazard

    def _init_radiation_hazard(self):
        hs = (_sample_structured(self, rad_large, rad_large_sigma, rad_large_amp, bldg_margin_small)
              + _sample_structured(self, rad_small, rad_small_sigma, rad_small_amp, bldg_margin_small))
        self._hz_rad_hotspots = hs
        field = _field_from(self, hs, 0.0)
        for x in range(self.w):
            row = field[x]
            for y in range(self.h):
                c = self.grid[x][y]
                c["rad"] = 0.0 if c["t"] in (M.T_WATER, M.T_BRIDGE) else float(row[y])
    M.GridWorld._init_radiation = _init_radiation_hazard

# ══════════════════════════════════════════════════════════════════════════════
# Design table — the "comparison document"
# ══════════════════════════════════════════════════════════════════════════════
DESIGN = [
    ('A0-FULL',   'control', 'nothing',
     'Full Framework M: HRBA (Layers 0-2 + rescues) x heterogeneous potential game.'),
    ('A1-NOCLUS', 'HRBA', 'Layer 0 spatial clustering (single global auction)',
     'Is sqrt(R)-clustering a scaling device only, or does local competition give better assignments?'),
    ('A2-GREEDY', 'HRBA', 'entire auction (bid/consensus) -> nearest-feasible-zone greedy',
     'Value of the CBBA bidding/consensus mechanism vs a cheap greedy heuristic. Also the PG-ONLY interaction cell.'),
    ('A3-NODIV',  'HRBA', 'capability-diversity enforcement in both consensus layers',
     'Does forcing heterogeneous types per zone improve coverage, or is it overhead?'),
    ('A4-NOL2',   'HRBA', 'Layer 2 inter-cluster conflict resolution',
     'Do boundary-zone double-claims meaningfully hurt outcomes?'),
    ('A5-NORESC', 'HRBA', 'fallback zone + stair rescue + open-terrain rescue',
     'How much of final coverage/found is delivered by the safety-net passes vs the primary auction?'),
    ('B1-NOCONG', 'PG', 'congestion pricing (gamma=0 all roles)',
     'Is congestion pricing what balances the role distribution?'),
    ('B2-NOCAP',  'PG', 'capability-yield term (fixed 1.0)',
     'Value of heterogeneity-aware self-selection vs treating all types as interchangeable.'),
    ('B3-NOMF',   'PG', 'mean-field signal (global coupling)',
     'Does the coupling that reduces the exact-potential guarantee to an ORDINAL one buy anything in practice?'),
    ('B4-NAIVRV', 'PG', 'mechanism-design relay valuation -> raw unknown-fraction',
     'Does expected-survivor relay pricing produce earlier/better relay placement? (life-safety override retained)'),
    ('A6-NOSWEEP', 'HRBA', 'residual-unknown endgame sweep (off)',
     'sweep closes the residual-unknown tail the zone auction quantizes away'),
    ('B5-NOTRAV', 'PG', 'relay travel-cost weighting (RELAY_TRAVEL_W=0)',
     'Does penalising distance-to-anchor in relay election reduce time-to-coverage?'),
    ('B6-NOSUB',  'PG', 'radius-bounded election sub-clustering (whole physical clusters)',
     'Is the radius bound what lets large buildings get multiple relays?'),
    ('B7-NOGAME', 'PG', 'entire potential game -> fixed nearest-to-cluster heuristic',
     'Ceiling case: what does ANY game-theoretic role selection buy? Also the HRBA-ONLY interaction cell.'),
    ('X-NEITHER', 'interaction', 'HRBA auction AND potential game (greedy zones + fixed roles)',
     'Floor case of the 2x2: are HRBA and PG additive, or only useful together?'),
]
ALL_CELLS = [d[0] for d in DESIGN]
INTERACTION = {'FULL': 'A0-FULL', 'PG-ONLY': 'A2-GREEDY',
               'HRBA-ONLY': 'B7-NOGAME', 'NEITHER': 'X-NEITHER'}

# ══════════════════════════════════════════════════════════════════════════════
# Variant loader
# ══════════════════════════════════════════════════════════════════════════════
_SRC = open(FRAMEWORK_PATH, 'r', encoding='utf-8', errors='replace').read()
_LOAD_N = 0


def load_variant(cell):
    """Fresh framework module with this cell's patches applied. Anchors are
    verified against exact expected counts — a broken anchor aborts loudly."""
    global _LOAD_N
    src = _SRC
    for anchor, repl, expect in SRC_PATCHES.get(cell, []):
        n = src.count(anchor)
        if n != expect:
            raise RuntimeError(
                f"[{cell}] anchor check failed: expected {expect} occurrence(s) of "
                f"{anchor!r}, found {n}. Framework source changed — update the patch.")
        src = src.replace(anchor, repl)
    _LOAD_N += 1
    tmp = os.path.join(tempfile.gettempdir(), f'_ablat_{cell.replace("-","_")}_{_LOAD_N}.py')
    with open(tmp, 'w', encoding='utf-8') as f:
        f.write(src)
    name = f'FrameworkM_{cell.replace("-","_")}_{_LOAD_N}'
    spec = importlib.util.spec_from_file_location(name, tmp)
    M = importlib.util.module_from_spec(spec)
    sys.modules[name] = M
    spec.loader.exec_module(M)
    for fn in METHOD_PATCHES.get(cell, []):
        fn(M)
    if LARGE_ENV:
        _apply_large_scale(M)
    if HAZARD_ENV:
        apply_hazard_env(M)
    return M

# ══════════════════════════════════════════════════════════════════════════════
# Progress bar (same style as ComparisonBenchLarge.py, so output looks familiar)
# ══════════════════════════════════════════════════════════════════════════════
BAR_WIDTH = 32


def _progress(done, total, prefix='', suffix=''):
    filled = int(BAR_WIDTH * done / max(total, 1))
    bar = '#' * filled + '-' * (BAR_WIDTH - filled)
    pct = 100.0 * done / max(total, 1)
    sys.stdout.write(f'\r  {prefix}[{bar}] {pct:5.1f}%  {suffix:<40s}')
    sys.stdout.flush()


def _clear():
    sys.stdout.write('\r' + ' ' * 100 + '\r')
    sys.stdout.flush()


# ══════════════════════════════════════════════════════════════════════════════
# One run
# ══════════════════════════════════════════════════════════════════════════════
RECORD_EVERY = 25


def run_one(cell, seed, steps, deadline, overall_prefix=''):
    M = load_variant(cell)
    random.seed(seed); np.random.seed(seed)
    sim = M.FleetSim()
    n_surv = len(sim.survivors)

    stair_mask = np.array(
        [[sim.world.grid[x][y]["t"] == M.T_STAIRS for y in range(sim.world.h)]
         for x in range(sim.world.w)], dtype=bool)
    n_stair = max(1, int(stair_mask.sum()))

    cov_ts, found_ts, stair_ts, relay_ts = [], [], [], []
    step_ms = []
    relay_duty = 0
    t_first = t_half = t_all = -1
    t_cov90 = t_stair50 = -1
    found_deadline = None; cov_deadline = None   # captured at the deadline tick
    wall0 = time.time()
    label = f'{cell}/seed={seed}'

    for step in range(1, steps + 1):
        t0 = time.perf_counter()
        alive = sim.step()
        step_ms.append((time.perf_counter() - t0) * 1000.0)

        nf = len(sim.found)
        if nf >= 1 and t_first < 0: t_first = step
        if nf >= (n_surv + 1) // 2 and t_half < 0: t_half = step
        if nf >= n_surv and t_all < 0: t_all = step
        nr = sum(1 for r in sim.robots if r.active and r.role == M.Role.RELAY)
        relay_duty += nr

        # Deadline snapshot: mission state at the fixed rescue deadline. This
        # is the headline SUCCESS metric — with a generous fleet, every policy
        # eventually finds everyone, so final found%% saturates at 100%% and
        # differentiates nothing; found%% AT A DEADLINE converts the
        # time-to-rescue spread into a success spread, which is what matters
        # in SAR (survivors do not wait).
        if step == deadline:
            found_deadline = nf
            cov_deadline = float(np.mean(sim.union_belief != M.T_UNKNOWN)) * 100.0

        if step % RECORD_EVERY == 0 or step == 1:
            cov = float(np.mean(sim.union_belief != M.T_UNKNOWN)) * 100.0
            scv = float(np.mean(sim.union_belief[stair_mask] != M.T_UNKNOWN)) * 100.0 \
                if n_stair else 0.0
            if cov >= 90.0 and t_cov90 < 0: t_cov90 = step
            if scv >= 50.0 and t_stair50 < 0: t_stair50 = step
            cov_ts.append((step, cov)); found_ts.append((step, nf))
            stair_ts.append((step, scv)); relay_ts.append((step, nr))
        if not alive:
            break

        if step % 5 == 0 or step == steps:
            elapsed = time.time() - wall0
            eta = (steps - step) / max(step, 1) * elapsed
            _progress(step, steps, prefix=overall_prefix,
                      suffix=f'{label}  step {step}/{steps}  '
                             f'cov={float(np.mean(sim.union_belief != M.T_UNKNOWN))*100:4.1f}%  '
                             f'found={nf}/{n_surv}  ETA {int(eta//60):02d}:{int(eta%60):02d}')

    _clear()
    cov = float(np.mean(sim.union_belief != M.T_UNKNOWN)) * 100.0
    if found_deadline is None:
        # Run ended before the deadline tick — either mission complete (all
        # found: deadline value is the full count) or fleet dead (deadline
        # value is whatever was found by then). Both equal the final state.
        found_deadline = len(sim.found)
        cov_deadline = cov
    scv = float(np.mean(sim.union_belief[stair_mask] != M.T_UNKNOWN)) * 100.0
    rep = sim.comms_loss_report() if hasattr(sim, 'comms_loss_report') else \
        {'stranded_at_end': [], 'blackout_ticks_total': 0, 'blackout_ticks_max': 0}
    A = np.array(step_ms) if step_ms else np.zeros(1)
    row = dict(
        variant=cell, seed=seed, steps_ran=sim.timestep,
        survivors=n_surv, found=len(sim.found),
        found_pct=round(100.0 * len(sim.found) / max(1, n_surv), 2),
        cov_pct=round(cov, 2), stair_pct=round(scv, 2),
        completed=int(t_all > 0),
        deadline=deadline,
        found_deadline=found_deadline,
        found_deadline_pct=round(100.0 * found_deadline / max(1, n_surv), 2),
        cov_deadline=round(cov_deadline, 2),
        t_first_found=t_first, t_half_found=t_half, t_all_found=t_all,
        t_cov90=t_cov90, t_stair50=t_stair50,
        deaths=sum(1 for r in sim.robots if not r.active),
        hazard_deaths=sum(1 for r in sim.robots if getattr(r, 'hazard_killed', False)),
        stranded_end=len(rep['stranded_at_end']),
        blackout_total=rep['blackout_ticks_total'],
        blackout_max=rep['blackout_ticks_max'],
        relay_duty_ticks=relay_duty,
        battery_used=round(float(sum(
            max(0.0, M.MAX_BATTERY - getattr(r, 'battery', M.MAX_BATTERY))
            for r in sim.robots)), 1),
        step_mean_ms=round(float(A.mean()), 2),
        step_p90_ms=round(float(np.percentile(A, 90)), 2),
        wall_s=round(time.time() - wall0, 1),
    )
    ts = dict(coverage_pct=cov_ts, found=found_ts, stair_pct=stair_ts,
              active_relays=relay_ts)
    return row, ts

# ══════════════════════════════════════════════════════════════════════════════
# Outputs
# ══════════════════════════════════════════════════════════════════════════════

def write_csvs(rows, ts_store, out):
    fields = list(rows[0].keys())
    with open(os.path.join(out, 'ablation_runs.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in rows: w.writerow(r)
    with open(os.path.join(out, 'ablation_timeseries.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['variant', 'seed', 'step', 'metric', 'value'])
        for (cell, seed), ts in ts_store.items():
            for metric, series in ts.items():
                for s, v in series:
                    w.writerow([cell, seed, s, metric, v])
    by = defaultdict(list)
    for r in rows: by[r['variant']].append(r)
    num_keys = [k for k in fields if k not in ('variant',)]
    with open(os.path.join(out, 'ablation_summary.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['variant', 'n_seeds'] + num_keys)
        for cell in ALL_CELLS:
            if cell not in by: continue
            g = by[cell]
            w.writerow([cell, len(g)] +
                       [round(float(np.mean([x[k] for x in g])), 3) for k in num_keys])
    with open(os.path.join(out, 'ablation_design.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['cell', 'axis', 'removed', 'isolates'])
        for d in DESIGN: w.writerow(d)


def make_plots(rows, ts_store, out):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    by = defaultdict(list)
    for r in rows: by[r['variant']].append(r)
    cells = [c for c in ALL_CELLS if c in by]

    def mean(cell, k): return float(np.mean([x[k] for x in by[cell]]))

    # 1 — REVIEWER HEADLINE deltas vs control, paired per seed.
    # found%% and coverage%% saturate with a generous fleet (every variant
    # eventually finds everyone) so they differentiate nothing — the metrics
    # that actually separate the components are: mission success AT A
    # DEADLINE, time-to-complete, fleet energy, and compute cost. Deltas are
    # computed PER SEED against the same seed's control run (paired design —
    # this is also the pairing a Wilcoxon signed-rank test should use).
    if 'A0-FULL' in by:
        others = [c for c in cells if c != 'A0-FULL']
        ctrl_by_seed = {r['seed']: r for r in by['A0-FULL']}

        def paired_deltas(cell, key, pct_of_ctrl):
            """Per-seed (variant − control); pct_of_ctrl -> %% of control value.
            For t_all_found, a −1 (never completed) is censored at that run's
            steps_ran — an UNDERestimate of how much worse it is."""
            out_d = []
            for r in by[cell]:
                c = ctrl_by_seed.get(r['seed'])
                if c is None: continue
                v = float(r[key]); cv = float(c[key])
                if key == 't_all_found':
                    if v < 0: v = float(r['steps_ran'])     # censored
                    if cv < 0: cv = float(c['steps_ran'])
                out_d.append(100.0 * (v - cv) / max(1e-9, cv) if pct_of_ctrl else v - cv)
            return out_d

        panels = [
            ('found_deadline_pct', False, 'Δ found%% at deadline (pp)\n← worse | better →', '#c0392b'),
            ('t_all_found',        True,  'Δ time-to-all-found (%% of control)\n← faster | slower →', '#8e44ad'),
            ('battery_used',       True,  'Δ fleet energy (%% of control)\n← cheaper | costlier →', '#e67e22'),
            ('step_mean_ms',       True,  'Δ compute ms/step (%% of control)\n← cheaper | costlier →', '#2980b9'),
        ]
        fig, axes = plt.subplots(2, 2, figsize=(15, 0.42 * len(others) + 7))
        for ax, (key, is_pct, xlabel, colr) in zip(axes.flat, panels):
            y = np.arange(len(others))
            means, alldots = [], []
            for c in others:
                d = paired_deltas(c, key, is_pct)
                means.append(float(np.mean(d)) if d else 0.0)
                alldots.append(d)
            ax.barh(y, means, 0.6, color=colr, alpha=0.85)
            for yy, dots in zip(y, alldots):
                for d in dots:
                    ax.plot(d, yy, 'k.', ms=4, alpha=0.55)
            ax.set_yticks(y); ax.set_yticklabels(others, fontsize=8); ax.invert_yaxis()
            ax.axvline(0, color='k', lw=1)
            ax.set_xlabel(xlabel, fontsize=9)
            ax.grid(alpha=0.3, axis='x')
        fig.suptitle('Ablation — reviewer headline metrics: paired per-seed deltas vs A0-FULL\n'
                     '(dots = individual seeds; DNF runs censored at step budget, so their bars are underestimates)',
                     fontsize=11)
        fig.tight_layout(); fig.savefig(os.path.join(out, 'ablation_deltas.png'), dpi=130)
        plt.close(fig)

    # 1b — secondary: the saturating outcome metrics, kept for completeness
    if 'A0-FULL' in by:
        base_f = mean('A0-FULL', 'found_pct'); base_c = mean('A0-FULL', 'cov_pct')
        others = [c for c in cells if c != 'A0-FULL']
        df = [mean(c, 'found_pct') - base_f for c in others]
        dc = [mean(c, 'cov_pct') - base_c for c in others]
        y = np.arange(len(others))
        fig, ax = plt.subplots(figsize=(11, 0.5 * len(others) + 2.5))
        ax.barh(y - 0.2, df, 0.38, label='Δ final found % (saturates — see deltas plot)', color='#c0392b')
        ax.barh(y + 0.2, dc, 0.38, label='Δ final coverage %', color='#2980b9')
        ax.set_yticks(y); ax.set_yticklabels(others); ax.invert_yaxis()
        ax.axvline(0, color='k', lw=1)
        ax.set_xlabel('change vs A0-FULL control (percentage points)')
        ax.set_title('Secondary outcome metrics (saturating — not headline)')
        ax.legend(fontsize=8); ax.grid(alpha=0.3, axis='x')
        fig.tight_layout(); fig.savefig(os.path.join(out, 'ablation_final_outcomes.png'), dpi=130)
        plt.close(fig)

    # 2 — trajectories (mean across seeds)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    cmap = plt.get_cmap('tab20')
    for i, cell in enumerate(cells):
        for ax, metric in zip(axes, ('coverage_pct', 'found')):
            per_seed = [dict(ts_store[(cell, r['seed'])][metric]) for r in by[cell]
                        if (cell, r['seed']) in ts_store]
            if not per_seed: continue
            steps = sorted(set().union(*[set(d) for d in per_seed]))
            ys = [float(np.mean([d[s] for d in per_seed if s in d])) for s in steps]
            kw = dict(color='black', lw=2.4, zorder=5) if cell == 'A0-FULL' \
                else dict(color=cmap(i % 20), lw=1.3, alpha=0.9)
            ax.plot(steps, ys, label=cell, **kw)
    axes[0].set_title('Union coverage %'); axes[1].set_title('Survivors found')
    for ax in axes:
        ax.set_xlabel('tick'); ax.grid(alpha=0.3)
    axes[1].legend(fontsize=7, ncol=2)
    fig.suptitle('Ablation trajectories (mean over seeds; control in black)')
    fig.tight_layout(); fig.savefig(os.path.join(out, 'ablation_curves.png'), dpi=130)
    plt.close(fig)

    # 3 — interaction 2x2
    have = {k: v for k, v in INTERACTION.items() if v in by}
    if len(have) == 4:
        labels = ['FULL', 'PG-ONLY', 'HRBA-ONLY', 'NEITHER']
        f_vals = [mean(INTERACTION[l], 'found_deadline_pct') for l in labels]
        c_vals = [mean(INTERACTION[l], 'cov_pct') for l in labels]
        x = np.arange(4)
        fig, ax = plt.subplots(figsize=(9, 5.5))
        ax.bar(x - 0.2, f_vals, 0.38, label='found % at deadline', color='#c0392b')
        ax.bar(x + 0.2, c_vals, 0.38, label='coverage %', color='#2980b9')
        for xi, l in zip(x, labels):
            for r in by[INTERACTION[l]]:
                ax.plot(xi - 0.2, r['found_deadline_pct'], 'k.', ms=5, alpha=0.6)
                ax.plot(xi + 0.2, r['cov_pct'], 'k.', ms=5, alpha=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([f'{l}\n({INTERACTION[l]})' for l in labels])
        ax.set_ylabel('%'); ax.set_ylim(0, 105)
        ax.set_title('HRBA × potential-game interaction — do the subsystems need each other?')
        ax.legend(); ax.grid(alpha=0.3, axis='y')
        # Additivity annotation: does FULL exceed the sum of the marginal gains?
        add_f = f_vals[3] + (f_vals[1] - f_vals[3]) + (f_vals[2] - f_vals[3])
        ax.axhline(add_f, color='gray', ls='--', lw=1)
        ax.text(3.42, add_f, ' additive\n prediction\n (found@deadline %)', fontsize=7,
                va='center', color='gray')
        fig.tight_layout(); fig.savefig(os.path.join(out, 'ablation_interaction.png'), dpi=130)
        plt.close(fig)

# ══════════════════════════════════════════════════════════════════════════════

def main():
    global LARGE_ENV
    p = argparse.ArgumentParser()
    p.add_argument('--cells', type=str, default='all',
                   help="comma list of cell ids, or 'all'. See ablation_design.csv.")
    p.add_argument('--seeds', type=str, default='3',
                   help="count ('3' -> 0,1,2) or explicit list ('1,2,7')")
    p.add_argument('--steps', type=int, default=1500,
                   help="default raised for the large environment (200x200 needs "
                        "more ticks to converge than the old 128x128 default)")
    p.add_argument('--out', type=str, default='ablation_out')
    p.add_argument('--deadline', type=int, default=250,
                   help="rescue deadline (ticks) for the headline success "
                        "metric found%%-at-deadline. Final found%% saturates at "
                        "100%% with a generous fleet; success at a deadline is "
                        "the SAR-relevant success measure. Default 250 (~the "
                        "control's typical completion time in the large env).")
    p.add_argument('--plain-env', action='store_true',
                   help="disable the hazard-dense environment (4-8 buildings, "
                        "lethal temp/rad zones + small hotspot scatter) and use "
                        "the plain 4-building near-benign hazard field instead.")
    p.add_argument('--small', action='store_true',
                   help="use the small default environment (~10 robots, 18 "
                        "survivors, 128x128) instead of the large one "
                        "(50 robots, 45 survivors, 200x200, matches "
                        "ComparisonBenchLarge.py). Faster but more forgiving — "
                        "components differentiate less clearly.")
    p.add_argument('--quick', action='store_true',
                   help='control + A2 + B3 + B7 + X-NEITHER, 1 seed, 250 steps, '
                        'small environment (fast sanity check only)')
    a = p.parse_args()

    if a.quick:
        a.cells = 'A0-FULL,A2-GREEDY,B3-NOMF,B7-NOGAME,X-NEITHER'
        a.seeds = '1'; a.steps = 250; a.small = True
    global HAZARD_ENV
    LARGE_ENV = not a.small
    # Hazard env is tied to the LARGE environment: its spawn-clearance
    # guardrail protects the large builder's 9 spawn clusters; the small env
    # spawns elsewhere and would risk tick-1 hotspot deaths.
    HAZARD_ENV = (not a.plain_env) and LARGE_ENV
    cells = ALL_CELLS if a.cells == 'all' else [c.strip() for c in a.cells.split(',')]
    unknown = [c for c in cells if c not in ALL_CELLS]
    if unknown:
        print("unknown cells:", unknown, "\nvalid:", ALL_CELLS); sys.exit(1)
    seeds = ([int(s) for s in a.seeds.split(',')] if ',' in a.seeds
             else list(range(int(a.seeds))))
    if a.steps < a.deadline:
        print(f"  [WARNING] --steps {a.steps} < --deadline {a.deadline}: the "
              f"'found_deadline' column will record end-of-run values, NOT "
              f"deadline values. Raise --steps or lower --deadline.")
    os.makedirs(a.out, exist_ok=True)

    env_desc = (f"LARGE  {LARGE_GRID_W}x{LARGE_GRID_H}, "
                f"{sum(LARGE_ROBOTS.values())} robots, {LARGE_N_SURVIVORS} survivors" if LARGE_ENV else
                "SMALL  default env (~10 robots, 18 survivors, 128x128)")
    env_desc += ("  | HAZARD-DENSE (4-8 buildings, lethal temp/rad zones + hotspot scatter)"
                 if HAZARD_ENV else "  | plain hazard field")
    print(f"Framework M ablation | env={env_desc}")
    print(f"  cells={cells} | seeds={seeds} | steps={a.steps}")
    print("design:")
    for d in DESIGN:
        if d[0] in cells:
            print(f"  {d[0]:<10} [{d[1]:<11}] removes: {d[2]}")

    rows, ts_store = [], {}
    t0 = time.time()
    total_runs = len(cells) * len(seeds)
    run_idx = 0
    for cell in cells:
        for seed in seeds:
            run_idx += 1
            overall_prefix = f'Overall [{run_idx}/{total_runs}]  '
            try:
                row, ts = run_one(cell, seed, a.steps, a.deadline, overall_prefix=overall_prefix)
            except Exception as e:
                _clear()
                import traceback; print(f"  {cell:<10} seed={seed}  FAIL:", e); traceback.print_exc()
                continue
            rows.append(row); ts_store[(cell, seed)] = ts
            print(f"  {cell:<10} seed={seed}  "
                  f"found@{a.deadline}={row['found_deadline']}/{row['survivors']}"
                  f"  t_all={row['t_all_found'] if row['t_all_found']>0 else 'DNF'}"
                  f"  energy={row['battery_used']:.0f}"
                  f"  step={row['step_mean_ms']:.1f}ms"
                  f"  cov={row['cov_pct']:5.1f}%")
    print(f"total {(time.time() - t0) / 60:.1f} min")
    if not rows:
        print("no rows — aborting"); return

    # ── REVIEWER HEADLINE table ─────────────────────────────────────────────
    # The metrics to put in front of reviewers, as paired per-seed deltas vs
    # the control. Final found%/coverage% saturate with a generous fleet and
    # belong in an appendix; these four differentiate the components:
    #   found@deadline  — mission success under time pressure (SAR-relevant)
    #   t_all_found     — time to complete the rescue (DNF censored at budget)
    #   battery_used    — fleet energy cost
    #   step_mean_ms    — coordination compute cost
    _by = defaultdict(list)
    for r in rows: _by[r['variant']].append(r)
    if 'A0-FULL' in _by:
        _ctrl = {r['seed']: r for r in _by['A0-FULL']}
        def _pd(cell, key, pct):
            ds = []
            for r in _by[cell]:
                c = _ctrl.get(r['seed'])
                if c is None: continue
                v, cv = float(r[key]), float(c[key])
                if key == 't_all_found':
                    if v < 0: v = float(r['steps_ran'])
                    if cv < 0: cv = float(c['steps_ran'])
                ds.append(100.0*(v-cv)/max(1e-9, cv) if pct else v-cv)
            return float(np.mean(ds)) if ds else 0.0
        print("\n  REVIEWER HEADLINE — paired per-seed deltas vs A0-FULL")
        print("  (negative found@deadline = worse; positive time/energy/compute = worse)")
        print(f"  {'variant':<12} {'Δfound@dl (pp)':>15} {'Δt_all (%)':>11} "
              f"{'Δenergy (%)':>12} {'Δcompute (%)':>13}  DNF")
        for c in [x for x in _by if x != 'A0-FULL']:
            dnf = sum(1 for r in _by[c] if float(r['t_all_found']) < 0)
            print(f"  {c:<12} {_pd(c,'found_deadline_pct',False):>15.1f} "
                  f"{_pd(c,'t_all_found',True):>11.0f} "
                  f"{_pd(c,'battery_used',True):>12.0f} "
                  f"{_pd(c,'step_mean_ms',True):>13.0f}  "
                  f"{dnf}/{len(_by[c])}")
        n_seed_set = len({r['seed'] for r in rows})
        if n_seed_set >= 5:
            print("  (>=5 seeds: report Wilcoxon signed-rank on these per-seed pairs)")
        else:
            print(f"  (only {n_seed_set} seed(s) — run >=5 before citing any of this)")

    write_csvs(rows, ts_store, a.out)
    try:
        make_plots(rows, ts_store, a.out)
    except Exception as e:
        print("plotting failed:", e)
    print(f"\nwrote -> {a.out}/: ablation_runs.csv, ablation_timeseries.csv, "
          f"ablation_summary.csv, ablation_design.csv,\n"
          f"          ablation_deltas.png, ablation_curves.png, ablation_interaction.png")


if __name__ == '__main__':
    main()