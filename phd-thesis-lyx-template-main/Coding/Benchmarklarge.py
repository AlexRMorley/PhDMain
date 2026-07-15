"""
Large-Scale Fleet Sim Benchmark
================================
Configuration:
  Grid    : 200 × 200 cells @ 5m/cell = 1,000m × 1,000m = 1 km²
  Robots  : 50  (13 Legged, 17 Drone, 8 Boat, 12 Rover)
  Zones   : 10 × 10 = 100 zones (20-cell zones, ZONE_CHUNKS=10)
  Battery : 2500 (scaled from 1000 @ 128×128 by area ratio ~2.4×)
  Steps   : 2000 (equivalent mission horizon to 1500 on small map)

Purpose: Demonstrate scalability of TLC-CBBA + potential game at R=50.
Produces comparison-ready output against R=12 baseline.
"""

import sys, os, random, json, time, math
from unittest.mock import MagicMock

# ── Headless pygame stub ──────────────────────────────────────────────────────
pg = MagicMock()
for m in ['pygame', 'pygame.display', 'pygame.font']:
    sys.modules[m] = pg
pg.SRCALPHA = 0
pg.Surface  = lambda *a, **k: MagicMock()
pg.Rect     = lambda *a, **k: MagicMock()

import numpy as np
import importlib.util
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

# ── Configuration ─────────────────────────────────────────────────────────────
N_SEEDS   = (10,11,12,13,14,15,16,17,18,19)           # fewer seeds — each run is longer
MAX_STEPS = 2000
SEEDS     = list(N_SEEDS)
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
SIM_FILE  = os.path.join(BASE_DIR, "2DFleetFrameworkK.py")
OUT_DIR   = os.path.join(BASE_DIR, "benchmark_large_results_10")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Large-scale parameters ────────────────────────────────────────────────────
LARGE_GRID_W     = 200
LARGE_GRID_H     = 200
LARGE_CELL_SIZE  = 5        # 5m cells → 1km × 1km
LARGE_ZONE_CHUNKS = 10      # zone = 10×2 = 20 cells → 10 zones/axis = 100 total
LARGE_MAX_BATTERY = 2500    # scaled for larger map
LARGE_ROBOTS = {
    "Legged": 13,
    "Drone":  17,
    "Boat":    8,
    "Rover":  12,
}  # total = 50
LARGE_N_SURVIVORS = 45      # ~proportional to area (18 → 45 for 1km²)
LARGE_MAX_BUNDLE  = 8       # more zones per robot on larger map

# ── Load and patch sim module ─────────────────────────────────────────────────
spec = importlib.util.spec_from_file_location("hsim", SIM_FILE)
M    = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)

# Patch module-level constants BEFORE any FleetSim is instantiated
M.GRID_W        = LARGE_GRID_W
M.GRID_H        = LARGE_GRID_H
M.CELL_SIZE     = LARGE_CELL_SIZE
M.ZONE_CHUNKS   = LARGE_ZONE_CHUNKS
M.MAX_BATTERY   = LARGE_MAX_BATTERY
M.MAX_BUNDLE    = LARGE_MAX_BUNDLE

# Patch _build_robots desired counts
_orig_build_robots = M.FleetSim._build_robots
def _patched_build_robots(self):
    """Override robot counts for large-scale benchmark."""
    import random as _random
    templates = [
        ("Legged", {M.Capability.LAND, M.Capability.STAIRS},
         np.array([10., 10.]), (M.TEMP_LIMIT, M.RAD_LIMIT)),
        ("Drone",  {M.Capability.AIR},
         np.array([10., 10.]), (M.TEMP_LIMIT, M.RAD_LIMIT)),
        ("Boat",   {M.Capability.WATER},
         np.array([0., 0.]),   (9999., 9999.)),
        ("Rover",  {M.Capability.LAND},
         np.array([-2., -2.]), (9999., 9999.)),
    ]
    tpl = {n: (n, c, w, l) for n, c, w, l in templates}
    spawn = []
    for t, n in LARGE_ROBOTS.items():
        spawn += [t] * n
    _random.shuffle(spawn)

    W, H = M.GRID_W, M.GRID_H
    # 9 spawn clusters spread across the map (3×3 grid)
    clusters = [
        (W//6,   H//6),   (W//2, H//6),   (5*W//6, H//6),
        (W//6,   H//2),   (W//2, H//2),   (5*W//6, H//2),
        (W//6, 5*H//6),   (W//2, 5*H//6), (5*W//6, 5*H//6),
    ]
    water_cells = [(x, y) for x in range(W) for y in range(H)
                   if self.world.grid[x][y]["t"] == M.T_WATER]

    self.robots = []
    for i, tname in enumerate(spawn):
        _, caps, weights, (tlim, rlim) = tpl[tname]
        name = f"{tname}{i}"
        center = clusters[i % len(clusters)]

        if tname == "Boat" and water_cells:
            sx, sy = _random.choice(water_cells)
        else:
            sx, sy = center
            for _ in range(50):
                cx = max(1, min(W-2, center[0] + _random.randint(-12, 12)))
                cy = max(1, min(H-2, center[1] + _random.randint(-12, 12)))
                tt = self.world.grid[cx][cy]["t"]
                if (tt == M.T_FREE or
                        (tt == M.T_STAIRS and M.Capability.STAIRS in caps)):
                    sx, sy = cx, cy; break
        self.robots.append(
            M.Robot(name, sx, sy, caps, self.world, self, weights, tlim, rlim))

M.FleetSim._build_robots = _patched_build_robots

# Patch survivor count
_orig_build_survivors = M.FleetSim._build_survivors
def _patched_build_survivors(self):
    W, H = M.GRID_W, M.GRID_H
    free_open   = [(x, y) for x in range(W) for y in range(H)
                   if self.world.grid[x][y]["t"] == M.T_FREE]
    free_stair  = [(x, y) for x in range(W) for y in range(H)
                   if self.world.grid[x][y]["t"] == M.T_STAIRS]

    # Target: ~40% inside buildings (require relay to find),
    # ~60% open terrain but spread across the full map
    n_total   = LARGE_N_SURVIVORS
    n_building = max(3, round(n_total * 0.40))
    n_open     = n_total - n_building

    # Building survivors — sample from stair cells
    building_survivors = random.sample(free_stair,
                                       min(n_building, len(free_stair)))

    # Open survivors — divide map into a grid and place one per cell
    # so they're spread across the whole 1km² not clustered
    grid_k = round(n_open ** 0.5)  # e.g. 6×6 = 36 open survivors
    cell_w = W // grid_k; cell_h = H // grid_k
    open_survivors = []
    for gx in range(grid_k):
        for gy in range(grid_k):
            x0 = gx * cell_w; x1 = min(W, x0 + cell_w)
            y0 = gy * cell_h; y1 = min(H, y0 + cell_h)
            pool = [(x, y) for x, y in free_open
                    if x0 <= x < x1 and y0 <= y < y1]
            if pool:
                open_survivors.append(random.choice(pool))
            if len(open_survivors) >= n_open:
                break
        if len(open_survivors) >= n_open:
            break

    self.survivors = building_survivors + open_survivors

M.FleetSim._build_survivors = _patched_build_survivors

FleetSim = M.FleetSim
Role      = M.Role

# ── Progress bar ──────────────────────────────────────────────────────────────
def progress_bar(current, total, prefix='', suffix='', bar_len=40):
    filled = int(bar_len * current / max(total, 1))
    bar    = '█' * filled + '░' * (bar_len - filled)
    pct    = 100.0 * current / max(total, 1)
    sys.stdout.write(f'\r{prefix} [{bar}] {pct:5.1f}%  {suffix}')
    sys.stdout.flush()

# ── Result container ──────────────────────────────────────────────────────────
class Result:
    def __init__(self, seed):
        self.seed            = seed
        self.completed       = False
        self.completion_step = None
        self.found_over_time = []
        self.coverage_over_time = []
        self.step_times_ms   = []
        self.role_over_time  = []
        self.deaths_over_time = []
        self.buildings_entered = 0
        self.buildings_total   = 0
        self.n_survivors     = 0

# ── Main benchmark loop ───────────────────────────────────────────────────────
print(f"\nLarge-Scale Fleet Sim Benchmark — {N_SEEDS} seeds × {MAX_STEPS} steps")
print(f"Grid: {LARGE_GRID_W}×{LARGE_GRID_H} @ {LARGE_CELL_SIZE}m = "
      f"{LARGE_GRID_W*LARGE_CELL_SIZE}m × {LARGE_GRID_H*LARGE_CELL_SIZE}m = "
      f"{(LARGE_GRID_W*LARGE_CELL_SIZE/1000)**2:.2f} km²")
print(f"Robots: {sum(LARGE_ROBOTS.values())} "
      f"({', '.join(f'{n} {t}' for t,n in LARGE_ROBOTS.items())})")
print(f"Survivors: {LARGE_N_SURVIVORS}  |  Zones: "
      f"{LARGE_GRID_W//( LARGE_ZONE_CHUNKS*2)}×"
      f"{LARGE_GRID_H//(LARGE_ZONE_CHUNKS*2)} = "
      f"{(LARGE_GRID_W//(LARGE_ZONE_CHUNKS*2))**2}")
print("─" * 70)

results = []
wall_start = time.time()

for seed in SEEDS:
    random.seed(seed); np.random.seed(seed)
    sim = FleetSim()
    res = Result(seed)
    res.n_survivors = len(sim.survivors)

    prev_found = 0
    t_start = time.time()

    for step in range(1, MAX_STEPS + 1):
        t0 = time.perf_counter()
        sim.step()
        dt = (time.perf_counter() - t0) * 1000
        res.step_times_ms.append((step, dt))

        n_found = len(sim.found)
        cov     = float(np.mean(sim.union_belief != M.T_UNKNOWN)) * 100
        active  = sum(1 for r in sim.robots if r.active)
        deaths  = sum(1 for r in sim.robots if not r.active)

        if step % 10 == 0:
            res.found_over_time.append((step, n_found))
            res.coverage_over_time.append((step, cov))
            res.deaths_over_time.append((step, deaths))
            from collections import Counter
            roles = Counter(r.role.name for r in sim.robots if r.active)
            res.role_over_time.append((step, dict(roles)))

        if n_found == res.n_survivors and not res.completed:
            res.completed       = True
            res.completion_step = step
            break

        # Stop if no robots remain active
        if active == 0:
            break

        # Stop if fully explored but stalled — no new survivors for 200 steps
        if cov >= 99.9 and step > 200:
            recent_found = [v for s, v in res.found_over_time if s >= step - 200]
            if recent_found and recent_found[-1] == recent_found[0]:
                break

        if step % 100 == 0:
            wall = time.time() - t_start
            mean_ms = np.mean([ms for _, ms in res.step_times_ms[-100:]])
            eta = mean_ms * (MAX_STEPS - step) / 1000
            progress_bar(step, MAX_STEPS,
                         prefix=f'seed {seed:2d}',
                         suffix=f'{n_found}/{res.n_survivors} found  '
                                f'{cov:.0f}%cov  {active}alive  '
                                f'{mean_ms:.0f}ms/step  ETA {eta:.0f}s')

    # Buildings entered
    stair_clusters = set()
    for r in sim.robots:
        if r.active or True:  # all robots including dead
            for x in range(M.GRID_W):
                for y in range(M.GRID_H):
                    pass  # simplified — count via zone type
    building_zones = [z for z, zt in sim._shadow_zone_type.items()
                      if zt == 'stair']
    entered = sum(1 for z in building_zones
                  if sim.zone_stats(z)['unknown_frac'] < 0.90)
    res.buildings_entered = entered
    res.buildings_total   = len(building_zones)

    wall = time.time() - t_start
    mean_ms = np.mean([ms for _, ms in res.step_times_ms])
    max_ms  = max(ms for _, ms in res.step_times_ms)
    deaths  = sum(1 for r in sim.robots if not r.active)
    status  = f'✓ t={res.completion_step}' if res.completed else \
              f'✗ {len(sim.found)}/{res.n_survivors} found'

    print(f'\n  seed {seed:2d}  {status:20s}  '
          f'cov={cov:.1f}%  bldgs={entered}/{res.buildings_total}  '
          f'deaths={deaths}  {mean_ms:.1f}ms/step (max {max_ms:.0f}ms)  '
          f'wall={wall:.0f}s')
    results.append(res)

print("\n" + "─" * 70)
total_wall = time.time() - wall_start
print(f"Total wall time: {total_wall:.0f}s  ({total_wall/60:.1f} min)")

# ── Summary stats ─────────────────────────────────────────────────────────────
completed  = [r for r in results if r.completed]
comp_steps = [r.completion_step for r in completed]
avg_ms_all = [np.mean([ms for _, ms in r.step_times_ms]) for r in results]
max_ms_all = [max(ms for _, ms in r.step_times_ms) for r in results]

summary = {
    "configuration": {
        "grid":       f"{LARGE_GRID_W}×{LARGE_GRID_H}",
        "cell_m":     LARGE_CELL_SIZE,
        "area_km2":   round((LARGE_GRID_W * LARGE_CELL_SIZE / 1000) ** 2, 3),
        "n_robots":   sum(LARGE_ROBOTS.values()),
        "robot_mix":  LARGE_ROBOTS,
        "n_survivors": LARGE_N_SURVIVORS,
        "max_battery": LARGE_MAX_BATTERY,
        "n_zones":    (LARGE_GRID_W // (LARGE_ZONE_CHUNKS * 2)) ** 2,
    },
    "n_seeds":          N_SEEDS,
    "max_steps":        MAX_STEPS,
    "n_completed":      len(completed),
    "completion_rate_pct": round(100 * len(completed) / max(N_SEEDS, 1), 1),
    "completion_steps": {r.seed: r.completion_step for r in completed},
    "computational_efficiency": {
        "mean_ms_per_step": round(float(np.mean(avg_ms_all)), 2),
        "std_ms_per_step":  round(float(np.std(avg_ms_all)), 2),
        "max_ms_per_step":  round(float(np.max(max_ms_all)), 2),
        "total_wall_s":     round(total_wall, 1),
    },
}
if comp_steps:
    summary["mean_completion"]   = float(np.mean(comp_steps))
    summary["median_completion"] = float(np.median(comp_steps))
    summary["min_completion"]    = int(np.min(comp_steps))
    summary["max_completion"]    = int(np.max(comp_steps))

with open(os.path.join(OUT_DIR, "summary_large.json"), "w") as f:
    json.dump(summary, f, indent=2)

# ── Plots ──────────────────────────────────────────────────────────────────────
PALETTE = plt.cm.tab10.colors

# 1. Survivors found over time
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax = axes[0]
for i, res in enumerate(results):
    xs = [s for s, _ in res.found_over_time]
    ys = [n for _, n in res.found_over_time]
    lbl = f'seed {res.seed}' + (' ✓' if res.completed else ' ✗')
    ax.plot(xs, ys, color=PALETTE[i % 10], label=lbl, linewidth=1.5)
ax.axhline(LARGE_N_SURVIVORS, color='black', linestyle='--', alpha=0.4,
           label=f'All {LARGE_N_SURVIVORS} found')
ax.set_xlabel('Step'); ax.set_ylabel('Survivors Found')
ax.set_title(f'Survivors Found — R=50, 1km²')
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# 2. Coverage over time
ax = axes[1]
for i, res in enumerate(results):
    xs = [s for s, _ in res.coverage_over_time]
    ys = [c for _, c in res.coverage_over_time]
    ax.plot(xs, ys, color=PALETTE[i % 10], linewidth=1.5)
ax.set_xlabel('Step'); ax.set_ylabel('Union Coverage (%)')
ax.set_title('Map Coverage Over Time')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "01_survivors_coverage.png"), dpi=150)
plt.close()

# 3. Step time distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax = axes[0]
all_times = [ms for res in results for _, ms in res.step_times_ms]
ax.hist(all_times, bins=60, color='steelblue', alpha=0.7, edgecolor='white')
ax.axvline(np.mean(all_times), color='red', linestyle='--',
           label=f'Mean: {np.mean(all_times):.0f}ms')
ax.axvline(np.percentile(all_times, 95), color='orange', linestyle='--',
           label=f'P95: {np.percentile(all_times, 95):.0f}ms')
ax.set_xlabel('ms/step'); ax.set_ylabel('Count')
ax.set_title(f'Step Time Distribution — R=50')
ax.legend(); ax.grid(alpha=0.3)

# 4. Scalability comparison: R=12 vs R=50
ax = axes[1]
r_vals   = [12, 50]
mean_ms  = [30, float(np.mean(avg_ms_all))]
# Theoretical O(R²) line anchored at R=12
r_theory = list(range(10, 55, 5))
ms_theory = [30 * (r/12)**2 for r in r_theory]
ms_tlc    = [30 * (r/12)**1.4 for r in r_theory]  # TLC-CBBA sub-quadratic

ax.plot(r_theory, ms_theory, 'r--', alpha=0.6, label='Flat CBBA O(R²) theory')
ax.plot(r_theory, ms_tlc,    'g--', alpha=0.6, label='TLC-CBBA O(R^1.4) theory')
ax.scatter(r_vals, mean_ms, s=120, zorder=5, color=['steelblue', 'darkorange'],
           label='Measured')
for r, ms in zip(r_vals, mean_ms):
    ax.annotate(f'R={r}\n{ms:.0f}ms', (r, ms),
                textcoords='offset points', xytext=(8, 5), fontsize=9)
ax.set_xlabel('Fleet size R'); ax.set_ylabel('Mean ms/step')
ax.set_title('Scalability: R=12 vs R=50')
ax.legend(fontsize=8); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "02_timing_scalability.png"), dpi=150)
plt.close()

# 5. Role distribution over time (average across seeds)
fig, ax = plt.subplots(figsize=(10, 5))
role_names = ['SCOUT', 'SCAN', 'RELAY', 'LOITER']
role_colors = {'SCOUT': '#2196F3', 'SCAN': '#4CAF50',
               'RELAY': '#FF9800', 'LOITER': '#9E9E9E'}
all_steps = sorted(set(s for res in results for s, _ in res.role_over_time))
for rname in role_names:
    means = []
    for s in all_steps:
        vals = [d.get(rname, 0) for res in results
                for ss, d in res.role_over_time if ss == s]
        means.append(np.mean(vals) if vals else 0)
    ax.plot(all_steps, means, label=rname, color=role_colors[rname], linewidth=2)
ax.set_xlabel('Step'); ax.set_ylabel('Avg robots in role')
ax.set_title(f'Role Distribution Over Time — R=50')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "03_role_distribution.png"), dpi=150)
plt.close()

# ── Final summary print ───────────────────────────────────────────────────────
print(f"\nLarge-Scale Results Summary:")
print(f"  Completion rate: {len(completed)}/{N_SEEDS} "
      f"({summary['completion_rate_pct']}%)")
if comp_steps:
    print(f"  Completion steps: mean={np.mean(comp_steps):.0f}  "
          f"median={np.median(comp_steps):.0f}  "
          f"range={np.min(comp_steps)}–{np.max(comp_steps)}")
print(f"  Mean ms/step:  {summary['computational_efficiency']['mean_ms_per_step']:.1f}")
print(f"  Max  ms/step:  {summary['computational_efficiency']['max_ms_per_step']:.1f}")
print(f"  Total wall:    {total_wall:.0f}s ({total_wall/60:.1f} min)")
print(f"\nAll outputs saved to: {OUT_DIR}/")
print(f"  summary_large.json")
print(f"  01_survivors_coverage.png")
print(f"  02_timing_scalability.png")
print(f"  03_role_distribution.png")

# Scalability comparison print
print(f"\nScalability comparison:")
print(f"  R=12 baseline:  ~30ms/step")
print(f"  R=50 measured:  {np.mean(avg_ms_all):.0f}ms/step")
print(f"  Scaling factor: {np.mean(avg_ms_all)/30:.1f}x  "
      f"(O(R²) naive would be {(50/12)**2:.1f}x)")