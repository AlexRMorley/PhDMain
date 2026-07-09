"""
Fleet Sim Benchmark Runner
Runs N seeds up to MAX_STEPS, collects stats, saves graphs to ./benchmark_results/

New in this version:
  - Progress bar with ETA
  - Buildings entered per seed
  - Computational efficiency: measured ms/step, theoretical O(·) analysis
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
import matplotlib.gridspec as gridspec
from collections import defaultdict

# ── Config ────────────────────────────────────────────────────────────────────
N_SEEDS    = 5
MAX_STEPS  = 2500
SEEDS      = list(range(N_SEEDS))
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
SIM_FILE   = os.path.join(BASE_DIR, "2DFleetFrameworkI.py")
OUT_DIR    = os.path.join(BASE_DIR, "benchmark_results_I")

# ── Load sim ──────────────────────────────────────────────────────────────────
spec = importlib.util.spec_from_file_location("hsim", SIM_FILE)
M    = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)
FleetSim, Role = M.FleetSim, M.Role

os.makedirs(OUT_DIR, exist_ok=True)

# ── Progress bar ──────────────────────────────────────────────────────────────
def progress_bar(current, total, prefix='', suffix='', bar_len=40):
    """Progress bar — sys.stdout.write for reliable Windows/PowerShell output."""
    filled = int(bar_len * current / max(total, 1))
    bar    = '█' * filled + '░' * (bar_len - filled)
    pct    = 100.0 * current / max(total, 1)
    sys.stdout.write(f'\r{prefix} [{bar}] {pct:5.1f}%  {suffix}')
    sys.stdout.flush()

# ── Complexity analysis ────────────────────────────────────────────────────────
def analyse_complexity(sim):
    """
    Return a dict of key size parameters and a theoretical O(·) string.

    The dominant costs per step are:
      A* planning  : O(W·H·log(W·H))  per robot per replanning tick
      CBBA auction : O(R² · Z)        every 50 ticks  (R robots, Z zones)
      Belief update: O(R · W · H)     (numpy union ops)
      Frontier BFS : O(Z · W·H/Z)     = O(W·H) total

    Combined per-step (amortised):  O(R · W·H · log(W·H))
    """
    W = M.GRID_W; H = M.GRID_H
    R = len(sim.robots)
    Z = sim.zone_nx * sim.zone_ny
    stair_cells = int(np.sum(sim._world_stair_arr))
    shadow_cells = int(np.sum(sim.radio_shadow))

    wh      = W * H
    wh_log  = wh * math.log2(wh)
    cbba_z  = R * R * Z          # per CBBA call (every 50 steps, amortised /50)
    belief  = R * wh
    astar   = R * wh_log         # upper bound: full replan every step

    return {
        "grid_W":          W,
        "grid_H":          H,
        "grid_cells":      wh,
        "n_robots":        R,
        "n_zones":         Z,
        "stair_cells":     stair_cells,
        "shadow_cells":    shadow_cells,
        "O_astar_per_step":      f"O(R·WH·log(WH)) = O({R}·{wh}·{math.log2(wh):.1f}) ≈ {astar:,.0f}",
        "O_cbba_amortised":      f"O(R²·Z/50) = O({R}²·{Z}/50) ≈ {cbba_z/50:,.0f} / step",
        "O_belief_per_step":     f"O(R·WH) = O({R}·{wh}) = {belief:,}",
        "O_combined_per_step":   f"O(R · WH · log(WH))  with R={R}, WH={wh:,}",
        "dominant_term":         f"{astar:,.0f}  ops/step (A* bound)",
    }


# ── Per-seed data collection ───────────────────────────────────────────────────
class SeedResult:
    def __init__(self, seed):
        self.seed                    = seed
        self.completed               = False
        self.completion_step         = None
        self.total_survivors         = 0
        self.found_over_time         = []    # (step, n_found)
        self.coverage_over_time      = []    # (step, pct)
        self.deaths                  = []    # (step, name, reason)
        self.robot_roles_over_time   = []    # (step, {role: count})
        self.n_robots                = 0
        self.final_found             = 0
        self.final_coverage          = 0.0
        self.wall_time_s             = 0.0
        # new
        self.buildings_entered       = 0     # distinct buildings with ≥1 explorer entry
        self.buildings_total         = 0     # total buildings on map
        self.buildings_coverage      = []    # (step, n_entered)  sampled every RECORD_EVERY
        self.step_times_ms           = []    # wall-clock ms per step (sampled)
        self.complexity              = {}    # from analyse_complexity()

RECORD_EVERY = 20

# ── Helper: count distinct buildings entered ──────────────────────────────────
# Precomputed per-sim cache — components are static, only belief changes
_bldg_components_cache = {}   # sim_id -> list of (x0,x1,y0,y1) bounding boxes

def count_buildings_entered(sim):
    """
    Fast numpy version — precomputes building bounding boxes once per sim,
    then checks union_belief slices each call. ~20x faster than BFS version.
    Returns (n_entered, n_total).
    """
    sid = id(sim)
    if sid not in _bldg_components_cache:
        # Build connected components of T_STAIRS cells once (static world)
        stair = np.array([[sim.world.grid[x][y]['t'] == M.T_STAIRS
                           for y in range(M.GRID_H)]
                          for x in range(M.GRID_W)], dtype=bool)
        visited = np.zeros((M.GRID_W, M.GRID_H), dtype=bool)
        boxes = []
        for x in range(M.GRID_W):
            for y in range(M.GRID_H):
                if not stair[x, y] or visited[x, y]: continue
                # BFS — runs once per sim, not per call
                q = [(x, y)]; comp = []
                while q:
                    cx, cy = q.pop()
                    if visited[cx, cy]: continue
                    visited[cx, cy] = True; comp.append((cx, cy))
                    for nx, ny in sim.world.neighbours((cx, cy)):
                        if stair[nx, ny] and not visited[nx, ny]:
                            q.append((nx, ny))
                if comp:
                    xs = [c[0] for c in comp]; ys = [c[1] for c in comp]
                    boxes.append((min(xs), max(xs)+1, min(ys), max(ys)+1))
        _bldg_components_cache[sid] = boxes

    boxes = _bldg_components_cache[sid]
    entered = sum(
        1 for x0, x1, y0, y1 in boxes
        if np.any(sim.union_belief[x0:x1, y0:y1] != M.T_UNKNOWN)
    )
    return entered, len(boxes)


# ══════════════════════════════════════════════════════════════════════════════
# Main benchmark loop
# ══════════════════════════════════════════════════════════════════════════════
results      = []
failed_seeds = []
total_steps  = N_SEEDS * MAX_STEPS
done_steps   = 0
run_start    = time.time()

print(f"Fleet Sim Benchmark — {N_SEEDS} seeds × {MAX_STEPS} steps")
print(f"{'─'*60}")

for seed_idx, seed in enumerate(SEEDS):
    random.seed(seed); np.random.seed(seed)
    sim = FleetSim()
    res = SeedResult(seed)
    res.total_survivors = len(sim.survivors)
    res.n_robots        = len(sim.robots)
    res.complexity      = analyse_complexity(sim)

    # Snapshot total buildings at init
    _, res.buildings_total = count_buildings_entered(sim)

    prev_dead   = set()
    t0          = time.time()
    step_t_acc  = []   # accumulate step times for sampling

    for step in range(1, MAX_STEPS + 1):
        step_start = time.perf_counter()
        sim.step()
        step_ms = (time.perf_counter() - step_start) * 1000
        step_t_acc.append(step_ms)

        # Track deaths
        current_dead = {r.name for r in sim.robots if not r.active}
        new_dead = current_dead - prev_dead
        for name in new_dead:
            r = next(rr for rr in sim.robots if rr.name == name)
            reason = getattr(r, 'death_reason', 'unknown') or 'unknown'
            res.deaths.append((step, name, reason))
        prev_dead = current_dead

        # Record metrics every N steps
        if step % RECORD_EVERY == 0 or step == 1:
            union = sim.union_belief
            cov   = float(np.sum(union != M.T_UNKNOWN)) / (M.GRID_W * M.GRID_H) * 100
            res.found_over_time.append((step, len(sim.found)))
            res.coverage_over_time.append((step, cov))

            role_counts = defaultdict(int)
            for r in sim.robots:
                if r.active: role_counts[r.role.name] += 1
            res.robot_roles_over_time.append((step, dict(role_counts)))

            n_ent, _ = count_buildings_entered(sim)
            res.buildings_coverage.append((step, n_ent))

            # Sample step time (rolling mean over last RECORD_EVERY steps)
            if step_t_acc:
                res.step_times_ms.append((step, float(np.mean(step_t_acc))))
                step_t_acc = []

        # Completion check
        if len(sim.found) >= res.total_survivors and not res.completed:
            res.completed        = True
            res.completion_step  = step

        # Progress bar update every 10 steps
        done_steps += 1
        if step % 10 == 0 or step == MAX_STEPS:
            elapsed  = time.time() - run_start
            rate     = done_steps / max(elapsed, 0.001)   # steps/s
            remaining= (total_steps - done_steps) / max(rate, 0.001)
            eta_min  = int(remaining // 60)
            eta_sec  = int(remaining % 60)
            suffix   = (f"seed {seed}/{SEEDS[-1]}  step {step}/{MAX_STEPS}  "
                        f"ETA {eta_min:02d}:{eta_sec:02d}  "
                        f"{rate:.0f} sim-steps/s")
            progress_bar(done_steps, total_steps, prefix='Progress', suffix=suffix)

    # Finalise
    res.wall_time_s   = time.time() - t0
    union = sim.union_belief
    res.final_coverage    = float(np.sum(union != M.T_UNKNOWN)) / (M.GRID_W * M.GRID_H) * 100
    res.final_found       = len(sim.found)
    res.buildings_entered, _ = count_buildings_entered(sim)

    if not res.completed:
        failed_seeds.append(seed)

    results.append(res)

    # Print per-seed summary — clear the progress bar line first
    sys.stdout.write('\r' + ' ' * 120 + '\r')  # blank out the bar line
    sys.stdout.flush()
    status = (f"✓ t={res.completion_step}" if res.completed
              else f"✗ {res.final_found}/{res.total_survivors} found")
    avg_ms = np.mean([ms for _, ms in res.step_times_ms]) if res.step_times_ms else 0
    print(f"  seed {seed:3d}  {status:20s}  cov={res.final_coverage:.1f}%  "
          f"bldgs={res.buildings_entered}/{res.buildings_total}  "
          f"deaths={len(res.deaths)}  {avg_ms:.1f} ms/step  "
          f"wall={res.wall_time_s:.0f}s")

print(f"\n{'─'*60}")
print(f"Total wall time: {time.time()-run_start:.0f}s")

# ── Save failed seeds ──────────────────────────────────────────────────────────
failed_path = os.path.join(OUT_DIR, "failed_seeds.txt")
with open(failed_path, "w") as f:
    f.write(f"Seeds that did not complete within {MAX_STEPS} steps:\n\n")
    if failed_seeds:
        for s in failed_seeds:
            res = next(r for r in results if r.seed == s)
            f.write(f"  seed {s}: {res.final_found}/{res.total_survivors} found  "
                    f"cov={res.final_coverage:.1f}%  deaths={len(res.deaths)}\n")
            # Death breakdown
            by_reason = defaultdict(int)
            for _, _, reason in res.deaths: by_reason[reason] += 1
            for reason, cnt in by_reason.items():
                f.write(f"    deaths: {cnt}x '{reason}'\n")
    else:
        f.write("  None — all seeds completed!\n")

# ── Failure diagnosis ──────────────────────────────────────────────────────────
# Re-run failed seeds briefly to diagnose stuck survivors
print(f"\nDiagnosing {len(failed_seeds)} failed seeds...")
failure_diagnosis = {}
for seed in failed_seeds:
    random.seed(seed); np.random.seed(seed)
    sim_d = FleetSim()
    # Run to MAX_STEPS but stop early if we can diagnose
    for _ in range(min(MAX_STEPS, 500)): sim_d.step()

    unfound = [s for s in sim_d.survivors if s not in sim_d.found]
    diag = {"unfound": len(unfound), "reasons": []}
    for s in unfound:
        sx, sy = s
        z = sim_d.cell_to_zone(sx, sy)
        in_shadow = bool(sim_d.radio_shadow[sx, sy])
        relay_ok  = bool(sim_d._relay_ok_flood.get(z, False))
        uf        = float(sim_d.zone_stats(z)['unknown_frac']) if z else -1
        zt        = sim_d._shadow_zone_type.get(z, 'open')
        # Check if any capable robot is still alive
        capable_alive = sum(1 for r in sim_d.robots
                            if r.active and r.caps_mask & (M.CAP_STAIRS | M.CAP_AIR))
        battery_dead  = sum(1 for r in sim_d.robots
                            if not r.active and r.death_reason == 'battery depleted')
        comms_dead    = sum(1 for r in sim_d.robots
                            if not r.active and r.death_reason and 'comms' in r.death_reason)
        if in_shadow and not relay_ok:
            reason = f"shadow zone {z} (type={zt}) — no relay coverage, uf={uf:.2f}"
        elif in_shadow and relay_ok and uf > 0.15:
            reason = f"shadow zone {z} (type={zt}) — relay ok but zone unexplored uf={uf:.2f}"
        elif capable_alive == 0:
            reason = f"no capable robots alive (battery_dead={battery_dead})"
        else:
            reason = f"zone {z} type={zt} uf={uf:.2f} shadow={in_shadow}"
        diag["reasons"].append(reason)

    diag["capable_alive"]  = capable_alive
    diag["battery_deaths"] = battery_dead
    diag["comms_deaths"]   = comms_dead
    failure_diagnosis[seed] = diag
    print(f"  seed {seed}: {len(unfound)} unfound survivors")
    for r in diag["reasons"]: print(f"    → {r}")

# Append diagnosis to failed_seeds.txt
with open(failed_path, "a") as f:
    f.write("\n\nFailure Diagnosis (at t=500):\n")
    for seed, diag in failure_diagnosis.items():
        f.write(f"\n  seed {seed}:\n")
        f.write(f"    capable robots alive: {diag['capable_alive']}\n")
        f.write(f"    battery deaths: {diag['battery_deaths']}\n")
        f.write(f"    comms deaths:   {diag['comms_deaths']}\n")
        for r in diag["reasons"]:
            f.write(f"    unfound: {r}\n")

# ── Summary stats ──────────────────────────────────────────────────────────────
completed  = [r for r in results if r.completed]
incomplete = [r for r in results if not r.completed]
comp_steps = [r.completion_step for r in completed]

avg_ms_all = [np.mean([ms for _, ms in r.step_times_ms])
              for r in results if r.step_times_ms]
complexity_ref = results[0].complexity if results else {}

summary = {
    "n_seeds":               N_SEEDS,
    "max_steps":             MAX_STEPS,
    "seeds":                 SEEDS,
    "n_completed":           len(completed),
    "n_failed":              len(incomplete),
    "completion_rate_pct":   round(100 * len(completed) / max(N_SEEDS, 1), 1),
    "failed_seeds":          failed_seeds,
    "completion_steps":      {r.seed: r.completion_step for r in completed},
    "buildings_entered":     {r.seed: {"entered": r.buildings_entered,
                                       "total":   r.buildings_total} for r in results},
    "building_coverage_pct": {
        "mean":   round(float(np.mean([r.buildings_entered/max(r.buildings_total,1)*100 for r in results])), 1),
        "min":    round(float(np.min( [r.buildings_entered/max(r.buildings_total,1)*100 for r in results])), 1),
        "max":    round(float(np.max( [r.buildings_entered/max(r.buildings_total,1)*100 for r in results])), 1),
        "all_entered_pct": round(100*sum(1 for r in results if r.buildings_entered>=r.buildings_total)/max(N_SEEDS,1), 1),
    },
    "failure_analysis": {
        seed: {
            "unfound":        diag["unfound"],
            "capable_alive":  diag["capable_alive"],
            "battery_deaths": diag["battery_deaths"],
            "comms_deaths":   diag["comms_deaths"],
            "reasons":        diag["reasons"],
        } for seed, diag in failure_diagnosis.items()
    } if failure_diagnosis else {},
    "failure_cause_summary": {
        "shadow_no_relay":    sum(1 for d in failure_diagnosis.values()
                                  for r in d["reasons"] if "no relay coverage" in r),
        "shadow_unexplored":  sum(1 for d in failure_diagnosis.values()
                                  for r in d["reasons"] if "relay ok but zone unexplored" in r),
        "open_unexplored":    sum(1 for d in failure_diagnosis.values()
                                  for r in d["reasons"] if "type=none" in r or "type=open" in r),
        "no_capable_robot":   sum(1 for d in failure_diagnosis.values()
                                  for r in d["reasons"] if "no capable" in r),
    } if failure_diagnosis else {},
    "computational_efficiency": {
        "mean_ms_per_step":  float(np.mean(avg_ms_all)) if avg_ms_all else None,
        "std_ms_per_step":   float(np.std(avg_ms_all))  if avg_ms_all else None,
        "grid_cells":        complexity_ref.get("grid_cells"),
        "n_robots":          complexity_ref.get("n_robots"),
        "n_zones":           complexity_ref.get("n_zones"),
        "O_combined":        complexity_ref.get("O_combined_per_step"),
        "dominant_term":     complexity_ref.get("dominant_term"),
        "O_astar":           complexity_ref.get("O_astar_per_step"),
        "O_cbba_amortised":  complexity_ref.get("O_cbba_amortised"),
        "O_belief":          complexity_ref.get("O_belief_per_step"),
    },
}
if comp_steps:
    summary["mean_completion"]   = float(np.mean(comp_steps))
    summary["median_completion"] = float(np.median(comp_steps))
    summary["std_completion"]    = float(np.std(comp_steps))
    summary["min_completion"]    = int(np.min(comp_steps))
    summary["max_completion"]    = int(np.max(comp_steps))

with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

# ════════════════════════════════════════════════════════════════════════════════
# GRAPHS
# ════════════════════════════════════════════════════════════════════════════════
PALETTE = plt.cm.tab10.colors

# ── 1. Survivors found over time ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 5))
for i, res in enumerate(results):
    xs = [s for s, _ in res.found_over_time]
    ys = [n / res.total_survivors * 100 for _, n in res.found_over_time]
    ax.plot(xs, ys, '-' if res.completed else '--',
            color=PALETTE[i % 10], linewidth=1.6,
            label=f"seed {res.seed}" + ("" if res.completed else " ✗"))
ax.set_xlabel("Timestep"); ax.set_ylabel("Survivors found (%)")
ax.set_title("Survivor Discovery Rate — All Seeds")
ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3)
ax.set_xlim(0, MAX_STEPS); ax.set_ylim(0, 105)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "01_survivors_over_time.png"), dpi=130)
plt.close(fig)

# ── 2. Map coverage over time ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 5))
for i, res in enumerate(results):
    xs = [s for s, _ in res.coverage_over_time]
    ys = [c for _, c in res.coverage_over_time]
    ax.plot(xs, ys, '-' if res.completed else '--',
            color=PALETTE[i % 10], linewidth=1.6,
            label=f"seed {res.seed}" + ("" if res.completed else " ✗"))
ax.set_xlabel("Timestep"); ax.set_ylabel("Coverage (%)")
ax.set_title("Map Coverage Over Time — All Seeds")
ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3)
ax.set_xlim(0, MAX_STEPS)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "02_coverage_over_time.png"), dpi=130)
plt.close(fig)

# ── 3. Completion time distribution ───────────────────────────────────────────
if completed:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(comp_steps, bins=min(10, len(comp_steps)),
                 color='steelblue', edgecolor='white', linewidth=0.8)
    axes[0].axvline(np.mean(comp_steps), color='red', linestyle='--',
                    linewidth=1.5, label=f"mean = {np.mean(comp_steps):.0f}")
    axes[0].axvline(np.median(comp_steps), color='orange', linestyle=':',
                    linewidth=1.5, label=f"median = {np.median(comp_steps):.0f}")
    axes[0].set_xlabel("Steps"); axes[0].set_ylabel("Count")
    axes[0].set_title("Completion Time Distribution")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    bar_seeds = [r.seed for r in completed]
    axes[1].bar([str(s) for s in bar_seeds], comp_steps,
                color='steelblue', edgecolor='white')
    axes[1].axhline(np.mean(comp_steps), color='red', linestyle='--', linewidth=1.2,
                    label=f"mean={np.mean(comp_steps):.0f} ± {np.std(comp_steps):.0f}")
    axes[1].set_xlabel("Seed"); axes[1].set_ylabel("Steps")
    axes[1].set_title("Completion Time per Seed")
    axes[1].legend(); axes[1].grid(alpha=0.3, axis='y')
    fig.suptitle(f"Completion Time  (n={len(completed)}/{N_SEEDS})", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "03_completion_time.png"), dpi=130)
    plt.close(fig)

# ── 4. Robot deaths by reason ─────────────────────────────────────────────────
all_deaths = [(res.seed, step, name, reason)
              for res in results for step, name, reason in res.deaths]
reason_counts = defaultdict(int)
for _, _, _, reason in all_deaths:
    reason_counts[reason] += 1

if reason_counts:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    labels = list(reason_counts.keys())
    sizes  = [reason_counts[l] for l in labels]
    axes[0].pie(sizes, labels=labels, autopct='%1.0f%%',
                colors=PALETTE[:len(labels)], startangle=90,
                wedgeprops=dict(edgecolor='white'))
    axes[0].set_title(f"Death Reasons (total={sum(sizes)})")

    seed_reason = defaultdict(lambda: defaultdict(int))
    for seed, _, _, reason in all_deaths:
        seed_reason[seed][reason] += 1
    all_reasons  = sorted(reason_counts.keys())
    seed_labels  = [str(r.seed) for r in results]
    bottoms      = np.zeros(len(results))
    for j, reason in enumerate(all_reasons):
        vals = [seed_reason[r.seed][reason] for r in results]
        axes[1].bar(seed_labels, vals, bottom=bottoms,
                    label=reason, color=PALETTE[j % 10], edgecolor='white')
        bottoms += np.array(vals)
    axes[1].set_xlabel("Seed"); axes[1].set_ylabel("Deaths")
    axes[1].set_title("Deaths by Reason per Seed")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3, axis='y')
    fig.suptitle("Robot Deaths", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "04_robot_deaths.png"), dpi=130)
    plt.close(fig)

# ── 5. Cumulative deaths over time ────────────────────────────────────────────
if all_deaths:
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, res in enumerate(results):
        if not res.deaths: continue
        steps_d = sorted([s for s, _, _ in res.deaths])
        ax.step(steps_d, range(1, len(steps_d)+1), where='post',
                color=PALETTE[i % 10], linewidth=1.6, label=f"seed {res.seed}")
    ax.set_xlabel("Timestep"); ax.set_ylabel("Cumulative deaths")
    ax.set_title("Cumulative Robot Deaths Over Time")
    ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3)
    ax.set_xlim(0, MAX_STEPS)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "05_deaths_over_time.png"), dpi=130)
    plt.close(fig)

# ── 6. Role distribution over time ───────────────────────────────────────────
role_names   = ['SCOUT', 'SCAN', 'LOITER', 'RELAY']
role_colors  = {'SCOUT': 'steelblue', 'SCAN': 'seagreen',
                'LOITER': 'goldenrod', 'RELAY': 'tomato'}
ref_steps    = [s for s, _ in results[0].found_over_time]

fig, ax = plt.subplots(figsize=(11, 5))
for role in role_names:
    per_seed = [[d.get(role, 0) for _, d in res.robot_roles_over_time]
                for res in results]
    min_len  = min(len(v) for v in per_seed)
    arr      = np.array([v[:min_len] for v in per_seed], dtype=float)
    mean     = arr.mean(axis=0)
    std      = arr.std(axis=0)
    xs       = ref_steps[:min_len]
    ax.plot(xs, mean, color=role_colors[role], linewidth=2, label=role)
    ax.fill_between(xs, mean-std, mean+std, alpha=0.15, color=role_colors[role])
ax.set_xlabel("Timestep"); ax.set_ylabel("Active robots (mean ± std)")
ax.set_title("Role Distribution Over Time (mean across seeds)")
ax.legend(); ax.grid(alpha=0.3)
ax.set_xlim(0, MAX_STEPS)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "06_role_distribution.png"), dpi=130)
plt.close(fig)

# ── 7. Buildings entered over time ────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: buildings entered over time per seed
for i, res in enumerate(results):
    if not res.buildings_coverage: continue
    xs = [s for s, _ in res.buildings_coverage]
    ys = [n / max(res.buildings_total, 1) * 100
          for _, n in res.buildings_coverage]
    axes[0].plot(xs, ys, '-' if res.completed else '--',
                 color=PALETTE[i % 10], linewidth=1.6, label=f"seed {res.seed}")
axes[0].set_xlabel("Timestep")
axes[0].set_ylabel("Buildings entered (%)")
axes[0].set_title("Buildings Entered Over Time")
axes[0].legend(fontsize=8, ncol=2); axes[0].grid(alpha=0.3)
axes[0].set_xlim(0, MAX_STEPS); axes[0].set_ylim(0, 105)

# Right: final buildings entered per seed bar chart
seed_labels = [str(r.seed) for r in results]
entered_pct = [r.buildings_entered / max(r.buildings_total, 1) * 100
               for r in results]
bar_c = ['steelblue' if r.completed else 'tomato' for r in results]
axes[1].bar(seed_labels, entered_pct, color=bar_c, edgecolor='white')
axes[1].axhline(np.mean(entered_pct), color='red', linestyle='--', linewidth=1.2,
                label=f"mean={np.mean(entered_pct):.0f}%")
for i, (label, val, res) in enumerate(zip(seed_labels, entered_pct, results)):
    axes[1].text(i, val + 1.5,
                 f"{res.buildings_entered}/{res.buildings_total}",
                 ha='center', va='bottom', fontsize=8)
axes[1].set_xlabel("Seed"); axes[1].set_ylabel("Buildings entered (%)")
axes[1].set_title("Final Buildings Entered per Seed")
axes[1].legend(); axes[1].grid(alpha=0.3, axis='y')
axes[1].set_ylim(0, 115)
fig.suptitle("Building Exploration", fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "07_buildings_entered.png"), dpi=130)
plt.close(fig)

# ── 8. Computational efficiency ───────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: ms/step over time for each seed
for i, res in enumerate(results):
    if not res.step_times_ms: continue
    xs = [s for s, _ in res.step_times_ms]
    ys = [ms for _, ms in res.step_times_ms]
    axes[0].plot(xs, ys, color=PALETTE[i % 10], linewidth=1.4,
                 alpha=0.8, label=f"seed {res.seed}")
# Mean line
if results[0].step_times_ms:
    min_len = min(len(r.step_times_ms) for r in results if r.step_times_ms)
    all_ms  = np.array([[ms for _, ms in r.step_times_ms[:min_len]]
                         for r in results if r.step_times_ms])
    xs_ref  = [s for s, _ in results[0].step_times_ms[:min_len]]
    axes[0].plot(xs_ref, all_ms.mean(axis=0),
                 color='black', linewidth=2.5, linestyle='--', label='mean', zorder=5)
axes[0].set_xlabel("Timestep"); axes[0].set_ylabel("Wall time (ms/step)")
axes[0].set_title("Computational Cost per Step")
axes[0].legend(fontsize=7, ncol=2); axes[0].grid(alpha=0.3)
axes[0].set_xlim(0, MAX_STEPS)

# Right: complexity table as text
axes[1].axis('off')
cx = complexity_ref
lines = [
    "Theoretical Complexity  (per timestep)",
    "─" * 44,
    f"Grid:          {cx.get('grid_W')} × {cx.get('grid_H')} = {cx.get('grid_cells'):,} cells",
    f"Robots:        R = {cx.get('n_robots')}",
    f"Zones:         Z = {cx.get('n_zones')}",
    f"Stair cells:   {cx.get('stair_cells'):,}",
    f"Shadow cells:  {cx.get('shadow_cells'):,}",
    "",
    "Per-step costs (amortised):",
    f"  A* (replanning):",
    f"    O(R · WH · log WH)",
    f"    ≈ {cx.get('dominant_term', '')}",
    "",
    f"  CBBA (every 50 steps):",
    f"    O(R² · Z / 50)",
    f"    = {cx.get('O_cbba_amortised', '')}",
    "",
    f"  Belief union:",
    f"    {cx.get('O_belief_per_step', '')}",
    "",
    "Overall (dominant):",
    f"  O(R · WH · log WH)",
]
if avg_ms_all:
    lines += [
        "",
        f"Measured: {np.mean(avg_ms_all):.1f} ± {np.std(avg_ms_all):.1f} ms/step",
        f"  ≈ {1000/max(np.mean(avg_ms_all),0.001):.0f} sim-steps/sec",
    ]
axes[1].text(0.03, 0.97, "\n".join(lines), transform=axes[1].transAxes,
             fontsize=8.5, va='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#f0f4ff', alpha=0.9))
axes[1].set_title("Complexity Analysis")
fig.suptitle("Computational Efficiency", fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "08_computational_efficiency.png"), dpi=130)
plt.close(fig)

# ── 9. Summary dashboard ──────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 9))
gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.38)

# Final survivors
ax1 = fig.add_subplot(gs[0, 0])
found_pct = [r.final_found / r.total_survivors * 100 for r in results]
bar_c     = ['steelblue' if r.completed else 'tomato' for r in results]
ax1.bar([str(r.seed) for r in results], found_pct, color=bar_c, edgecolor='white')
ax1.axhline(100, color='green', linestyle='--', linewidth=1, alpha=0.6)
ax1.set_ylim(0, 110); ax1.set_ylabel("% survivors found")
ax1.set_title("Survivors Found"); ax1.grid(alpha=0.3, axis='y')

# Final coverage
ax2 = fig.add_subplot(gs[0, 1])
cov_vals = [r.final_coverage for r in results]
ax2.bar([str(r.seed) for r in results], cov_vals,
        color='mediumslateblue', edgecolor='white')
ax2.set_ylabel("Coverage (%)"); ax2.set_title("Final Map Coverage")
ax2.grid(alpha=0.3, axis='y')

# Buildings entered
ax3 = fig.add_subplot(gs[0, 2])
ent_pct = [r.buildings_entered / max(r.buildings_total, 1) * 100 for r in results]
ax3.bar([str(r.seed) for r in results], ent_pct, color='darkorange', edgecolor='white')
ax3.axhline(np.mean(ent_pct), color='red', linestyle='--', linewidth=1.2)
for i, (val, res) in enumerate(zip(ent_pct, results)):
    ax3.text(i, val+1.5, f"{res.buildings_entered}/{res.buildings_total}",
             ha='center', fontsize=7)
ax3.set_ylim(0, 115); ax3.set_ylabel("% buildings entered")
ax3.set_title("Buildings Entered"); ax3.grid(alpha=0.3, axis='y')

# Deaths
ax4 = fig.add_subplot(gs[0, 3])
death_counts = [len(r.deaths) for r in results]
ax4.bar([str(r.seed) for r in results], death_counts, color='salmon', edgecolor='white')
ax4.set_ylabel("Deaths"); ax4.set_title("Robot Deaths per Seed")
ax4.grid(alpha=0.3, axis='y')

# Completion time box
ax5 = fig.add_subplot(gs[1, 0])
if comp_steps:
    ax5.boxplot(comp_steps, vert=True, patch_artist=True,
                boxprops=dict(facecolor='lightsteelblue', color='steelblue'),
                medianprops=dict(color='red', linewidth=2))
    ax5.scatter([1]*len(comp_steps), comp_steps, color='steelblue', alpha=0.7, zorder=3, s=40)
    ax5.set_xticks([1]); ax5.set_xticklabels([f"n={len(comp_steps)}"])
    ax5.set_ylabel("Steps"); ax5.set_title("Completion Time")
    ax5.grid(alpha=0.3, axis='y')
    ax5.text(1.35, np.median(comp_steps),
             f"mean={np.mean(comp_steps):.0f}\nstd={np.std(comp_steps):.0f}\n"
             f"min={np.min(comp_steps)}\nmax={np.max(comp_steps)}",
             fontsize=8, va='center')
else:
    ax5.text(0.5, 0.5, "No completions", ha='center', va='center',
             transform=ax5.transAxes, fontsize=12, color='gray')
    ax5.set_title("Completion Time")

# Coverage vs survivors scatter
ax6 = fig.add_subplot(gs[1, 1])
ax6.scatter(cov_vals, found_pct,
            c=['steelblue' if r.completed else 'tomato' for r in results],
            s=80, zorder=3, edgecolors='white', linewidths=0.8)
for r in results:
    ax6.annotate(str(r.seed), (r.final_coverage, r.final_found/r.total_survivors*100),
                 fontsize=7, xytext=(3, 3), textcoords='offset points')
ax6.set_xlabel("Final Coverage (%)"); ax6.set_ylabel("Survivors found (%)")
ax6.set_title("Coverage vs Survivors"); ax6.grid(alpha=0.3)

# ms/step over time (mean + std band)
ax7 = fig.add_subplot(gs[1, 2])
if results[0].step_times_ms:
    min_len2 = min(len(r.step_times_ms) for r in results if r.step_times_ms)
    all_ms2  = np.array([[ms for _, ms in r.step_times_ms[:min_len2]]
                          for r in results if r.step_times_ms])
    xs2      = [s for s, _ in results[0].step_times_ms[:min_len2]]
    m2       = all_ms2.mean(axis=0); s2 = all_ms2.std(axis=0)
    ax7.plot(xs2, m2, color='darkblue', linewidth=2)
    ax7.fill_between(xs2, m2-s2, m2+s2, alpha=0.2, color='blue')
    ax7.set_xlabel("Timestep"); ax7.set_ylabel("ms / step")
    ax7.set_title(f"Step Time (mean={m2.mean():.1f} ms)")
    ax7.grid(alpha=0.3); ax7.set_xlim(0, MAX_STEPS)

# Text summary
ax8 = fig.add_subplot(gs[1, 3])
ax8.axis('off')
lines = [
    f"Seeds run:    {N_SEEDS}",
    f"Completed:    {len(completed)}  ({len(completed)/N_SEEDS*100:.0f}%)",
    f"Failed:       {len(incomplete)}",
    f"",
]
if comp_steps:
    lines += [
        f"Completion:",
        f"  mean:  {np.mean(comp_steps):.0f} steps",
        f"  std:   {np.std(comp_steps):.0f}",
        f"  range: {np.min(comp_steps)}–{np.max(comp_steps)}",
        f"",
    ]
lines += [
    f"Buildings:",
    f"  mean entered: {np.mean(ent_pct):.0f}%",
    f"",
    f"Compute:",
]
if avg_ms_all:
    lines += [
        f"  {np.mean(avg_ms_all):.1f} ± {np.std(avg_ms_all):.1f} ms/step",
        f"  O(R · WH · log WH)",
    ]
lines += [f"", f"Failed: {failed_seeds if failed_seeds else 'none'}"]
ax8.text(0.05, 0.95, "\n".join(lines), transform=ax8.transAxes,
         fontsize=9, va='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
ax8.set_title("Run Summary")

fig.suptitle(f"Fleet Sim Benchmark — {N_SEEDS} Seeds, max {MAX_STEPS} steps",
             fontsize=14, fontweight='bold')
fig.savefig(os.path.join(OUT_DIR, "09_summary_dashboard.png"), dpi=140)
plt.close(fig)

# ── Done ──────────────────────────────────────────────────────────────────────
print(f"\nAll outputs saved to: {OUT_DIR}/")
print("  01_survivors_over_time.png")
print("  02_coverage_over_time.png")
print("  03_completion_time.png")
print("  04_robot_deaths.png")
print("  05_deaths_over_time.png")
print("  06_role_distribution.png")
print("  07_buildings_entered.png      ← NEW")
print("  08_computational_efficiency.png ← NEW")
print("  09_summary_dashboard.png")
print("  summary.json")
print("  failed_seeds.txt")

if comp_steps:
    print(f"\nMean completion: {np.mean(comp_steps):.0f} ± {np.std(comp_steps):.0f} steps")
print(f"Failed seeds: {failed_seeds if failed_seeds else 'none'}")
if avg_ms_all:
    print(f"Compute: {np.mean(avg_ms_all):.1f} ms/step  →  O(R · WH · log WH)")

# ── Hot-seed profiler ─────────────────────────────────────────────────────────
# Identifies the slowest functions on the two most expensive seeds.
# Writes a human-readable report to benchmark_results_J/profile_seed_N.txt
# Run once after normal benchmarking — does NOT re-run all seeds.
import cProfile, pstats, io

PROFILE_SEEDS  = [3, 4]     # seeds to profile — change if your slow seeds differ
PROFILE_STEPS  = 300        # steps to profile (enough for all code paths to run)
PROFILE_TOP_N  = 25         # functions to show in the report

print(f"\nProfiling seeds {PROFILE_SEEDS} for {PROFILE_STEPS} steps each...")
for pseed in PROFILE_SEEDS:
    random.seed(pseed); np.random.seed(pseed)
    sim_p = FleetSim()

    pr = cProfile.Profile()
    pr.enable()
    for _ in range(PROFILE_STEPS):
        sim_p.step()
    pr.disable()

    # Sort by cumulative time — shows which functions own the most wall time
    sio = io.StringIO()
    ps  = pstats.Stats(pr, stream=sio).sort_stats('cumulative')
    ps.print_stats(PROFILE_TOP_N)
    report = sio.getvalue()

    out_path = os.path.join(OUT_DIR, f"profile_seed_{pseed}.txt")
    with open(out_path, "w") as f:
        f.write(f"cProfile — seed {pseed}, {PROFILE_STEPS} steps\n")
        f.write("Sorted by cumulative time. Top functions most likely to optimise.\n\n")
        f.write(report)

    # Print the top 10 lines to console for quick inspection
    lines = [l for l in report.splitlines() if l.strip()]
    header_done = False
    shown = 0
    print(f"\n  seed {pseed} — top hotspots:")
    for line in lines:
        if 'cumtime' in line:
            header_done = True
        if header_done and shown < 10:
            print(f"    {line}")
            shown += 1
    print(f"  Full report: {out_path}")

print("\nProfiling complete.")