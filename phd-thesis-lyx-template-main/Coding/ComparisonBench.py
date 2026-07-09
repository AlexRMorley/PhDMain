"""
benchmark.py  --  Five-way comparison benchmark
================================================
Compares: Fleet | GNF | Greedy RL | CARA-Base | CARA-Dynamic

Usage:
    python benchmark.py [--steps N] [--seeds 0,1,2] [--out ./results]

Outputs
-------
  - Terminal table (per-seed + averaged summary)
  - 09 PNG plots saved to --out directory
  - summary.json
"""

import sys, os, random, time, argparse, json
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
    """Return path of first file found matching any name in any search dir."""
    for name in names:
        for folder in _SEARCH_DIRS:
            p = os.path.join(folder, name)
            if os.path.exists(p):
                return os.path.abspath(p)
    print("\nERROR: Cannot find any of:", list(names))
    print("Searched folders:")
    for d in _SEARCH_DIRS:
        print("  " + d)
    print("\nPlace all sim files in the same folder as benchmark.py and retry.")
    sys.exit(1)

def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M); return M

# Canonical name first, then any local aliases you may be using
M      = _load(_find('hetero_robot_fleet_sim.py', '2DFleetFrameworkK.py'), 'fleet')
M_gnf  = _load(_find('gnf_sim.py', 'GNF_Sim.py', 'GNF_sim.py'),           'gnf')
M_rl   = _load(_find('greedy_rl_sim.py', 'greedyRL.py', 'GreedyRL.py'),    'rl')
M_cara = _load(_find('Cara_sim.py'),                                        'cara')

FleetSim    = M.FleetSim
GNFSim      = M_gnf.GNFSim
GreedyRLSim = M_rl.make_greedy_rl_sim(M_gnf, M)

# use_exec_layer was added in the updated cara_sim.py.
# Falls back gracefully if you have an older local version.
try:
    CARABase    = M_cara.make_cara_sim(M_gnf, M, use_exec_layer=False)
    CARADynamic = M_cara.make_cara_sim(M_gnf, M, use_exec_layer=True)
except TypeError:
    print("  [cara_sim.py: use_exec_layer not supported -- update for Base vs Dynamic]")
    CARABase    = M_cara.make_cara_sim(M_gnf, M)
    CARADynamic = M_cara.make_cara_sim(M_gnf, M)

# Sim registry: (name, factory, colour, linestyle)
SIMS = [
    ('Fleet',         lambda: FleetSim(),      '#2166ac', '-'),
    ('GNF',           lambda: GNFSim(M),       '#d6604d', '--'),
    ('GreedyRL',      lambda: GreedyRLSim(M),  '#4dac26', ':'),
    ('CARA-Base',     lambda: CARABase(M),      '#984ea3', '-.'),
    ('CARA-Dynamic',  lambda: CARADynamic(M),   '#ff7f00', (0,(3,1,1,1))),
]
RECORD_EVERY = 25
BAR_WIDTH    = 32   # characters in the progress bar

def _progress(done, total, prefix='', suffix=''):
    """Overwrite the current terminal line with a progress bar."""
    filled = int(BAR_WIDTH * done / max(total, 1))
    bar    = '#' * filled + '-' * (BAR_WIDTH - filled)
    pct    = 100.0 * done / max(total, 1)
    sys.stdout.write(f'\r  {prefix}[{bar}] {pct:5.1f}%  {suffix:<35s}')
    sys.stdout.flush()

def _clear():
    sys.stdout.write('\r' + ' ' * 90 + '\r')
    sys.stdout.flush()

# ── Helpers ────────────────────────────────────────────────────────────────────
def _stair_mask(sim):
    if hasattr(sim, '_world_stair_arr'): return sim._world_stair_arr
    sm = np.zeros((M.GRID_W, M.GRID_H), dtype=bool)
    for x in range(M.GRID_W):
        for y in range(M.GRID_H):
            if sim.world.grid[x][y]['t'] == M.T_STAIRS: sm[x,y] = True
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
    for i in range(1, n+1):
        bld = (lab == i)
        if np.any(sim.union_belief[bld] != M.T_UNKNOWN): entered += 1
    return entered, n

def _redundancy(sim):
    """
    Cell-level redundancy: fraction of active robots sharing their exact cell
    with at least one other robot.  Fires rarely — mainly shows collision events.
    """
    active = [r for r in sim.robots if r.active]
    if not active: return 0.0
    pos_counts = defaultdict(int)
    for r in active: pos_counts[r.pos] += 1
    return sum(1 for r in active if pos_counts[r.pos] > 1) / len(active)

def _zone_redundancy(sim):
    """
    Zone-level redundancy: fraction of active robots whose assigned zone is
    also occupied by at least one other active robot.

    This is the meaningful coordination metric — it shows whether the allocation
    system (CBBA / MILP / greedy) successfully spreads robots across zones or
    lets them cluster.  Fleet's CBBA explicitly prevents this via the bundle
    mechanism; GNF and CARA-Base have no such guarantee.

    For sims without explicit zone assignment (GNF, GreedyRL) we use the
    robot's current zone position as a proxy.
    """
    active = [r for r in sim.robots if r.active]
    if not active: return 0.0
    zone_counts = defaultdict(int)
    for r in active:
        # Use task_zone if assigned, otherwise current cell's zone
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
        # time-series
        self.found_ts = []; self.cov_ts = []; self.stair_ts = []
        self.bldg_ts = []; self.redundancy_ts = []; self.zone_redundancy_ts = []; self.step_ms_ts = []
        self.flipflop_ts = []; self.idle_move_ts = []
        self.milp_ms_ts = []   # CARA only: (step, milp_ms) at each MILP solve
        # final scalars
        self.final_found = 0; self.final_cov = 0.0; self.final_stair = 0.0
        self.deaths = 0; self.hazard_deaths = 0; self.battery_deaths = 0
        self.viol = 0; self.trapped = 0
        self.p50 = 0.0; self.p90 = 0.0; self.avg_ms = 0.0
        self.milp_avg = 0.0; self.milp_max = 0.0; self.milp_n = 0
        self.hold = 0; self.ejects = 0
        self.wall_s = 0.0

# ── Runner ─────────────────────────────────────────────────────────────────────
def run_sim(sim_name, factory, seed, steps):
    random.seed(seed); np.random.seed(seed)
    sim = factory()
    res = Result(sim_name, seed)
    res.total_survivors = len(sim.survivors)
    _, res.buildings_total = _buildings_entered(sim)

    step_acc = []; trapped = 0; prev_milp_n = 0
    t_wall = time.time()
    # Per-robot position history for flip-flop detection (last 5 positions)
    from collections import deque as _deque
    _pos_hist = {r.name: _deque(maxlen=5) for r in sim.robots}
    _prev_union_count = int(np.sum(sim.union_belief != M.T_UNKNOWN))
    _prev_pos = {r.name: r.pos for r in sim.robots}

    for step in range(1, steps + 1):
        t0 = time.perf_counter()
        sim.step()
        ms = (time.perf_counter() - t0) * 1000
        step_acc.append(ms)

        # Flip-flop: robot returned to a recently visited position
        active_robots = [r for r in sim.robots if r.active]
        flip_count = 0
        for r in active_robots:
            hist = _pos_hist[r.name]
            if r.pos in hist:
                flip_count += 1
            hist.append(r.pos)
        flip_pct = flip_count / max(len(active_robots), 1) * 100

        # Idle movement: robot moved but fleet gained no new cells that step
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

        # CARA: capture each new MILP solve time
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
            step_acc = []

        if len(sim.found) >= res.total_survivors and not res.completed:
            res.completed = True; res.completion_step = step

        # Progress bar — update every 10 steps to avoid slowing the sim
        if step % 10 == 0 or step == steps:
            elapsed = time.time() - t_wall
            eta     = (steps - step) / max(step, 1) * elapsed
            _progress(step, steps,
                      prefix=f'{sim_name:<14} ',
                      suffix=f'step {step}/{steps}  ETA {int(eta//60):02d}:{int(eta%60):02d}')

    _clear()   # wipe progress bar before printing results
    res.wall_s = time.time() - t_wall
    all_ms = [ms for _, ms in res.step_ms_ts]
    st = sorted([ms for _, ms in res.step_ms_ts]) if all_ms else [0]

    res.final_found = len(sim.found)
    res.final_cov   = float(np.mean(sim.union_belief != M.T_UNKNOWN)) * 100
    res.final_stair = _stair_cov(sim)
    res.deaths         = sum(1 for r in sim.robots if not r.active)
    res.hazard_deaths  = sum(1 for r in sim.robots if not r.active and getattr(r, 'hazard_killed', False))
    res.battery_deaths = res.deaths - res.hazard_deaths
    res.viol    = sum(1 for r in sim.robots if r.active
                      and sim.radio_shadow[r.pos[0], r.pos[1]]
                      and not _relay_covered(sim, r))
    res.trapped = trapped
    n = len(st)
    res.p50 = st[n//2]; res.p90 = st[int(n*0.9)]; res.avg_ms = sum(st)/n

    if hasattr(sim, 'milp_solve_times') and sim.milp_solve_times:
        mt = sim.milp_solve_times
        res.milp_avg = sum(mt)/len(mt); res.milp_max = max(mt); res.milp_n = len(mt)
    if hasattr(sim, 'hold_ticks'):  res.hold   = sum(sim.hold_ticks.values())
    if hasattr(sim, 'eject_events'): res.ejects = len(sim.eject_events)
    return res

# ── Plotting helpers ───────────────────────────────────────────────────────────
ALPHA_BAND = 0.15

def _mean_std(results_by_name, name, getter):
    series = [getter(r) for r in results_by_name[name]]
    if not series or not series[0]: return np.array([]), np.array([]), np.array([])
    xs  = np.array([s for s, _ in series[0]])
    arr = np.array([[v for _, v in s] for s in series], dtype=float)
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
    ap.add_argument('--steps', type=int, default=2000)
    ap.add_argument('--seeds', type=str, default='0,1,2,3,4,5,6,7,8,9,10')
    ap.add_argument('--out',   type=str, default=None,
                    help='Output directory for plots (default: ./benchmark_results/)')
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(',')]
    steps = args.steps
    out_dir = args.out or os.path.join(_HERE, 'benchmark_results_mini')
    os.makedirs(out_dir, exist_ok=True)

    # results_by_name[sim_name] = [Result, Result, ...]
    results_by_name = {name: [] for name, _, _, _ in SIMS}
    # accs for summary table
    accs = {name: defaultdict(float) for name, _, _, _ in SIMS}

    W = 128
    print(f"\n{'='*W}")
    print(f"  Five-Way Benchmark: Fleet | GNF | GreedyRL | CARA-Base | CARA-Dynamic")
    print(f"  {steps} steps  |  seeds: {seeds}  |  plots -> {out_dir}")
    print(f"{'='*W}")

    for seed in seeds:
        print(f"\n  seed={seed}:")
        print(f"  {'─'*124}")
        total_runs = len(seeds) * len(SIMS)
        for run_idx, (name, factory, _, _) in enumerate(SIMS):
            overall = (seeds.index(seed) * len(SIMS) + run_idx)
            _progress(overall, total_runs,
                      prefix='Overall  ',
                      suffix=f'seed={seed}  starting {name}...')
            res = run_sim(name, factory, seed, steps)
            results_by_name[name].append(res)
            is_cara = res.milp_n > 0
            base = (f"  {name:<14}"
                    f"  cov={res.final_cov:5.1f}%"
                    f"  stair={res.final_stair:5.1f}%"
                    f"  found={res.final_found}/{res.total_survivors}"
                    f"  deaths={res.deaths}(haz={res.hazard_deaths}/bat={res.battery_deaths})"
                    f"  trapped={res.trapped}"
                    f"  P50={res.p50:5.1f}ms  P90={res.p90:6.1f}ms  avg={res.avg_ms:5.1f}ms")
            if is_cara:
                base += (f"  | milp_avg={res.milp_avg:.0f}ms"
                         f"  milp_max={res.milp_max:.0f}ms"
                         f"  n={res.milp_n}"
                         f"  hold={res.hold}  ejects={res.ejects}")
            print(base)
            for k in ('final_cov','final_stair','final_found','deaths','hazard_deaths','battery_deaths','trapped',
                      'p50','p90','avg_ms','milp_avg','milp_max','milp_n','hold','ejects'):
                accs[name][k] += getattr(res, k, 0)

        # Delta rows
        print(f"  {'─'*124}")
        fm = results_by_name['Fleet'][-1]
        for name, _, _, _ in SIMS[1:]:
            om = results_by_name[name][-1]
            dc = fm.final_cov   - om.final_cov
            ds = fm.final_stair - om.final_stair
            df = fm.final_found - om.final_found
            dt = om.avg_ms      - fm.avg_ms
            print(f"  Fleet vs {name:<14}"
                  f"  cov={dc:+.1f}%  stair={ds:+.1f}%  found={df:+d}"
                  f"  speed={dt:+.1f}ms/step (+ = Fleet faster)")

    # ── Averaged summary ───────────────────────────────────────────────────────
    n_seeds = len(seeds)
    avgs = {name: {k: v/n_seeds for k,v in accs[name].items()}
            for name in accs}
    avgs = {name: {**d, 'total': 18} for name, d in avgs.items()}

    if n_seeds > 1:
        print(f"\n{'='*W}")
        print(f"  Averages over {n_seeds} seeds:")
        print(f"  {'─'*124}")
        for name, _, _, _ in SIMS:
            d = avgs[name]; is_cara = d['milp_n'] > 0
            base = (f"  {name:<14}"
                    f"  cov={d['final_cov']:5.1f}%"
                    f"  stair={d['final_stair']:5.1f}%"
                    f"  found={d['final_found']:.1f}/18"
                    f"  deaths={d['deaths']:.1f}(haz={d['hazard_deaths']:.1f}/bat={d['battery_deaths']:.1f})"
                    f"  trapped={d['trapped']:.0f}"
                    f"  avg={d['avg_ms']:5.1f}ms")
            if is_cara:
                base += (f"  | milp_avg={d['milp_avg']:.0f}ms"
                         f"  hold={d['hold']:.0f}  ejects={d['ejects']:.0f}")
            print(base)

    # ── Summary table ──────────────────────────────────────────────────────────
    print(f"\n{'='*W}")
    print(f"  Summary  ({steps} steps, {n_seeds} seed{'s' if n_seeds>1 else ''})")
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
    print(f"{'='*W}\n")

    # ── Save JSON ──────────────────────────────────────────────────────────────
    summary = {name: {k: round(float(v), 2) for k,v in avgs[name].items()}
               for name in avgs}
    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # ══════════════════════════════════════════════════════════════════════════
    # Plots
    # ══════════════════════════════════════════════════════════════════════════
    print("Generating plots...", end=' ', flush=True)
    saved = []

    # 01 — Survivor discovery
    fig, ax = plt.subplots(figsize=(11,5))
    _plot_metric(ax, results_by_name,
                 lambda r: [(s, v/r.total_survivors*100) for s,v in r.found_ts],
                 'Survivors found (%)', 'Survivor Discovery Rate', steps, ylim=(0,105))
    saved.append(_save(fig, out_dir, '01_survivors.png'))

    # 02 — Map coverage
    fig, ax = plt.subplots(figsize=(11,5))
    _plot_metric(ax, results_by_name, lambda r: r.cov_ts,
                 'Coverage (%)', 'Map Coverage Over Time', steps)
    saved.append(_save(fig, out_dir, '02_coverage.png'))

    # 03 — Stair / building coverage
    fig, ax = plt.subplots(figsize=(11,5))
    _plot_metric(ax, results_by_name, lambda r: r.stair_ts,
                 'Stair coverage (%)', 'Building (Stair) Coverage Over Time', steps, ylim=(0,105))
    saved.append(_save(fig, out_dir, '03_stair_coverage.png'))

    # 04 — Buildings entered
    fig, ax = plt.subplots(figsize=(11,5))
    _plot_metric(ax, results_by_name,
                 lambda r: [(s, v/max(r.buildings_total,1)*100) for s,v in r.bldg_ts],
                 'Buildings entered (%)', 'Building Exploration Over Time', steps, ylim=(0,105))
    saved.append(_save(fig, out_dir, '04_buildings.png'))

    # 05 — Completion time
    fig, axes = plt.subplots(1, 2, figsize=(13,5))
    for name, _, col, _ in SIMS:
        comp = [r.completion_step for r in results_by_name[name] if r.completed]
        fail = sum(1 for r in results_by_name[name] if not r.completed)
        lbl  = f"{name} ({len(comp)}/{n_seeds}" + (f", {fail} fail)" if fail else ")")
        if comp: axes[0].hist(comp, bins=max(1,min(8,len(comp))), alpha=0.5,
                               color=col, label=lbl, edgecolor='white')
    axes[0].set_xlabel('Steps to completion'); axes[0].set_ylabel('Count')
    axes[0].set_title('Completion Time Distribution')
    axes[0].legend(fontsize=7); axes[0].grid(alpha=0.3)
    x = np.arange(n_seeds); w = 0.15
    for i, (name, _, col, _) in enumerate(SIMS):
        vals = [r.completion_step if r.completed else steps
                for r in sorted(results_by_name[name], key=lambda r: r.seed)]
        offset = (i - len(SIMS)//2) * w
        axes[1].bar(x + offset, vals, w, color=col, label=name, alpha=0.8)
    axes[1].axhline(steps, color='grey', lw=1, ls=':', label='timeout')
    axes[1].set_xticks(x); axes[1].set_xticklabels([str(s) for s in seeds])
    axes[1].set_xlabel('Seed'); axes[1].set_ylabel('Steps')
    axes[1].set_title('Completion Time per Seed')
    axes[1].legend(fontsize=7); axes[1].grid(alpha=0.3, axis='y')
    fig.suptitle('Completion Time', fontsize=13)
    saved.append(_save(fig, out_dir, '05_completion_time.png'))

    # 06 — Robot deaths
    fig, axes = plt.subplots(1, 2, figsize=(13,5))
    x = np.arange(n_seeds); w = 0.15
    for i, (name, _, col, _) in enumerate(SIMS):
        vals = [r.deaths for r in sorted(results_by_name[name], key=lambda r: r.seed)]
        offset = (i - len(SIMS)//2) * w
        axes[0].bar(x + offset, vals, w, color=col, label=name, alpha=0.8)
    axes[0].set_xticks(x); axes[0].set_xticklabels([str(s) for s in seeds])
    axes[0].set_xlabel('Seed'); axes[0].set_ylabel('Deaths')
    axes[0].set_title('Robot Deaths per Seed')
    axes[0].legend(fontsize=7); axes[0].grid(alpha=0.3, axis='y')
    # Trapped ticks bar
    for i, (name, _, col, _) in enumerate(SIMS):
        vals = [r.trapped for r in sorted(results_by_name[name], key=lambda r: r.seed)]
        offset = (i - len(SIMS)//2) * w
        axes[1].bar(x + offset, vals, w, color=col, label=name, alpha=0.8)
    axes[1].set_xticks(x); axes[1].set_xticklabels([str(s) for s in seeds])
    axes[1].set_xlabel('Seed'); axes[1].set_ylabel('Robot-ticks in uncovered shadow')
    axes[1].set_title('Shadow Entrapment (Trapped Ticks) per Seed')
    axes[1].legend(fontsize=7); axes[1].grid(alpha=0.3, axis='y')
    fig.suptitle('Safety Metrics', fontsize=13)
    saved.append(_save(fig, out_dir, '06_safety.png'))

    # 07 — Compute cost
    fig, ax = plt.subplots(figsize=(11,5))
    _plot_metric(ax, results_by_name, lambda r: r.step_ms_ts,
                 'ms / step', 'Computational Cost per Step', steps)
    # Add avg annotation
    txt = '\n'.join(f"{name}: {avgs[name]['avg_ms']:.1f} ms/step"
                    for name, _, _, _ in SIMS)
    ax.text(0.98, 0.95, txt, transform=ax.transAxes, ha='right', va='top',
            fontsize=8, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    saved.append(_save(fig, out_dir, '07_compute.png'))

    # 08 — Redundancy (cell-level and zone-level side by side)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    _plot_metric(axes[0], results_by_name, lambda r: r.zone_redundancy_ts,
                 'Robots in shared zone (%)',
                 'Zone-Level Redundancy\n(multiple robots assigned same zone)',
                 steps, ylim=(0, None))
    axes[0].annotate(
        'Lower = better coordination\nFleet CBBA minimises this',
        xy=(0.97, 0.97), xycoords='axes fraction', ha='right', va='top',
        fontsize=8, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    _plot_metric(axes[1], results_by_name, lambda r: r.redundancy_ts,
                 'Robots sharing exact cell (%)',
                 'Cell-Level Redundancy\n(robots on same grid cell)',
                 steps, ylim=(0, None))
    axes[1].annotate(
        'Fires rarely — mainly\ncollision / waiting events',
        xy=(0.97, 0.97), xycoords='axes fraction', ha='right', va='top',
        fontsize=8, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    fig.suptitle('Coordination Redundancy Metrics', fontsize=13)
    saved.append(_save(fig, out_dir, '08_redundancy.png'))

    # 09 — CARA MILP solve times over run
    fig, axes = plt.subplots(1, 2, figsize=(13,5))
    for name, _, col, _ in [s for s in SIMS if 'CARA' in s[0]]:
        for res in results_by_name[name]:
            if res.milp_ms_ts:
                xs = [s for s,_ in res.milp_ms_ts]
                ys = [v for _,v in res.milp_ms_ts]
                axes[0].scatter(xs, ys, color=col, s=30, alpha=0.7, label=f"{name} s={res.seed}")
    axes[0].set_xlabel('Timestep'); axes[0].set_ylabel('MILP solve time (ms)')
    axes[0].set_title('MILP Solve Times per Trigger')
    axes[0].legend(fontsize=7); axes[0].grid(alpha=0.3)
    # Hold gate ticks per seed
    cara_names = [s[0] for s in SIMS if 'CARA' in s[0]]
    cara_cols  = {s[0]: s[2] for s in SIMS if 'CARA' in s[0]}
    x2 = np.arange(n_seeds); w2 = 0.3
    for i, name in enumerate(cara_names):
        vals = [r.hold for r in sorted(results_by_name[name], key=lambda r: r.seed)]
        axes[1].bar(x2 + (i-0.5)*w2, vals, w2, color=cara_cols[name], label=name, alpha=0.8)
    axes[1].set_xticks(x2); axes[1].set_xticklabels([str(s) for s in seeds])
    axes[1].set_xlabel('Seed'); axes[1].set_ylabel('Robot-ticks at hold gate')
    axes[1].set_title('Hold Gate Latency (CARA-Dynamic only)')
    axes[1].legend(fontsize=7); axes[1].grid(alpha=0.3, axis='y')
    fig.suptitle('CARA Execution Layer Metrics', fontsize=13)
    saved.append(_save(fig, out_dir, '09_cara_exec_layer.png'))

    # 11 — Flip-flop rate
    fig, ax = plt.subplots(figsize=(11,5))
    _plot_metric(ax, results_by_name, lambda r: r.flipflop_ts,
                 'Flip-flopping robots (%)',
                 'Flip-Flop Rate — robots revisiting recent positions\n'
                 '(indicates oscillation / no committed frontier)',
                 steps, ylim=(0, None))
    ax.annotate('Lower = better  |  High GNF/GreedyRL = robots bouncing off same frontier\n'
                'CARA spikes at re-solve ticks when assignments change',
                xy=(0.01, 0.97), xycoords='axes fraction', ha='left', va='top',
                fontsize=8, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    saved.append(_save(fig, out_dir, '11_flipflop.png'))

    # 12 — Idle movement rate
    fig, ax = plt.subplots(figsize=(11,5))
    _plot_metric(ax, results_by_name, lambda r: r.idle_move_ts,
                 'Moving robots gaining no new cells (%)',
                 'Idle Movement Rate — robots moving without information gain\n'
                 '(transit through known terrain / waiting / replanning)',
                 steps, ylim=(0, None))
    ax.annotate('Lower = better  |  Spikes = replanning steps or robots in transit\n'
                'CARA-Base high between MILP solves (stale assignments)',
                xy=(0.01, 0.97), xycoords='axes fraction', ha='left', va='top',
                fontsize=8, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    saved.append(_save(fig, out_dir, '12_idle_movement.png'))

    # 10 — Summary dashboard
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

    # Bottom row: compute cost bar, survivor time-series, summary text
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
                 lambda r: [(s, v/r.total_survivors*100) for s,v in r.found_ts],
                 'Survivors found (%)', 'Survivor Discovery Over Time', steps, ylim=(0,105))

    ax_txt = fig.add_subplot(gs[1, 4]); ax_txt.axis('off')
    lines  = [
        'Five-Way Summary',
        '─' * 30,
        f"Seeds: {n_seeds}  |  Steps: {steps}",
        '',
        f"{'Sim':<14} {'Cov%':>5} {'Stair':>6} {'Found':>6} {'ms':>6}",
        '─' * 38,
    ]
    for name in names:
        d = avgs[name]
        lines.append(f"{name:<14} {d['final_cov']:5.1f} {d['final_stair']:6.1f}"
                     f" {d['final_found']:6.1f} {d['avg_ms']:6.1f}")
    lines += ['', 'CARA extras:',
              f"  Base:    MILP {avgs['CARA-Base']['milp_avg']:.0f}ms avg",
              f"  Dynamic: {avgs['CARA-Dynamic']['hold']:.0f} hold-ticks",
              f"           {avgs['CARA-Dynamic']['ejects']:.0f} eject events"]
    ax_txt.text(0.03, 0.97, '\n'.join(lines), transform=ax_txt.transAxes,
                fontsize=8, va='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    fig.suptitle(
        f'Fleet vs GNF vs GreedyRL vs CARA-Base vs CARA-Dynamic  —  '
        f'{n_seeds} Seed{"s" if n_seeds>1 else ""} x {steps} Steps',
        fontsize=12, fontweight='bold')
    saved.append(_save(fig, out_dir, '10_dashboard.png'))

    print("done")
    print(f"\nPlots saved to: {out_dir}/")
    for f in saved:
        print(f"  {f}")
    print(f"  summary.json")


if __name__ == '__main__':
    main()