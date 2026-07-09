#!/usr/bin/env python3
"""
ScalingBenchL.py — Framework L fleet-size scaling curve.

Sweeps fleet size N ∈ {10..100, 1000} for Framework L ONLY (belief-safe K).
No baseline models — this is a scaling study of L in isolation.

KEY DESIGN — constant robot density:
    The map grows with the fleet so cells-per-robot stays ≈ constant. This
    isolates "does per-step COORDINATION cost scale with N" from "does the
    mission get trivially easy/hard as robots are packed denser". Anchor:
    50 robots on 200×200 = 40000 cells ⇒ 800 cells/robot. Every N uses
    grid = round(sqrt(N × 800)), clamped even, so density is held.
    (Set --fixed-grid to instead hold the map at 200×200 and let density rise
    — useful as a contrasting saturation curve.)

Robot TYPE MIX held at the large-bench proportion:
    Legged 26% / Drone 34% / Boat 16% / Rover 24%  (13:17:8:12 → 50).
Survivors scale with map area at the large-bench density (45 per 40000 cells).

COMPUTE INSTRUMENTATION (per step, summarised over the run):
    step_total     — full sim.step() wall time
    role           — _pg_best_response_roles (potential game)
    alloc          — _assign_zones_cbba (HRBA)
    cover          — relay coverage union + disk machinery
    astar_sum      — Σ over robots of A* time on the single sim CPU
    astar_max      — slowest single robot's A* this step (parallel proxy)
    ground         — step_total − astar_sum  (centralised coordination only)
    dist_latency   — ground + astar_max      (realistic per-step latency if
                                              each robot plans onboard)
  Each reported as mean, p50, p90, max, AND per-robot (mean / N) so O(N)
  vs O(N²) vs O(1) phase scaling is directly readable.

OUTPUTS (to --out dir):
    scaling_L_summary.xlsx  — formatted workbook, one row per (N, seed) on a
                              raw sheet + an aggregated-by-N sheet + a config
                              sheet + a phase-scaling sheet with Excel formulas.
    scaling_L_long.csv      — tidy/long format, one row per (N, seed, metric)
                              — ideal for R (read.csv → ggplot/lm/anova).
    scaling_L_wide.csv      — one row per (N, seed), all metrics as columns.
    scaling_L_curve.png     — quick-look scaling plots.

Usage:
    python3 ScalingBenchL.py --seeds 3 --steps 1500       # seeds 0,1,2
    python3 ScalingBenchL.py --quick                     # tiny smoke
    python3 ScalingBenchL.py --sizes 10,20,50 --steps 800
    python3 ScalingBenchL.py --seeds 1,2,7,42             # explicit seeds
    PYTHONHASHSEED=0 python3 ScalingBenchL.py --seeds 5  # reproducible
"""
import argparse, csv, importlib.util, math, os, random, sys, time
from collections import defaultdict
import numpy as np

# ── Load Framework L ─────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_SEARCH_DIRS = [_HERE, os.path.join(_HERE, '..'), os.path.join(_HERE, '..', 'outputs'),
                '/mnt/user-data/outputs', '/home/claude']

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
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod
    spec.loader.exec_module(mod); return mod
L = _load(_find('2DFleetFrameworkM.py', 'FleetFrameworkM.py',
                 'FleetFrameworkL.py', 'hetero_robot_fleet_sim.py'), 'FrameworkM')

# ── A* timing wrap (module-global accumulator, reset each step) ───────────────
_ASTAR = {'sum': 0.0, 'max': 0.0, 'n': 0}
def _wrap_astar(cls):
    _orig = cls.search
    def _timed(*a, **k):
        t0 = time.perf_counter(); r = _orig(*a, **k)
        dt = (time.perf_counter() - t0) * 1000.0
        _ASTAR['sum'] += dt
        if dt > _ASTAR['max']: _ASTAR['max'] = dt
        _ASTAR['n'] += 1
        return r
    cls.search = staticmethod(_timed)
if hasattr(L, 'AStar') and hasattr(L.AStar, 'search'):
    _wrap_astar(L.AStar)

# ── Scaling configuration ────────────────────────────────────────────────────
DENSITY_CELLS_PER_ROBOT = 800          # 50 robots @ 200×200 anchor
SURV_PER_CELL           = 45 / 40000.0  # large-bench survivor density
TYPE_MIX = [("Legged", 13), ("Drone", 17), ("Boat", 8), ("Rover", 12)]
_MIX_TOTAL = sum(n for _, n in TYPE_MIX)  # 50

def grid_for(n_robots, fixed_grid):
    if fixed_grid:
        return 200
    g = int(round(math.sqrt(n_robots * DENSITY_CELLS_PER_ROBOT)))
    return max(60, g + (g & 1))          # even, min 60

def counts_for(n_robots):
    """Split N into the four types preserving the mix; remainder to Drone."""
    raw = {t: n_robots * n / _MIX_TOTAL for t, n in TYPE_MIX}
    out = {t: int(math.floor(v)) for t, v in raw.items()}
    short = n_robots - sum(out.values())
    for t in sorted(raw, key=lambda k: raw[k] - out[k], reverse=True):
        if short <= 0: break
        out[t] += 1; short -= 1
    return out

# ── Build a Framework L sim at a given scale (patches module constants) ───────
def make_sim(n_robots, grid, n_survivors):
    L.GRID_W = grid; L.GRID_H = grid
    counts = counts_for(n_robots)

    def _build_robots(self):
        Cap = L.Capability
        templates = {
            "Legged": ({Cap.LAND, Cap.STAIRS}, np.array([10., 10.]), (L.TEMP_LIMIT, L.RAD_LIMIT)),
            "Drone":  ({Cap.AIR},              np.array([10., 10.]), (L.TEMP_LIMIT, L.RAD_LIMIT)),
            "Boat":   ({Cap.WATER},            np.array([0., 0.]),   (9999., 9999.)),
            "Rover":  ({Cap.LAND},             np.array([-2., -2.]), (9999., 9999.)),
        }
        spawn = []
        for t, n in counts.items(): spawn += [t] * n
        random.shuffle(spawn)
        W, H = L.GRID_W, L.GRID_H
        clusters = [(W//6,H//6),(W//2,H//6),(5*W//6,H//6),
                    (W//6,H//2),(W//2,H//2),(5*W//6,H//2),
                    (W//6,5*H//6),(W//2,5*H//6),(5*W//6,5*H//6)]
        water_cells = [(x, y) for x in range(W) for y in range(H)
                       if self.world.grid[x][y]["t"] == L.T_WATER]
        self.robots = []
        for i, tname in enumerate(spawn):
            caps, weights, (tlim, rlim) = templates[tname]
            center = clusters[i % len(clusters)]
            if tname == "Boat" and water_cells:
                ns = [c for c in water_cells if not self.radio_shadow[c[0], c[1]]] or water_cells
                sx, sy = random.choice(ns)
            else:
                sx, sy = center
                for _ in range(60):
                    cx = max(1, min(W-2, center[0] + random.randint(-14, 14)))
                    cy = max(1, min(H-2, center[1] + random.randint(-14, 14)))
                    tt = self.world.grid[cx][cy]["t"]
                    if (tt == L.T_FREE or (tt == L.T_STAIRS and L.Capability.STAIRS in caps)) \
                            and not self.radio_shadow[cx, cy]:
                        sx, sy = cx, cy; break
            self.robots.append(L.Robot(f"{tname}{i}", sx, sy, caps, self.world,
                                        self, weights, tlim, rlim))

    def _build_survivors(self):
        W, H = L.GRID_W, L.GRID_H
        free_open = [(x, y) for x in range(W) for y in range(H)
                     if self.world.grid[x][y]["t"] == L.T_FREE]
        stair = [(x, y) for x in range(W) for y in range(H)
                 if self.world.grid[x][y]["t"] == L.T_STAIRS]
        random.shuffle(free_open); random.shuffle(stair)
        n_stair = min(len(stair), int(round(n_survivors * 0.55)))
        chosen = stair[:n_stair] + free_open[:max(0, n_survivors - n_stair)]
        self.survivors = chosen[:n_survivors]

    L.FleetSim._build_robots = _build_robots
    L.FleetSim._build_survivors = _build_survivors
    return L.FleetSim()

# ── Run one (N, seed) ────────────────────────────────────────────────────────
def run_one(n_robots, seed, steps, fixed_grid):
    grid = grid_for(n_robots, fixed_grid)
    n_surv = max(5, int(round(grid * grid * SURV_PER_CELL)))
    random.seed(seed); np.random.seed(seed)
    sim = make_sim(n_robots, grid, n_surv)
    actual_n = len(sim.robots); actual_surv = len(sim.survivors)

    # phase timers
    ph = defaultdict(float)
    def timed(method, key):
        def w(*a, **k):
            t0 = time.perf_counter()
            try: return method(*a, **k)
            finally: ph[key] += time.perf_counter() - t0
        return w
    if hasattr(sim, '_pg_best_response_roles'):
        sim._pg_best_response_roles = timed(sim._pg_best_response_roles, 'role')
    if hasattr(sim, '_assign_zones_cbba'):
        sim._assign_zones_cbba = timed(sim._assign_zones_cbba, 'alloc')
    if hasattr(sim, '_active_relay_coverage_union'):
        sim._active_relay_coverage_union = timed(sim._active_relay_coverage_union, 'cover')

    step_ms, ground_ms, amax_ms, asum_ms, dist_ms = [], [], [], [], []
    t_first = t_half = t_all = -1
    peak_relays = 0
    t_wall = time.time()
    Role = getattr(L, 'Role', None)

    for step in range(1, steps + 1):
        _ASTAR['sum'] = 0.0; _ASTAR['max'] = 0.0; _ASTAR['n'] = 0
        t0 = time.perf_counter()
        alive = sim.step()
        dt_ms = (time.perf_counter() - t0) * 1000.0
        asum = _ASTAR['sum']; amax = _ASTAR['max']
        ground = max(0.0, dt_ms - asum)
        step_ms.append(dt_ms); asum_ms.append(asum); amax_ms.append(amax)
        ground_ms.append(ground); dist_ms.append(ground + amax)

        nf = len(sim.found)
        if nf >= 1 and t_first < 0: t_first = step
        if nf >= actual_surv // 2 and t_half < 0: t_half = step
        if nf >= actual_surv and t_all < 0: t_all = step
        if Role is not None:
            nr = sum(1 for r in sim.robots if r.active and r.role == Role.RELAY)
            peak_relays = max(peak_relays, nr)
        if not alive: break

    wall = time.time() - t_wall
    steps_run = sim.timestep
    cov = float(np.mean(sim.union_belief != 0)) * 100.0
    found = len(sim.found)
    deaths = sum(1 for r in sim.robots if not r.active)
    hazard = sum(1 for r in sim.robots if getattr(r, 'hazard_killed', False))
    battery = sum(1 for _, rs in getattr(sim, 'dead_robots', []) if 'batt' in (rs or '').lower())

    def stats(a):
        if not a: return dict(mean=0, p50=0, p90=0, max=0)
        A = np.array(a)
        return dict(mean=float(A.mean()), p50=float(np.percentile(A,50)),
                     p90=float(np.percentile(A,90)), max=float(A.max()))
    S = {k: stats(v) for k, v in
         dict(step=step_ms, ground=ground_ms, astar_sum=asum_ms,
              astar_max=amax_ms, dist=dist_ms).items()}

    # phase totals → ms/step
    role_ms  = ph['role']  * 1000.0 / max(1, steps_run)
    alloc_ms = ph['alloc'] * 1000.0 / max(1, steps_run)
    cover_ms = ph['cover'] * 1000.0 / max(1, steps_run)

    row = dict(
        n_requested=n_robots, n_actual=actual_n, grid=grid,
        cells_per_robot=round(grid*grid/max(1,actual_n), 1),
        seed=seed, steps_run=steps_run, survivors=actual_surv,
        final_cov=round(cov,2), final_found=found,
        found_frac=round(found/max(1,actual_surv),3),
        deaths=deaths, hazard_deaths=hazard, battery_deaths=battery,
        peak_relays=peak_relays,
        t_first_found=t_first, t_half_found=t_half, t_all_found=t_all,
        completed=int(t_all > 0),
        wall_s=round(wall,1),
        step_mean_ms=round(S['step']['mean'],3),  step_p50_ms=round(S['step']['p50'],3),
        step_p90_ms=round(S['step']['p90'],3),    step_max_ms=round(S['step']['max'],3),
        ground_mean_ms=round(S['ground']['mean'],3), ground_p90_ms=round(S['ground']['p90'],3),
        astar_sum_mean_ms=round(S['astar_sum']['mean'],3),
        astar_max_mean_ms=round(S['astar_max']['mean'],3),
        astar_max_p90_ms=round(S['astar_max']['p90'],3),
        dist_mean_ms=round(S['dist']['mean'],3), dist_p90_ms=round(S['dist']['p90'],3),
        role_ms=round(role_ms,3), alloc_ms=round(alloc_ms,3), cover_ms=round(cover_ms,3),
        # per-robot normalisations (reveal O(N) vs O(1) phase scaling)
        step_ms_per_robot=round(S['step']['mean']/max(1,actual_n),4),
        ground_ms_per_robot=round(S['ground']['mean']/max(1,actual_n),4),
        role_ms_per_robot=round(role_ms/max(1,actual_n),4),
        alloc_ms_per_robot=round(alloc_ms/max(1,actual_n),4),
        cover_ms_per_robot=round(cover_ms/max(1,actual_n),4),
    )
    return row

# ── Excel writer ─────────────────────────────────────────────────────────────
def write_xlsx(rows, agg, cfg, path):
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.chart import LineChart, Reference

    HEAD = Font(name='Arial', bold=True, color='FFFFFF', size=10)
    HFILL = PatternFill('solid', fgColor='2F5496')
    NORM = Font(name='Arial', size=10)
    CEN = Alignment(horizontal='center'); THIN = Side(style='thin', color='D9D9D9')
    BORD = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)

    def sheet_from(ws, headers, data_rows):
        ws.append(headers)
        for c in range(1, len(headers)+1):
            cell = ws.cell(1, c); cell.font = HEAD; cell.fill = HFILL
            cell.alignment = CEN; cell.border = BORD
        for r in data_rows:
            ws.append([r.get(h) for h in headers])
        for row in ws.iter_rows(min_row=2, max_row=ws.max_row):
            for cell in row:
                cell.font = NORM; cell.border = BORD
                if isinstance(cell.value, float): cell.alignment = CEN
        for col in ws.columns:
            w = max((len(str(c.value)) for c in col if c.value is not None), default=8)
            ws.column_dimensions[col[0].column_letter].width = min(16, max(9, w+2))
        ws.freeze_panes = 'A2'

    wb = Workbook()

    # 1) Raw per-(N,seed)
    raw_headers = list(rows[0].keys())
    sheet_from(wb.active, raw_headers, rows); wb.active.title = 'raw_by_seed'

    # 2) Aggregated by N (means across seeds) with Excel formulas linking to raw
    ws = wb.create_sheet('agg_by_N')
    agg_headers = list(agg[0].keys())
    sheet_from(ws, agg_headers, agg)

    # 3) Phase-scaling sheet: ratios computed with Excel formulas
    ws3 = wb.create_sheet('phase_scaling')
    ph_headers = ['n_actual','step_mean_ms','ground_mean_ms','role_ms','alloc_ms',
                  'cover_ms','astar_sum_mean_ms','astar_max_mean_ms','dist_mean_ms']
    sheet_from(ws3, ph_headers, agg)
    # add derived columns via formulas (ground share, per-robot)
    last = ws3.max_row
    extra = ['ground_share_%','astar_share_%','ground_per_robot_ms']
    base = len(ph_headers)
    for j, name in enumerate(extra):
        c = ws3.cell(1, base+1+j, name); c.font = HEAD; c.fill = HFILL; c.alignment = CEN; c.border = BORD
    for i in range(2, last+1):
        ws3.cell(i, base+1, f'=100*C{i}/B{i}').font = NORM       # ground/step
        ws3.cell(i, base+2, f'=100*G{i}/B{i}').font = NORM       # astar_sum/step
        ws3.cell(i, base+3, f'=C{i}/A{i}').font = NORM           # ground per robot
    for col in ws3.columns:
        ws3.column_dimensions[col[0].column_letter].width = 16

    # 4) Chart: ground_mean_ms vs n_actual
    chart = LineChart(); chart.title = "Framework L — coordination cost vs fleet size"
    chart.x_axis.title = "robots (N)"; chart.y_axis.title = "ms/step"
    data = Reference(ws, min_col=agg_headers.index('ground_mean_ms')+1,
                      min_row=1, max_row=len(agg)+1)
    cats = Reference(ws, min_col=agg_headers.index('n_actual')+1,
                      min_row=2, max_row=len(agg)+1)
    chart.add_data(data, titles_from_data=True); chart.set_categories(cats)
    ws.add_chart(chart, f'A{len(agg)+4}')

    # 5) Config sheet
    wsc = wb.create_sheet('config')
    wsc.append(['key','value'])
    wsc.cell(1,1).font = HEAD; wsc.cell(1,1).fill = HFILL
    wsc.cell(1,2).font = HEAD; wsc.cell(1,2).fill = HFILL
    for k, v in cfg.items():
        wsc.append([k, str(v)])
    wsc.column_dimensions['A'].width = 26; wsc.column_dimensions['B'].width = 50

    wb.save(path)

# ── CSV writers (tidy long + wide) ───────────────────────────────────────────
def write_csv(rows, wide_path, long_path):
    headers = list(rows[0].keys())
    with open(wide_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=headers); w.writeheader()
        for r in rows: w.writerow(r)
    id_cols = ['n_requested','n_actual','grid','cells_per_robot','seed']
    metric_cols = [h for h in headers if h not in id_cols]
    with open(long_path, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(id_cols + ['metric','value'])
        for r in rows:
            for m in metric_cols:
                w.writerow([r[c] for c in id_cols] + [m, r[m]])

# ── Plot ─────────────────────────────────────────────────────────────────────
def plot_curve(agg, path):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    N = [a['n_actual'] for a in agg]
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    ax[0,0].plot(N, [a['ground_mean_ms'] for a in agg], 'o-', label='ground (coord)')
    ax[0,0].plot(N, [a['step_mean_ms'] for a in agg], 's--', label='full step (sim)')
    ax[0,0].plot(N, [a['dist_mean_ms'] for a in agg], '^:', label='dist latency')
    ax[0,0].set_title('Compute vs fleet size'); ax[0,0].set_xlabel('N'); ax[0,0].set_ylabel('ms/step')
    ax[0,0].legend(); ax[0,0].grid(alpha=.3)
    ax[0,1].plot(N, [a['ground_ms_per_robot'] for a in agg], 'o-', color='purple')
    ax[0,1].set_title('Coordination ms per robot (flat = O(N) total)')
    ax[0,1].set_xlabel('N'); ax[0,1].set_ylabel('ms/step/robot'); ax[0,1].grid(alpha=.3)
    for k, mk in [('role_ms','role'),('alloc_ms','alloc'),('cover_ms','cover')]:
        ax[1,0].plot(N, [a[k] for a in agg], 'o-', label=mk)
    ax[1,0].set_title('Coordination phase breakdown'); ax[1,0].set_xlabel('N')
    ax[1,0].set_ylabel('ms/step'); ax[1,0].legend(); ax[1,0].grid(alpha=.3)
    ax[1,1].plot(N, [a['final_found_frac_mean']*100 for a in agg], 'o-', color='green', label='found %')
    ax[1,1].plot(N, [a['final_cov'] for a in agg], 's--', color='teal', label='coverage %')
    ax[1,1].set_title('Mission outcome vs fleet size'); ax[1,1].set_xlabel('N')
    ax[1,1].set_ylabel('%'); ax[1,1].legend(); ax[1,1].grid(alpha=.3); ax[1,1].set_ylim(0,105)
    fig.suptitle('Framework L — fleet-size scaling', fontsize=13, fontweight='bold')
    fig.tight_layout(); fig.savefig(path, dpi=120, bbox_inches='tight'); plt.close(fig)

def _parse_seeds(spec):
    """--seeds accepts either a COUNT ('10' -> seeds 0..9) or an EXPLICIT
    comma list ('1,2,7,42' -> exactly those seeds), same style as --sizes.
    A bare count is what most runs want; an explicit list lets you add specific
    seeds to an existing dataset without renumbering or rerunning 0..N-1."""
    spec = str(spec).strip()
    if ',' in spec:
        return [int(s) for s in spec.split(',') if s.strip() != '']
    return list(range(int(spec)))

# ── Aggregate across seeds ───────────────────────────────────────────────────
def aggregate(rows):
    by_n = defaultdict(list)
    for r in rows: by_n[r['n_requested']].append(r)
    agg = []
    for n in sorted(by_n):
        g = by_n[n]
        def mean(k): return round(float(np.mean([x[k] for x in g])), 3)
        agg.append(dict(
            n_requested=n, n_actual=g[0]['n_actual'], grid=g[0]['grid'],
            cells_per_robot=g[0]['cells_per_robot'], n_seeds=len(g),
            final_cov=mean('final_cov'),
            final_found_frac_mean=mean('found_frac'),
            completed_frac=round(float(np.mean([x['completed'] for x in g])),3),
            deaths=mean('deaths'), hazard_deaths=mean('hazard_deaths'),
            battery_deaths=mean('battery_deaths'), peak_relays=mean('peak_relays'),
            t_all_found=mean('t_all_found'),
            step_mean_ms=mean('step_mean_ms'), step_p90_ms=mean('step_p90_ms'),
            ground_mean_ms=mean('ground_mean_ms'), ground_p90_ms=mean('ground_p90_ms'),
            astar_sum_mean_ms=mean('astar_sum_mean_ms'),
            astar_max_mean_ms=mean('astar_max_mean_ms'),
            dist_mean_ms=mean('dist_mean_ms'), dist_p90_ms=mean('dist_p90_ms'),
            role_ms=mean('role_ms'), alloc_ms=mean('alloc_ms'), cover_ms=mean('cover_ms'),
            ground_ms_per_robot=mean('ground_ms_per_robot'),
            step_ms_per_robot=mean('step_ms_per_robot'),
        ))
    return agg

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--sizes', type=str, default='10,20,30,40,50,60,70,80,90,100,1000')
    p.add_argument('--seeds', type=str, default='10',
                    help="Either a COUNT (e.g. '10' -> seeds 0..9) or an "
                         "explicit comma list (e.g. '1,2,7,42') to add "
                         "specific seeds without renumbering. Default: 10.")
    p.add_argument('--steps', type=int, default=1500)
    p.add_argument('--fixed-grid', action='store_true',
                    help='hold map at 200×200 (density rises with N) instead of density-held growth')
    p.add_argument('--max-cells', type=int, default=360000,
                    help='skip N whose density-held grid exceeds this many cells (sim memory ceiling; ~600x600). Default 360000.')
    p.add_argument('--quick', action='store_true')
    p.add_argument('--out', type=str, default='scaling_L_out')
    a = p.parse_args()
    if a.quick:
        a.sizes = '10,20'; a.seeds = '1'; a.steps = 120
    sizes = [int(s) for s in a.sizes.split(',')]
    seed_list = _parse_seeds(a.seeds)
    os.makedirs(a.out, exist_ok=True)

    print(f"Framework L scaling | sizes={sizes} | seeds={seed_list} | steps={a.steps} "
          f"| {'FIXED 200×200' if a.fixed_grid else 'density-held grid'}")
    rows = []
    skipped = []
    t0 = time.time()
    for n in sizes:
        g = grid_for(n, a.fixed_grid)
        # Guard rail: very large grids exhaust memory in this single-process,
        # dense-array simulator. Flag rather than crash — the ceiling is a
        # SIMULATION-IMPLEMENTATION limit (full-grid arrays × per-robot A*),
        # not a Framework-L coordination limit, and is reported as such.
        est_cells = g * g
        if est_cells > a.max_cells:
            print(f"  N={n:<5} grid={g}×{g} ({est_cells:,} cells) SKIPPED "
                  f"— exceeds --max-cells={a.max_cells:,} (sim memory ceiling, "
                  f"not a framework limit)")
            skipped.append(dict(n_requested=n, grid=g, cells=est_cells,
                                 reason='exceeds_sim_memory_ceiling'))
            continue
        for s in seed_list:
            print(f"  N={n:<5} grid={g}×{g} seed={s} ... ", end='', flush=True)
            try:
                r = run_one(n, s, a.steps, a.fixed_grid)
                print(f"cov={r['final_cov']:5.1f}% found={r['final_found']}/{r['survivors']} "
                      f"ground={r['ground_mean_ms']:.1f}ms step={r['step_mean_ms']:.1f}ms "
                      f"dist={r['dist_mean_ms']:.1f}ms")
                rows.append(r)
            except MemoryError:
                print("MemoryError — sim ceiling reached, flagged")
                skipped.append(dict(n_requested=n, grid=g, cells=est_cells,
                                     reason='MemoryError_at_runtime')); break
            except Exception as e:
                import traceback; print("FAIL:", e); traceback.print_exc()
    print(f"total {(time.time()-t0)/60:.1f} min")

    if not rows:
        print("no rows — aborting"); return
    agg = aggregate(rows)
    cfg = dict(framework='L (belief-safe K)', sizes=sizes, seeds=seed_list, steps=a.steps,
               grid_policy=('fixed 200x200' if a.fixed_grid else 'density-held (~800 cells/robot)'),
               type_mix='Legged26/Drone34/Boat16/Rover24 %',
               survivor_density=f'{SURV_PER_CELL:.6f} per cell',
               pythonhashseed=os.environ.get('PYTHONHASHSEED', 'unset'),
               skipped_sizes=';'.join(f"N={x['n_requested']}(grid={x['grid']},{x['reason']})" for x in skipped) or 'none',
               generated=time.strftime('%Y-%m-%d %H:%M:%S'))
    write_csv(rows, os.path.join(a.out, 'scaling_L_wide.csv'),
                    os.path.join(a.out, 'scaling_L_long.csv'))
    try:
        write_xlsx(rows, agg, cfg, os.path.join(a.out, 'scaling_L_summary.xlsx'))
    except Exception as e:
        print("xlsx write failed:", e)
    try:
        plot_curve(agg, os.path.join(a.out, 'scaling_L_curve.png'))
    except Exception as e:
        print("plot failed:", e)
    print(f"\nwrote → {a.out}/  (scaling_L_summary.xlsx, scaling_L_long.csv, "
          f"scaling_L_wide.csv, scaling_L_curve.png)")

if __name__ == '__main__':
    main()