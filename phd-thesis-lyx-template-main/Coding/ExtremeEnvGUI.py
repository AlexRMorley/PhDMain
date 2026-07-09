"""
Extreme-Environment GUI Runner  —  "Coastal Nuclear Industrial Disaster"
========================================================================
STRUCTURED disaster-response scenario (purpose-built, not a random map).
Static map setup only — no dynamic events.
"""

import sys, os, math, random, importlib.util, argparse
import numpy as np

MODELS = ('fleet', 'gnf', 'greedy', 'cara-base', 'cara-dynamic')


def _seed(value):
    """A non-negative integer RNG seed."""
    try:
        n = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is not an integer")
    if n < 0:
        raise argparse.ArgumentTypeError("seed must be >= 0")
    return n


def _zoom(value):
    """Display zoom in pixels per cell (rendering only)."""
    try:
        n = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is not an integer")
    if not 1 <= n <= 16:
        raise argparse.ArgumentTypeError("--px must be between 1 and 16")
    return n


ap = argparse.ArgumentParser(
    prog="ExtremeEnvGUI.py",
    description="Coastal Nuclear Industrial Disaster — heterogeneous SAR fleet runner.",
    epilog="Set PYTHONHASHSEED=0 for any run you intend to report.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
ap.add_argument("--model", choices=MODELS, default="fleet",
                help="algorithm to run")
ap.add_argument("--seed", type=_seed, default=0, metavar="N",
                help="RNG seed for terrain, robot spawns and survivor placement")
ap.add_argument("--px", type=_zoom, default=4, metavar="{1..16}",
                help="display zoom, pixels per cell; rendering only, "
                     "does not affect the simulation")
ap.add_argument("--preview", action="store_true",
                help="render a terrain+hazard PNG headless and exit (no GUI)")
args = ap.parse_args()

USER_SEED  = args.seed
MODEL_NAME = args.model
DISPLAY_PX = args.px

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
_SEARCH_DIRS = [BASE_DIR, os.path.join(BASE_DIR, '..')]

def _find(*names):
    for name in names:
        for folder in _SEARCH_DIRS:
            p = os.path.join(folder, name)
            if os.path.exists(p):
                return os.path.abspath(p)
    print("ERROR: Cannot find any of:", list(names)); sys.exit(1)

def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

M      = _load(_find('2DFleetFrameworkM.py', 'hetero_robot_fleet_sim.py'), 'fleet')
M_gnf  = _load(_find('GNF_Sim.py',  'gnf_sim.py',  'GNF_sim.py'),         'gnf')
M_rl   = _load(_find('GreedyRL.py', 'greedy_rl_sim.py', 'greedyRL.py'),   'rl')
M_cara = _load(_find('Cara_sim.py', 'cara_sim.py'),                        'cara')

T_UNKNOWN, T_FREE, T_OBS, T_STAIRS, T_WATER, T_BRIDGE = (
    M.T_UNKNOWN, M.T_FREE, M.T_OBS, M.T_STAIRS, M.T_WATER, M.T_BRIDGE)

EXT_GRID_W      = 200
EXT_GRID_H      = 200
EXT_CELL_SIZE   = 5
EXT_ZONE_CHUNKS = 10
EXT_MAX_BATTERY = 5000
EXT_MAX_BUNDLE  = 8
EXT_N_SURVIVORS = 30
EXT_REVEAL_R    = 3
EXT_ROBOTS      = {"Legged": 8, "Drone": 13, "Boat": 6, "Rover": 10}

sys.path.insert(0, BASE_DIR)
import ExtremeScenario as scen
scen.install(M, M_gnf, M_rl,
             grid_w=EXT_GRID_W, grid_h=EXT_GRID_H, cell=DISPLAY_PX,
             zone_chunks=EXT_ZONE_CHUNKS, max_battery=EXT_MAX_BATTERY,
             max_bundle=EXT_MAX_BUNDLE, n_survivors=EXT_N_SURVIVORS,
             reveal_r=EXT_REVEAL_R, robots=EXT_ROBOTS)

_OrigFleetSim = M.FleetSim
GNFSim        = M_gnf.GNFSim
GreedyRLSim   = M_rl.make_greedy_rl_sim(M_gnf, M)
try:
    CARABase    = M_cara.make_cara_sim(M_gnf, M, use_exec_layer=False)
    CARADynamic = M_cara.make_cara_sim(M_gnf, M, use_exec_layer=True)
except TypeError:
    CARABase    = M_cara.make_cara_sim(M_gnf, M)
    CARADynamic = M_cara.make_cara_sim(M_gnf, M)

# ── Stable snapshot of M for baseline sims (fixes a pre-existing ordering bug) ─
# GNFSim / CARABase / CARADynamic mix in unbound methods via M.FleetSim, e.g.
# `M.FleetSim._build_radio_shadow(self)`, so they need M.FleetSim to still be
# the ORIGINAL class at the moment they are instantiated. But M.FleetSim below
# gets reassigned to a seeding wrapper for the GUI loop's `sim = M.FleetSim()`
# call, and the baseline factories are LAZY (only invoked when a model other
# than 'fleet' is chosen) -- so by the time GNFSim(M) etc. actually runs,
# M.FleetSim is the wrapper function, not the class, and
# `M.FleetSim._build_radio_shadow` raises AttributeError. Only the 'fleet'
# path was unaffected, since its factory captures _OrigFleetSim directly
# rather than going through M. A shallow snapshot with FleetSim pinned back
# to the original class fixes every baseline without touching
# GNF_Sim.py / Cara_sim.py / GreedyRL.py.
import types
_M_baseline = types.SimpleNamespace(**{k: v for k, v in vars(M).items()
                                         if not k.startswith('__')})
_M_baseline.FleetSim = _OrigFleetSim

_FACTORIES = {
    'fleet':        lambda: _OrigFleetSim(),
    'gnf':          lambda: GNFSim(_M_baseline),
    'greedy':       lambda: GreedyRLSim(_M_baseline),
    'cara-base':    lambda: CARABase(_M_baseline),
    'cara-dynamic': lambda: CARADynamic(_M_baseline),
}
_MODEL_LABELS = {
    'fleet': 'Fleet (FrameworkJ)', 'gnf': 'GNF', 'greedy': 'Greedy RL',
    'cara-base': 'CARA-Base', 'cara-dynamic': 'CARA-Dynamic',
}
_chosen_factory = _FACTORIES[MODEL_NAME]

def _sim_factory():
    random.seed(USER_SEED); np.random.seed(USER_SEED)
    return _chosen_factory()
M.FleetSim = _sim_factory

def _build_hazard_surface(sim, key, lethal, hot, floor):
    """Pre-render a static temperature ('temp') or radiation ('rad') overlay."""
    import pygame
    W, H = sim.world.w, sim.world.h
    arr = np.array([[sim.world.grid[x][y][key] for y in range(H)] for x in range(W)],
                   dtype=np.float32)
    span = max(lethal - floor, 1e-6)
    nrm = np.clip((arr - floor) / span, 0.0, 1.0)
    rgb = np.zeros((W, H, 3), dtype=np.uint8)
    if hot:
        rgb[..., 0] = (np.clip(nrm * 2.2,     0, 1) * 255).astype(np.uint8)
        rgb[..., 1] = (np.clip(nrm * 2.0 - 0.5, 0, 1) * 255).astype(np.uint8)
        rgb[..., 2] = (np.clip(nrm * 2.4 - 1.4, 0, 1) * 255).astype(np.uint8)
    else:
        rgb[..., 0] = (np.clip(nrm * 2.0,       0, 1) * 255).astype(np.uint8)
        rgb[..., 2] = (np.clip(0.4 + nrm * 1.6, 0, 1) * 255).astype(np.uint8)
        rgb[..., 1] = (np.clip(nrm * 2.6 - 1.6, 0, 1) * 255).astype(np.uint8)
    alpha = np.where(nrm <= 0.02, 0.0, np.clip(0.28 + 0.62 * nrm, 0, 0.92)) * 255
    alpha = alpha.astype(np.uint8)
    leth = arr >= lethal
    rgb[leth] = (0, 255, 255) if hot else (255, 255, 255)
    alpha[leth] = 240
    cs = M.CELL_SIZE; ones = np.ones((cs, cs), dtype=np.uint8)
    surf = pygame.Surface((W * cs, H * cs), pygame.SRCALPHA)
    prgb = pygame.surfarray.pixels3d(surf)
    for c in range(3):
        prgb[:, :, c] = np.kron(rgb[:, :, c], ones)
    del prgb
    pa = pygame.surfarray.pixels_alpha(surf)
    pa[:] = np.kron(alpha, ones); del pa
    return surf


def _ext_gui_loop():
    import pygame
    pygame.init()
    GW, GH, CS, SB = M.GRID_W, M.GRID_H, M.CELL_SIZE, M.SIDEBAR_WIDTH
    screen = pygame.display.set_mode((GW * CS + SB, GH * CS))
    pygame.display.set_caption("Coastal Nuclear Industrial Disaster")
    font = pygame.font.SysFont(None, 22)
    sim = M.FleetSim()

    running = False
    show_map = show_surv = show_risk = show_plans = show_zones = show_shadow = False
    show_temp = show_rad = False

    try:
        shadow_surf = M.build_shadow_surface(sim)
    except Exception:
        shadow_surf = None
    temp_surf = _build_hazard_surface(sim, "temp", M.TEMP_LIMIT, hot=True,  floor=20.0)
    rad_surf  = _build_hazard_surface(sim, "rad",  M.RAD_LIMIT,  hot=False, floor=8.0)
    grid_surf = None; last_vk = None; last_union_tick = -1

    BX = GW * CS + 10
    rows = ["start", "map", "surv", "risk", "temp", "rad", "plans", "zones", "shadow"]
    buttons = [(nm, pygame.Rect(BX, 10 + 25 * i, 120, 20)) for i, nm in enumerate(rows)]
    stats_y0 = 10 + 25 * len(rows) + 12

    clock = pygame.time.Clock()
    while True:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if ev.type == pygame.KEYDOWN:
                if ev.key in (pygame.K_q, pygame.K_ESCAPE): pygame.quit(); sys.exit()
                elif ev.key == pygame.K_SPACE: running = not running
                elif ev.key == pygame.K_t: show_temp = not show_temp
                elif ev.key == pygame.K_r: show_rad = not show_rad
                elif ev.key == pygame.K_p: show_plans = not show_plans
            if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                for name, rect in buttons:
                    if not rect.collidepoint(ev.pos): continue
                    if   name == "start":  running = not running
                    elif name == "map":    show_map = not show_map; grid_surf = None
                    elif name == "surv":   show_surv = not show_surv; grid_surf = None
                    elif name == "risk":   show_risk = not show_risk; grid_surf = None
                    elif name == "temp":   show_temp = not show_temp
                    elif name == "rad":    show_rad = not show_rad
                    elif name == "plans":  show_plans = not show_plans
                    elif name == "zones":  show_zones = not show_zones
                    elif name == "shadow": show_shadow = not show_shadow

        if running and not sim.step():
            running = False
            print(f"[DONE] t={sim.timestep}  found={len(sim.found)}/{len(sim.survivors)}")

        union = sim.union_belief; uT = sim.union_T; uR = sim.union_R
        vk = (show_map, show_surv, show_risk)
        union_changed = sim.timestep != last_union_tick
        if grid_surf is None or vk != last_vk or (union_changed and not show_map):
            grid_surf = M.build_grid_surface(sim, show_map, show_surv, show_risk, union, uT, uR)
            last_vk = vk; last_union_tick = sim.timestep

        screen.blit(grid_surf, (0, 0))
        if show_temp: screen.blit(temp_surf, (0, 0))
        if show_rad:  screen.blit(rad_surf,  (0, 0))
        if show_shadow and shadow_surf is not None:
            screen.blit(shadow_surf, (0, 0))
            try: M.draw_shadow_coverage(screen, sim)
            except Exception: pass
            try:
                import math as _math
                R_disk = getattr(M, 'RELAY_COVERAGE_RADIUS_CELLS', None)
                if R_disk is not None:
                    for r in sim.robots:
                        if not r.active or r.role.name != 'RELAY':
                            continue
                        anchor = getattr(r, 'relay_anchor', None)
                        cur = tuple(r.pos)
                        cx = int((cur[0] + 0.5) * M.CELL_SIZE)
                        cy = int((cur[1] + 0.5) * M.CELL_SIZE)
                        px_r = int(R_disk * M.CELL_SIZE)
                        pygame.draw.circle(screen, (255, 240, 80), (cx, cy), px_r, width=2)
                        if anchor is not None and anchor != cur:
                            ax = int((anchor[0] + 0.5) * M.CELL_SIZE)
                            ay = int((anchor[1] + 0.5) * M.CELL_SIZE)
                            pygame.draw.circle(screen, (200, 200, 60), (ax, ay), 3, width=1)
            except Exception: pass
        if show_zones:
            try: M.draw_zones(screen, sim)
            except Exception: pass
        M.draw_robots(screen, sim.robots, show_plans)

        pygame.draw.rect(screen, (255, 255, 255), (GW * CS, 0, SB, GH * CS))
        labels = {"start": "Pause" if running else "Start",
                  "map": "Hide Map" if show_map else "Show Map",
                  "surv": "Hide Surv" if show_surv else "Show Surv",
                  "risk": "Hide Risk" if show_risk else "Show Risk",
                  "temp": "Hide Temp" if show_temp else "Show Temp",
                  "rad":  "Hide Rad" if show_rad else "Show Rad",
                  "plans": "Hide Plans" if show_plans else "Show Plans",
                  "zones": "Hide Zones" if show_zones else "Show Zones",
                  "shadow": "Hide Shadow" if show_shadow else "Show Shadow"}
        hot = {"temp": (200, 60, 0), "rad": (120, 0, 160)}
        for name, rect in buttons:
            pygame.draw.rect(screen, hot.get(name, (200, 200, 200)), rect)
            fg = (255, 255, 255) if name in hot else (0, 0, 0)
            screen.blit(font.render(labels[name], True, fg), (rect.x + 6, rect.y + 4))

        y_sb = stats_y0
        disc = int(np.sum(union != M.T_UNKNOWN)); pct = disc / (GW * GH) * 100
        for line in [f"Step: {sim.timestep}", f"Coverage: {pct:.1f}%",
                     f"Survivors: {len(sim.found)}/{len(sim.survivors)}"]:
            screen.blit(font.render(line, True, (0, 0, 0)), (BX, y_sb)); y_sb += 22
        y_sb += 8
        screen.blit(font.render("Battery:", True, (0, 0, 0)), (BX, y_sb)); y_sb += 20
        groups = {"Legged": [], "Drone": [], "Boat": [], "Rover": []}
        alive_c = {"Legged": 0, "Drone": 0, "Boat": 0, "Rover": 0}
        for r in sim.robots:
            t = M.robot_type(r.name)
            if t in groups:
                groups[t].append(r.battery)
                if getattr(r, "active", True) and r.battery > 0:
                    alive_c[t] += 1
        for t in ("Legged", "Drone", "Boat", "Rover"):
            v = groups[t]; avg = sum(v) / len(v) if v else 0
            screen.blit(font.render(f"{t}: {avg:.0f} ({alive_c[t]}/{len(v)})", True,
                                    (0, 0, 0)), (BX, y_sb)); y_sb += 18

        y_sb += 8
        screen.blit(font.render("Robots:", True, (0, 0, 0)), (BX, y_sb)); y_sb += 20
        for nm, clr in M.ROBOT_COLOUR.items():
            pygame.draw.rect(screen, clr, (BX, y_sb, 14, 14))
            screen.blit(font.render(nm, True, (0, 0, 0)), (BX + 20, y_sb)); y_sb += 19

        pygame.display.flip()
        clock.tick(M.FPS)


def _render_preview(path="extreme_map_preview.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    random.seed(USER_SEED); np.random.seed(USER_SEED)
    sim = _OrigFleetSim(); wd = sim.world; W, H = wd.w, wd.h
    terr = np.array([[wd.grid[x][y]["t"] for x in range(W)] for y in range(H)])
    rad  = np.array([[wd.grid[x][y]["rad"]  for x in range(W)] for y in range(H)])
    temp = np.array([[wd.grid[x][y]["temp"] for x in range(W)] for y in range(H)])

    cmap = ListedColormap(["#1b1b1b", "#d9d2c5", "#5b4636",
                           "#e6c200", "#2b6cb0", "#c08a3e"])
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    ax = axes[0]
    ax.imshow(terr, cmap=cmap, vmin=0, vmax=5, origin="upper", interpolation="nearest")
    sx = [p[0] for p in sim.survivors]; sy = [p[1] for p in sim.survivors]
    ax.scatter(sx, sy, c="#e53e3e", s=22, marker="*", edgecolors="k", linewidths=0.4,
               label=f"survivors ({len(sim.survivors)})", zorder=5)
    col = {"Legged": "#2f855a", "Drone": "#d53f8c", "Boat": "#319795", "Rover": "#dd6b20"}
    for r in sim.robots:
        base = ''.join(ch for ch in r.name if not ch.isdigit()); rx, ry = r.pos
        ax.scatter([rx], [ry], c=col.get(base, "w"), s=24, marker="o",
                   edgecolors="k", linewidths=0.5, zorder=6)
    hcx, hcy = wd.hill_center
    ax.annotate("rocky hill\n(no shadow)", (hcx, hcy), color="k", fontsize=8,
                ha="center", va="center")
    ax.set_title("Terrain + survivors + robot spawns"); ax.set_xticks([]); ax.set_yticks([])
    ax.legend(handles=[Patch(facecolor="#d9d2c5", label="open"),
                       Patch(facecolor="#5b4636", label="wall/debris"),
                       Patch(facecolor="#e6c200", label="stairs / hill"),
                       Patch(facecolor="#2b6cb0", label="water"),
                       Patch(facecolor="#c08a3e", label="bridge/pier")],
              loc="upper right", fontsize=8, framealpha=0.9)
    ax = axes[1]
    ax.imshow(terr, cmap=cmap, vmin=0, vmax=5, origin="upper", alpha=0.35, interpolation="nearest")
    im = ax.imshow(np.ma.masked_less(rad, 5), cmap="inferno", origin="upper", alpha=0.85)
    ax.contour(rad, levels=[M.RAD_LIMIT], colors="cyan", linewidths=1.4)
    ax.set_title(f"Radiation (cyan = lethal {M.RAD_LIMIT:.0f})"); ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax = axes[2]
    ax.imshow(terr, cmap=cmap, vmin=0, vmax=5, origin="upper", alpha=0.35, interpolation="nearest")
    im = ax.imshow(np.ma.masked_less(temp, 20), cmap="turbo", origin="upper", alpha=0.9,
                   vmin=20, vmax=M.TEMP_LIMIT)
    ax.contour(temp, levels=[50, 80, 105], colors=["#7fbfff", "#ffd24d", "#ff7a45"],
               linewidths=0.8, alpha=0.9)
    ax.contour(temp, levels=[M.TEMP_LIMIT], colors="cyan", linewidths=1.6)
    ax.set_title(f"Temperature: plume + fires (cyan = lethal {M.TEMP_LIMIT:.0f})")
    ax.set_xticks([]); ax.set_yticks([]); fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"Coastal Nuclear Industrial Disaster — seed {USER_SEED}", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(path, dpi=110)
    print(f"  saved preview -> {path}")
    return sim

if __name__ == "__main__":
    N_ROBOTS = sum(EXT_ROBOTS.values()); label = _MODEL_LABELS[MODEL_NAME]
    print(f"\nCoastal Nuclear Industrial Disaster")
    print(f"  Model     : {label}")
    print(f"  Grid      : {EXT_GRID_W}x{EXT_GRID_H} @ {EXT_CELL_SIZE}m = 1km^2")
    print(f"  Robots    : {N_ROBOTS} ({', '.join(f'{n} {t}' for t, n in EXT_ROBOTS.items())})")
    print(f"  Seed      : {USER_SEED}")
    if args.preview:
        print("\nRendering preview PNG (headless)..."); _render_preview(); sys.exit(0)
    print(f"\nGUI buttons incl. independent Temp / Rad heat-maps.  Keys: SPACE T R P Q\n")
    _ext_gui_loop()