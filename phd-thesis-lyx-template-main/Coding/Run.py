"""
run.py  —  Unified simulator launcher
======================================
Lets you pick which simulator to watch:
  1  Fleet Sim      (full coordination: CBBA + potential game + relay)
  2  GNF            (greedy nearest-frontier, no coordination)
  3  GNF-Risk       (GNF + belief-based risk-aware pathing)
  4  CARA-2022      (faithful published CARA: PDR-triggered self-relaying)
  5  CARA-EL-Base   (capability-aware MILP allocation, relay-constrained)
  6  CARA-EL-Dyn    (CARA-EL + execution layer: hold gate / eject)
  7  ACHORD-insp    (droppable radios + predictive comms extension)

Usage:
    python run.py [--sim fleet|gnf|gnf-risk|cara2022|cara-base|cara-dyn|achord|achord-risk] [--seed N] [--steps N]
    ('--sim cara' is kept as an alias for cara-dyn)

If --sim is omitted a selector screen is shown on startup.
Press SPACE to pause/resume, R to reset, Q to quit.
"""

import sys, os, argparse, random, time
import numpy as np
import importlib.util

# ── Locate files relative to this script ─────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))

def _find(*names):
    for name in names:
        for folder in (_HERE, os.path.join(_HERE, '..', 'outputs'),
                       '/mnt/user-data/outputs', '/home/claude'):
            p = os.path.join(folder, name)
            if os.path.exists(p):
                return os.path.abspath(p)
    raise FileNotFoundError(
        f"Cannot find any of {names}. Place one in the same folder as run.py.")

FLEET_PATH = _find('2DFleetFrameworkN.py', '2DFleetFrameworkM.py', '2DFleetFrameworkI.py')
GNF_PATH   = _find('GNF_Sim.py', 'gnf_sim.py')
CARA_PATH  = _find('Cara_sim.py', 'Cara_simDynamic.py')
try:
    CARA_PAPER_PATH = _find('Cara_paper.py')
except FileNotFoundError:
    CARA_PAPER_PATH = None
try:
    ACHORD_PATH = _find('ACHORD.py', 'Achord_sim.py', 'achord_sim.py')
except FileNotFoundError:
    ACHORD_PATH = None
try:
    RITAGS_PATH = _find('Ritags_sim.py', 'ritags_sim.py', 'RITAGS.py')
except FileNotFoundError:
    RITAGS_PATH = None

def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    M = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(M)
    return M

# ── Load pygame (real, not mocked) ───────────────────────────────────────────
import pygame

M_fleet = _load(FLEET_PATH, 'fleet')

build_grid_surface   = M_fleet.build_grid_surface
build_shadow_surface = M_fleet.build_shadow_surface
draw_shadow_coverage = M_fleet.draw_shadow_coverage
draw_robots          = M_fleet.draw_robots
ROBOT_COLOUR         = M_fleet.ROBOT_COLOUR
robot_type           = M_fleet.robot_type
CELL_SIZE            = M_fleet.CELL_SIZE
GRID_W               = M_fleet.GRID_W
GRID_H               = M_fleet.GRID_H
SIDEBAR_WIDTH        = M_fleet.SIDEBAR_WIDTH
T_UNKNOWN            = M_fleet.T_UNKNOWN
Role                 = M_fleet.Role

M_gnf  = _load(GNF_PATH,  'gnf')
M_cara = _load(CARA_PATH, 'cara')

GNFSim = M_gnf.GNFSim
try:
    CARABaseSim = M_cara.make_cara_sim(M_gnf, M_fleet, use_exec_layer=False)
    CARADynSim  = M_cara.make_cara_sim(M_gnf, M_fleet, use_exec_layer=True)
except TypeError:   # older single-variant module
    CARABaseSim = CARADynSim = M_cara.make_cara_sim(M_gnf, M_fleet)
CARAPaperSim = None
if CARA_PAPER_PATH is not None:
    M_cara_paper = _load(CARA_PAPER_PATH, 'cara_paper')
    CARAPaperSim = M_cara_paper.make_cara_paper_sim(M_gnf, M_fleet)
ACHORDSim = None
ACHORDRiskSim = None
if ACHORD_PATH is not None:
    M_achord = _load(ACHORD_PATH, 'achord')
    ACHORDSim = M_achord.make_achord_sim(M_gnf, M_fleet)
    if hasattr(M_achord, 'make_achord_risk_sim'):
        ACHORDRiskSim = M_achord.make_achord_risk_sim(M_gnf, M_fleet)
RitagsSim = None
if RITAGS_PATH is not None:
    M_ritags = _load(RITAGS_PATH, 'ritags')
    RitagsSim = M_ritags.make_ritags_sim(M_gnf, M_fleet)


def make_sim(kind: str, seed: int):
    random.seed(seed); np.random.seed(seed)
    if kind == 'fleet':
        sim = M_fleet.FleetSim()
    elif kind == 'gnf':
        sim = GNFSim(M_fleet)
    elif kind == 'gnf-risk':
        sim = M_gnf.GNFRiskSim(M_fleet)
    elif kind == 'cara2022':
        if CARAPaperSim is None:
            raise FileNotFoundError("Cara_paper.py not found next to run.py")
        sim = CARAPaperSim(M_fleet)
    elif kind == 'achord':
        if ACHORDSim is None:
            raise FileNotFoundError("ACHORD.py not found next to run.py")
        sim = ACHORDSim(M_fleet)
    elif kind == 'achord-risk':
        if ACHORDRiskSim is None:
            raise FileNotFoundError("ACHORD.py with make_achord_risk_sim not found")
        sim = ACHORDRiskSim(M_fleet)
    elif kind == 'ritags':
        if RitagsSim is None:
            raise FileNotFoundError("Ritags_sim.py not found next to run.py")
        sim = RitagsSim(M_fleet)
    elif kind == 'cara-base':
        sim = CARABaseSim(M_fleet)
    elif kind in ('cara', 'cara-dyn'):    # 'cara' kept as legacy alias
        sim = CARADynSim(M_fleet)
    else:
        raise ValueError(f"Unknown sim kind: {kind}")
    if not hasattr(sim, '_relay_ok_flood'):   sim._relay_ok_flood   = {}
    if not hasattr(sim, '_shadow_zone_type'): sim._shadow_zone_type = {}
    if not hasattr(sim, 'zone_tasks'):        sim.zone_tasks        = {}
    return sim


# ── Substrate-aware relay-coverage overlay ────────────────────────────────────
# The framework's draw_shadow_coverage() reads _active_relay_coverage_union(),
# which ONLY the Fleet sim (disk-radius model) implements. The GNF-substrate
# baselines (GNF, Greedy-Oracle, CARA-EL Base/Dyn, ACHORD) instead expose a
# boolean _relay_ok array (border-flood coverage). This helper renders THAT
# array so relay extension is visible on those sims too, matching the same
# green=covered / red=uncovered / white-border colour scheme.
def draw_substrate_coverage(screen, sim):
    relay_ok = getattr(sim, '_relay_ok', None)
    shd      = getattr(sim, 'radio_shadow', None)
    if relay_ok is None or shd is None:
        return
    covered = shd & relay_ok
    uncov   = shd & ~relay_ok
    overlay = pygame.Surface((GRID_W * CELL_SIZE, GRID_H * CELL_SIZE), pygame.SRCALPHA)
    # green covered
    xs, ys = np.where(covered)
    for x, y in zip(xs.tolist(), ys.tolist()):
        overlay.fill((0, 200, 80, 65),
                     (x*CELL_SIZE, y*CELL_SIZE, CELL_SIZE, CELL_SIZE))
    # red uncovered
    xs, ys = np.where(uncov)
    for x, y in zip(xs.tolist(), ys.tolist()):
        overlay.fill((220, 40, 40, 75),
                     (x*CELL_SIZE, y*CELL_SIZE, CELL_SIZE, CELL_SIZE))
    # white border: non-shadow cells adjacent to covered shadow (relay anchors)
    shd_i = shd.astype(np.uint8)
    nbr = ((np.roll(shd_i,1,0)|np.roll(shd_i,-1,0)|
            np.roll(shd_i,1,1)|np.roll(shd_i,-1,1)) > 0)
    border = (~shd) & nbr & relay_ok
    xs, ys = np.where(border)
    for x, y in zip(xs.tolist(), ys.tolist()):
        overlay.fill((255, 255, 255, 120),
                     (x*CELL_SIZE, y*CELL_SIZE, CELL_SIZE, CELL_SIZE))
    # Coverage-disk bound (R = RELAY_COVERAGE_RADIUS_CELLS): draw the actual
    # disk outline around every live coverage source, so bounded-disk vs
    # flood is verifiable BY EYE — any green outside a circle would be a
    # parity violation.
    R_cov = getattr(getattr(sim, 'M', None), 'RELAY_COVERAGE_RADIUS_CELLS', 30)
    border_m = getattr(sim, '_shadow_border_mask_cache', None)
    if border_m is not None:
        for r in sim.robots:
            if not getattr(r, 'active', False):
                continue
            rx, ry = r.pos
            if not shd[rx, ry] and border_m[rx, ry]:
                pygame.draw.circle(overlay, (0, 180, 220, 110),
                                   (rx*CELL_SIZE + CELL_SIZE//2,
                                    ry*CELL_SIZE + CELL_SIZE//2),
                                   int(R_cov*CELL_SIZE), 1)
    # ACHORD: dropped-radio nodes — gold cell + halo at the TRUE coverage
    # radius (was a small decorative pulse, misleading under disk parity).
    for rad in getattr(sim, 'radios', []):
        rx, ry = rad.pos
        overlay.fill((255, 215, 0, 235),
                     (rx*CELL_SIZE, ry*CELL_SIZE, CELL_SIZE, CELL_SIZE))
        cx = rx*CELL_SIZE + CELL_SIZE//2
        cy = ry*CELL_SIZE + CELL_SIZE//2
        pygame.draw.circle(overlay, (255, 215, 0, 170), (cx, cy),
                           int(R_cov*CELL_SIZE), 2)
        pygame.draw.circle(overlay, (120, 70, 10, 200), (cx, cy),
                           CELL_SIZE, 1)
    screen.blit(overlay, (0, 0))


# ── Selector screen ───────────────────────────────────────────────────────────
def selector_screen(screen, font_big, font_sm):
    options = [
        ('fleet',     '1  Fleet Sim',      'CBBA + potential game + relay coordination'),
        ('gnf',       '2  GNF Baseline',   'Greedy nearest-frontier, no coordination'),
        ('gnf-risk',  '3  GNF-Risk',       'GNF + belief-based risk-aware pathing'),
        ('cara2022',  '4  CARA-2022',      'Published CARA: PDR-triggered self-relaying'),
        ('cara-base', '5  CARA-EL-Base',   'Capability-aware MILP allocation'),
        ('cara-dyn',  '6  CARA-EL-Dyn',    'CARA-EL + exec layer (hold gate / eject)'),
        ('achord',    '7  ACHORD-insp',    'Droppable radios + predictive comms extension'),
        ('achord-risk','8  ACHORD-Risk',    'ACHORD + belief-based risk-aware pathing'),
        ('ritags',    '9  RITAGS-insp',    'Risk-tolerant trait-based coalitions (IROS 2023)'),
    ]
    if CARAPaperSim is None:
        options = [o for o in options if o[0] != 'cara2022']
    if ACHORDSim is None:
        options = [o for o in options if o[0] != 'achord']
    if ACHORDRiskSim is None:
        options = [o for o in options if o[0] != 'achord-risk']
    if RitagsSim is None:
        options = [o for o in options if o[0] != 'ritags']
    W = GRID_W * CELL_SIZE + SIDEBAR_WIDTH
    H = GRID_H * CELL_SIZE
    hover = None

    # Adaptive layout: fit however many options exist inside the window
    # (fixed 75px pitch used to push rows 9+ below the bottom edge, making
    # them unclickable).
    top = 120
    pitch = min(75, max(46, (H - top - 60) // max(1, len(options))))
    box_h = pitch - 8
    rects = {}
    for i, (kind, label, _) in enumerate(options):
        rects[kind] = pygame.Rect(W//2 - 200, top + i*pitch, 400, box_h)

    while True:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_1: return 'fleet'
                if ev.key == pygame.K_2: return 'gnf'
                if ev.key == pygame.K_3: return 'gnf-risk'
                if ev.key == pygame.K_4 and CARAPaperSim is not None: return 'cara2022'
                if ev.key == pygame.K_5: return 'cara-base'
                if ev.key == pygame.K_6: return 'cara-dyn'
                if ev.key == pygame.K_7 and ACHORDSim is not None: return 'achord'
                if ev.key == pygame.K_8 and ACHORDRiskSim is not None: return 'achord-risk'
                if ev.key == pygame.K_9 and RitagsSim is not None: return 'ritags'
                if ev.key == pygame.K_q: pygame.quit(); sys.exit()
            if ev.type == pygame.MOUSEMOTION:
                hover = next((k for k, rect in rects.items() if rect.collidepoint(ev.pos)), None)
            if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                clicked = next((k for k, rect in rects.items() if rect.collidepoint(ev.pos)), None)
                if clicked: return clicked

        screen.fill((18, 18, 28))
        title = font_big.render('Robot Fleet Simulator', True, (220, 220, 255))
        screen.blit(title, (W//2 - title.get_width()//2, 70))
        sub = font_sm.render('Choose a simulator to watch:', True, (160, 160, 200))
        screen.blit(sub, (W//2 - sub.get_width()//2, 112))

        for kind, label, desc in options:
            rect = rects[kind]
            col_bg  = (50, 80, 130) if hover == kind else (35, 55, 95)
            col_bdr = (100, 160, 255) if hover == kind else (70, 100, 170)
            # CARA gets a slightly different tint to signal it's the academic baseline
            if kind.startswith('cara'):
                col_bg  = (80, 55, 120) if hover == kind else (55, 35, 90)
                col_bdr = (180, 120, 255) if hover == kind else (120, 80, 180)
            pygame.draw.rect(screen, col_bg,  rect, border_radius=8)
            pygame.draw.rect(screen, col_bdr, rect, 2, border_radius=8)
            screen.blit(font_big.render(label, True, (230, 240, 255)), (rect.x+18, rect.y+4))
            if rect.h >= 44:
                screen.blit(font_sm.render(desc, True, (160, 175, 210)),
                            (rect.x+18, rect.y+rect.h-20))

        hint = font_sm.render('Press 1-9 / 0 or click to select   •   Q to quit',
                               True, (100, 100, 140))
        screen.blit(hint, (W//2 - hint.get_width()//2, H - 40))
        pygame.display.flip()
        pygame.time.Clock().tick(60)


# ── Sidebar renderer (adapts for each sim type) ───────────────────────────────
def draw_sidebar(screen, font, sim, kind, running, show_flags):
    show_map, show_surv, show_risk, show_plans, show_zones, show_shadow = show_flags
    BX = GRID_W * CELL_SIZE + 10
    W  = SIDEBAR_WIDTH - 10

    pygame.draw.rect(screen, (255,255,255),
                     (GRID_W*CELL_SIZE, 0, SIDEBAR_WIDTH, GRID_H*CELL_SIZE))

    # Buttons
    buttons = [
        ('start',  pygame.Rect(BX,  10, 80, 20)),
        ('map',    pygame.Rect(BX,  35, W,  20)),
        ('surv',   pygame.Rect(BX,  60, W,  20)),
        ('risk',   pygame.Rect(BX,  85, W,  20)),
        ('plans',  pygame.Rect(BX, 110, W,  20)),
        ('shadow', pygame.Rect(BX, 135, W,  20)),
    ]
    if kind == 'fleet':
        buttons.append(('zones', pygame.Rect(BX, 160, W, 20)))

    labels = {
        'start':  'Pause' if running else 'Start',
        'map':    'Hide Map'    if show_map    else 'Show Map',
        'surv':   'Hide Surv'   if show_surv   else 'Show Surv',
        'risk':   'Hide Risk'   if show_risk   else 'Show Risk',
        'plans':  'Hide Paths'  if show_plans  else 'Show Paths',
        'shadow': 'Hide Shadow' if show_shadow else 'Show Shadow',
        'zones':  'Hide Zones'  if show_zones  else 'Show Zones',
    }
    for name, rect in buttons:
        pygame.draw.rect(screen, (200,200,200), rect)
        screen.blit(font.render(labels[name], True, (0,0,0)), (rect.x+6, rect.y+4))

    # Stats
    y = 200
    union = sim.union_belief
    disc  = int(np.sum(union != T_UNKNOWN))
    pct   = disc / (GRID_W * GRID_H) * 100

    sim_names = {
        'fleet':     'Fleet Sim',
        'gnf':       'GNF Baseline',
        'gnf-risk':  'GNF-Risk',
        'cara2022':  'CARA-2022 (paper)',
        'cara-base': 'CARA-EL-Base',
        'cara':      'CARA-EL-Dyn',
        'cara-dyn':  'CARA-EL-Dyn',
        'achord':    'ACHORD-insp',
        'achord-risk': 'ACHORD-Risk',
        'ritags':    'RITAGS-insp',
    }
    hdr = font.render(sim_names.get(kind, kind), True, (40, 40, 160))
    screen.blit(hdr, (BX, y)); y += 28

    for line in [f"Step:      {sim.timestep}",
                 f"Coverage:  {pct:.1f}%",
                 f"Survivors: {len(sim.found)}/{len(sim.survivors)}"]:
        screen.blit(font.render(line, True, (0,0,0)), (BX, y)); y += 22

    # Fleet-specific: role breakdown
    if kind == 'fleet':
        from collections import Counter
        y += 8
        screen.blit(font.render("Roles:", True, (0,0,0)), (BX, y)); y += 20
        roles = Counter(r.role.name for r in sim.robots if r.active)
        for role_name, count in sorted(roles.items()):
            screen.blit(font.render(f"  {role_name}: {count}", True, (0,0,0)), (BX, y)); y += 18

    # CARA-EL-specific: MILP timing and relay assignment
    if kind.startswith('cara') and kind != 'cara2022':
        y += 8
        last_ms = getattr(sim, 'last_milp_time_ms', 0.0)
        times   = getattr(sim, 'milp_solve_times', [])
        avg_ms  = sum(times)/len(times) if times else 0.0
        col_ms  = (200, 50, 50) if last_ms > 500 else (0, 0, 0)
        screen.blit(font.render("MILP Allocation:", True, (80, 0, 140)), (BX, y)); y += 20
        screen.blit(font.render(f"  Last: {last_ms:.0f}ms", True, col_ms), (BX, y)); y += 18
        screen.blit(font.render(f"  Avg:  {avg_ms:.0f}ms", True, (0,0,0)), (BX, y)); y += 18
        screen.blit(font.render(f"  Solves: {len(times)}", True, (0,0,0)), (BX, y)); y += 18
        n_relays = sum(1 for r in sim.robots if r.active and getattr(r,'is_relay',False))
        n_explo  = sum(1 for r in sim.robots if r.active and not getattr(r,'is_relay',False))
        screen.blit(font.render(f"  Relays: {n_relays}  Explo: {n_explo}", True, (0,0,0)), (BX, y)); y += 18

    # All sims: battery
    y += 8
    screen.blit(font.render("Battery:", True, (0,0,0)), (BX, y)); y += 20
    from collections import defaultdict
    groups   = defaultdict(list)
    alive_c  = defaultdict(int)
    for r in sim.robots:
        t = robot_type(r.name)
        groups[t].append(r.battery)
        if r.active and r.battery > 0: alive_c[t] += 1
    for t in ("Legged","Drone","Boat","Rover"):
        v = groups[t]
        if not v: continue
        avg = sum(v)/len(v)
        screen.blit(font.render(f"  {t}: {avg:.0f} ({alive_c[t]}/{len(v)})",
                                True, (0,0,0)), (BX, y)); y += 18

    # Colour legend
    y += 8
    screen.blit(font.render("Types:", True, (0,0,0)), (BX, y)); y += 20
    for nm, clr in ROBOT_COLOUR.items():
        pygame.draw.rect(screen, clr, (BX, y, 14, 14))
        screen.blit(font.render(nm, True, (0,0,0)), (BX+20, y)); y += 18

    # ACHORD-specific: dropped-radio budget readout
    if kind.startswith('achord'):
        y += 8
        dropped = len(getattr(sim, 'radios', []))
        budget  = getattr(sim, 'radio_budget', 0)
        screen.blit(font.render("Droppable radios:", True, (110, 70, 40)), (BX, y)); y += 20
        screen.blit(font.render(f"  dropped: {dropped}", True, (0,0,0)), (BX+20, y)); y += 18
        screen.blit(font.render(f"  budget left: {budget}", True, (0,0,0)), (BX+20, y)); y += 18

    # GNF-substrate sims: hazard dose warning
    if kind != 'fleet':
        y += 8
        max_dose = max(
            (getattr(r, 'dose_T', 0) + getattr(r, 'dose_R', 0) for r in sim.robots),
            default=0.0
        )
        dose_col = (200, 50, 50) if max_dose > 5.0 else (0, 0, 0)
        screen.blit(font.render(f"Max dose: {max_dose:.1f}", True, dose_col), (BX, y)); y += 20
        note = font.render("(hazard-blind baseline)", True, (160, 80, 0))
        screen.blit(note, (BX, y)); y += 20

    # Survivor list
    y += 8
    screen.blit(font.render("Survivors:", True, (0,0,0)), (BX, y)); y += 20
    for i, pos in enumerate(sim.survivors, 1):
        found = pos in sim.found
        col   = (0, 160, 0) if found else (160, 0, 0)
        mark  = '✔' if found else '✖'
        screen.blit(font.render(f"{mark} S{i} {pos}", True, col), (BX, y)); y += 18
        if y > GRID_H * CELL_SIZE - 20: break

    return buttons


# ── Main GUI loop ─────────────────────────────────────────────────────────────
def gui_loop(kind: str, seed: int, max_steps: int):
    pygame.init()
    W_px = GRID_W * CELL_SIZE + SIDEBAR_WIDTH
    H_px = GRID_H * CELL_SIZE
    screen   = pygame.display.set_mode((W_px, H_px))
    font_big = pygame.font.SysFont(None, 32)
    font_sm  = pygame.font.SysFont(None, 22)
    font     = pygame.font.SysFont(None, 24)
    clock    = pygame.time.Clock()

    # Selector if kind not specified
    if kind is None:
        kind = selector_screen(screen, font_big, font_sm)

    captions = {
        'fleet':     'Fleet Sim',
        'gnf':       'GNF Baseline',
        'gnf-risk':  'GNF-Risk  [risk-aware pathing]',
        'cara2022':  'CARA-2022  [published mechanism: PDR-triggered self-relaying]',
        'cara-base': 'CARA-EL-Base  [MILP allocating]',
        'cara':      'CARA-EL-Dyn  [MILP + exec layer]',
        'cara-dyn':  'CARA-EL-Dyn  [MILP + exec layer]',
        'achord':    'ACHORD-insp  [droppable radios + predictive comms]',
        'achord-risk': 'ACHORD-Risk  [radios + risk-aware pathing]',
        'ritags':    'RITAGS-insp  [risk-tolerant trait coalitions]',
    }
    pygame.display.set_caption(f"Robot Simulator — {captions.get(kind,kind)}  (seed={seed})")

    sim = make_sim(kind, seed)

    # Pre-build static shadow surface (radio shadow never changes)
    if hasattr(sim, 'radio_shadow') and np.any(sim.radio_shadow):
        shadow_surf = build_shadow_surface(sim)
    else:
        shadow_surf = None

    running     = False
    show_map    = False
    show_surv   = False
    show_risk   = False
    show_plans  = False
    show_zones  = False
    show_shadow = True   # shadow on by default so it's visible immediately

    grid_surf       = None
    last_vk         = None
    last_union_tick = -1
    done            = False
    step_times      = []

    while True:
        # ── Events ──
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit(); sys.exit()

            if ev.type == pygame.KEYDOWN:
                if ev.key in (pygame.K_q, pygame.K_ESCAPE):
                    pygame.quit(); sys.exit()
                elif ev.key == pygame.K_SPACE:
                    running = not running
                elif ev.key == pygame.K_r:
                    # Reset with same seed
                    sim = make_sim(kind, seed)
                    if hasattr(sim, 'radio_shadow') and np.any(sim.radio_shadow):
                        shadow_surf = build_shadow_surface(sim)
                    grid_surf = None; done = False; running = False; step_times = []
                elif ev.key == pygame.K_m:
                    show_map    = not show_map;    grid_surf = None
                elif ev.key == pygame.K_s:
                    show_surv   = not show_surv;   grid_surf = None
                elif ev.key == pygame.K_h:
                    show_risk   = not show_risk;   grid_surf = None
                elif ev.key == pygame.K_p:
                    show_plans  = not show_plans
                elif ev.key == pygame.K_z:
                    show_zones  = not show_zones
                elif ev.key == pygame.K_w:
                    show_shadow = not show_shadow

            if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                show_flags = (show_map, show_surv, show_risk, show_plans,
                              show_zones, show_shadow)
                buttons = draw_sidebar(screen, font, sim, kind, running, show_flags)
                for name, rect in buttons:
                    if rect.collidepoint(ev.pos):
                        if   name == 'start':  running     = not running
                        elif name == 'map':    show_map    = not show_map;   grid_surf=None
                        elif name == 'surv':   show_surv   = not show_surv;  grid_surf=None
                        elif name == 'risk':   show_risk   = not show_risk;  grid_surf=None
                        elif name == 'plans':  show_plans  = not show_plans
                        elif name == 'zones':  show_zones  = not show_zones
                        elif name == 'shadow': show_shadow = not show_shadow

        # ── Step ──
        if running and not done:
            t0 = time.perf_counter()
            result = sim.step()
            step_times.append((time.perf_counter() - t0) * 1000)
            if not result or (max_steps > 0 and sim.timestep >= max_steps):
                running = False; done = True
                n = len(step_times); st = sorted(step_times)
                print(f"[DONE] sim={kind} seed={seed} t={sim.timestep} "
                      f"found={len(sim.found)}/{len(sim.survivors)} "
                      f"cov={np.mean(sim.union_belief!=T_UNKNOWN)*100:.1f}% "
                      f"P50={st[n//2]:.0f}ms P90={st[int(n*.9)]:.0f}ms avg={sum(st)/n:.0f}ms")

        # ── Draw grid ──
        union = sim.union_belief
        uT    = getattr(sim, 'union_T', None)
        uR    = getattr(sim, 'union_R', None)
        vk    = (show_map, show_surv, show_risk)
        if (grid_surf is None or vk != last_vk
                or sim.timestep != last_union_tick):
            grid_surf       = build_grid_surface(sim, show_map, show_surv,
                                                  show_risk, union, uT, uR)
            last_vk         = vk
            last_union_tick = sim.timestep

        screen.blit(grid_surf, (0, 0))

        # Shadow overlay. Fleet uses the disk-union renderer; the
        # GNF-substrate sims use the _relay_ok renderer (the disk method
        # they don't implement would otherwise paint everything 'uncovered').
        if show_shadow and shadow_surf is not None:
            screen.blit(shadow_surf, (0, 0))
            if kind == 'fleet':
                draw_shadow_coverage(screen, sim)
            else:
                draw_substrate_coverage(screen, sim)

        # Zone outlines (fleet only — GNF/RL have no zone tasks)
        if show_zones and kind == 'fleet':
            M_fleet.draw_zones(screen, sim)

        # Robot markers and paths
        draw_robots(screen, sim.robots, show_plans)

        # Sidebar
        show_flags = (show_map, show_surv, show_risk, show_plans,
                      show_zones, show_shadow)
        draw_sidebar(screen, font, sim, kind, running, show_flags)

        # Done banner
        if done:
            banner = font_big.render(
                f"DONE  t={sim.timestep}  "
                f"found={len(sim.found)}/{len(sim.survivors)}  "
                f"cov={np.mean(union!=T_UNKNOWN)*100:.1f}%",
                True, (255, 255, 80))
            bx = GRID_W * CELL_SIZE // 2 - banner.get_width() // 2
            pygame.draw.rect(screen, (30, 30, 30),
                             (bx-8, GRID_H*CELL_SIZE-42, banner.get_width()+16, 36))
            screen.blit(banner, (bx, GRID_H*CELL_SIZE - 38))

        # Keyboard hint strip at bottom of sidebar
        BX = GRID_W * CELL_SIZE + 10
        hints = [
            'SPACE: pause/resume',
            'R: reset  Q: quit',
            'M: map  S: survs',
            'H: hazard  P: paths',
            'W: shadow  Z: zones',
        ]
        y_hint = GRID_H * CELL_SIZE - len(hints) * 16 - 4
        for h in hints:
            hs = font_sm.render(h, True, (120, 120, 120))
            screen.blit(hs, (BX, y_hint)); y_hint += 16

        pygame.display.flip()
        clock.tick(M_fleet.FPS)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Robot Fleet Simulator Viewer')
    ap.add_argument('--sim',   choices=['fleet','gnf','gnf-risk','cara2022','cara-base','cara-dyn','cara','achord','achord-risk','ritags'],
                    default=None,
                    help='Simulator to run (omit for selector screen)')
    ap.add_argument('--seed',  type=int, default=0,   help='Random seed')
    ap.add_argument('--steps', type=int, default=0,
                    help='Stop after N steps (0 = run until done)')
    args = ap.parse_args()
    gui_loop(args.sim, args.seed, args.steps)