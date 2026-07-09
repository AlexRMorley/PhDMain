"""
extreme_scenario.py  —  shared "Coastal Nuclear Industrial Disaster" scenario
=============================================================================
Single source of truth for the structured extreme-environment map so the GUI
runner (ExtremeEnvGUI.py) and the comparison benchmark (ExtremeComparisonBench.py)
stay in lock-step.  Importing module then calls:

    import extreme_scenario as scen
    info = scen.install(M, M_gnf, M_rl)          # patches the modules in place

After install(), EVERY sim that builds its world via ``M.GridWorld`` and spawns
via ``M.FleetSim._build_robots`` — i.e. Fleet, GNF, GreedyRL and CARA (they all
delegate through these) — generates the IDENTICAL structured world, hazards,
robot spawns, regional survivors, and rocky-hill radio-shadow exemption.  This
keeps the five-way comparison fair by construction.

Scenario contents (static; no dynamic events):
  - irregular south coastline + sea, a meandering river estuary, piers
  - one intact + one collapsed bridge across the estuary
  - concentric nuclear reactor (single winding entrance, radiation gradient)
  - chemical plant with an eastward plume + tank-farm / warehouse / riverbank /
    reactor-perimeter fires
  - residential complex (a touching terrace block + a spaced block)
  - warehouse district, docks
  - a rocky hill = disc of yellow stairs that is NOT radio-shadowed
  - survivors distributed 30% open / 40% industrial / 20% reactor-corridor /
    10% waterfront, placed only on survivable + reachable cells
"""

import math
import random
import numpy as np

# Default configuration (matches the GUI demo). Override via install(**kw).
DEFAULTS = dict(
    grid_w=200, grid_h=200, cell=5, zone_chunks=10,
    max_battery=5000, max_bundle=8,
    n_survivors=30, reveal_r=3,
    robots={"Legged": 8, "Drone": 13, "Boat": 6, "Rover": 10},
)


def layout_for(grid_w, grid_h):
    """Region anchors derived from the grid size (tuned for 200x200)."""
    W, H = grid_w, grid_h
    return dict(
        COAST_Y      = int(H * 0.82),
        ESTUARY_X    = int(W * 0.58),
        ESTUARY_W    = 7,
        ESTUARY_HEAD = int(H * 0.30),
        REACTOR_C    = (int(W * 0.35), int(H * 0.37)),
        REACTOR_HALF = 21,
        CHEM_C       = (int(W * 0.70), int(H * 0.60)),
        WARE_BBOX    = (int(W * 0.13), int(H * 0.52), int(W * 0.48), int(H * 0.74)),
        DOCK_Y       = (int(H * 0.75), int(H * 0.81)),
        HILL_C       = (int(W * 0.84), int(H * 0.40)),
        HILL_R       = 13,
    )


def install(M, M_gnf=None, M_rl=None, **kw):
    """Patch M / M_gnf / M_rl in place to build the structured scenario.
    Returns a dict with the resolved config + layout. Idempotent per process."""
    cfg = dict(DEFAULTS); cfg.update(kw)
    W, H = cfg['grid_w'], cfg['grid_h']
    NSURV = cfg['n_survivors']; REVEAL = cfg['reveal_r']; ROBOTS = dict(cfg['robots'])
    LAYOUT = layout_for(W, H)

    # ── module constants ───────────────────────────────────────────────────────
    M.GRID_W = W; M.GRID_H = H; M.CELL_SIZE = cfg['cell']
    M.ZONE_CHUNKS = cfg['zone_chunks']
    M.MAX_BATTERY = cfg['max_battery']; M.MAX_BUNDLE = cfg['max_bundle']

    T_UNKNOWN, T_FREE, T_OBS, T_STAIRS, T_WATER, T_BRIDGE = (
        M.T_UNKNOWN, M.T_FREE, M.T_OBS, M.T_STAIRS, M.T_WATER, M.T_BRIDGE)

    if getattr(M, '_extreme_scenario_installed', False):
        return dict(config=cfg, layout=LAYOUT)   # avoid double-wrapping in one process

    # ── GridWorld stamps ────────────────────────────────────────────────────────
    def _stamp_reactor(self, cx, cy, half):
        g = self.grid; Wg, Hg = self.w, self.h
        x0, y0, x1, y1 = cx - half, cy - half, cx + half, cy + half
        x0, y0 = max(1, x0), max(1, y0); x1, y1 = min(Wg - 2, x1), min(Hg - 2, y1)
        for x in range(x0 - 1, x1 + 2):
            for y in range(y0 - 1, y1 + 2):
                if 0 <= x < Wg and 0 <= y < Hg:
                    g[x][y]["t"] = T_FREE
        for x in range(x0, x1 + 1):
            for y in range(y0, y1 + 1):
                g[x][y]["t"] = T_STAIRS
        for x in range(x0, x1 + 1):
            g[x][y0]["t"] = T_OBS; g[x][y1]["t"] = T_OBS
        for y in range(y0, y1 + 1):
            g[x0][y]["t"] = T_OBS; g[x1][y]["t"] = T_OBS
        for dx in (-1, 0, 1):
            g[cx + dx][y1]["t"] = T_STAIRS
        h2 = max(4, half // 2)
        ix0, iy0, ix1, iy1 = cx - h2, cy - h2, cx + h2, cy + h2
        for x in range(ix0, ix1 + 1):
            g[x][iy0]["t"] = T_OBS; g[x][iy1]["t"] = T_OBS
        for y in range(iy0, iy1 + 1):
            g[ix0][y]["t"] = T_OBS; g[ix1][y]["t"] = T_OBS
        for dx in (-1, 0, 1):
            g[cx + dx][iy0]["t"] = T_STAIRS
        for x in range(ix0 + 1, ix1):
            for y in range(iy0 + 1, iy1):
                g[x][y]["t"] = T_STAIRS

    def _stamp_tank(self, cx, cy, r):
        g = self.grid; Wg, Hg = self.w, self.h
        for x in range(cx - r, cx + r + 1):
            for y in range(cy - r, cy + r + 1):
                if 0 <= x < Wg and 0 <= y < Hg and (x - cx) ** 2 + (y - cy) ** 2 <= r * r:
                    g[x][y]["t"] = T_OBS

    def _stamp_complex(self, x0, y0, ncols, nrows, bw, bh, street, pad=1):
        centres = []
        for i in range(ncols):
            for j in range(nrows):
                hx = x0 + i * (bw + street); hy = y0 + j * (bh + street)
                if hx + bw >= self.w - 2 or hy + bh >= self.h - 2:
                    continue
                if self._rect_clear(hx, hy, bw, bh, pad=pad):
                    self._stamp_house(hx, hy, bw, bh)
                    centres.append((hx + bw // 2, hy + bh // 2))
        return centres

    def _stamp_hill(self, cx, cy, r):
        g = self.grid; Wg, Hg = self.w, self.h
        cells = []
        for x in range(cx - r, cx + r + 1):
            for y in range(cy - r, cy + r + 1):
                if 0 <= x < Wg and 0 <= y < Hg and (x - cx) ** 2 + (y - cy) ** 2 <= r * r:
                    if g[x][y]["t"] in (T_FREE, T_OBS, T_UNKNOWN):
                        g[x][y]["t"] = T_STAIRS; cells.append((x, y))
        return cells

    M.GridWorld._stamp_reactor = _stamp_reactor
    M.GridWorld._stamp_tank    = _stamp_tank
    M.GridWorld._stamp_complex = _stamp_complex
    M.GridWorld._stamp_hill    = _stamp_hill

    # ── world generation ─────────────────────────────────────────────────────────
    def _extreme_generate(self):
        g = self.grid; Wg, Hg = self.w, self.h; L = LAYOUT
        for x in range(Wg):
            for y in range(Hg):
                if random.random() < 0.03:
                    g[x][y]["t"] = T_OBS
        coast = {}
        for x in range(Wg):
            cy = int(L['COAST_Y'] + 5 * math.sin(x * 0.045) + random.randint(-2, 2))
            coast[x] = cy
            for y in range(cy, Hg):
                g[x][y]["t"] = T_WATER
        ex = L['ESTUARY_X']; ew = L['ESTUARY_W'] // 2
        for y in range(L['ESTUARY_HEAD'], Hg):
            ex += random.choice([-1, 0, 0, 0, 1]); ex = max(ew + 2, min(Wg - ew - 2, ex))
            for dx in range(-ew, ew + 1):
                g[ex + dx][y]["t"] = T_WATER
        self._fill_river_islands()

        def _bridge(by, collapsed):
            row = [x for x in range(Wg) if g[x][by]["t"] == T_WATER]
            if not row:
                return
            a, b = min(row), max(row)
            gap0 = a + (b - a) // 3; gap1 = a + 2 * (b - a) // 3
            for x in range(a - 2, b + 3):
                if not (0 <= x < Wg):
                    continue
                if collapsed and gap0 <= x <= gap1:
                    continue
                for dy in range(-2, 3):
                    yy = by + dy
                    if 0 <= yy < Hg and g[x][yy]["t"] == T_WATER:
                        g[x][yy]["t"] = T_BRIDGE
        _bridge(int(Hg * 0.55), collapsed=False)
        _bridge(int(Hg * 0.70), collapsed=True)

        rcx, rcy = L['REACTOR_C']
        self._stamp_reactor(rcx, rcy, L['REACTOR_HALF'])

        ccx, ccy = L['CHEM_C']
        for hx, hy, hw, hh in [(ccx - 20, ccy - 8, 12, 12),
                               (ccx - 4,  ccy + 4, 12, 11),
                               (ccx + 12, ccy - 6, 11, 12)]:
            if self._rect_clear(hx, hy, hw, hh, pad=2):
                self._stamp_house(hx, hy, hw, hh)
        tank_xy = [(ccx - 24, ccy + 10), (ccx + 2, ccy - 14), (ccx + 26, ccy + 8)]
        for tx, ty in tank_xy:
            if 0 <= tx < Wg and 0 <= ty < Hg:
                self._stamp_tank(tx, ty, 3)

        wx0, wy0, wx1, wy1 = L['WARE_BBOX']; ware_centres = []
        for _ in range(5):
            hw = random.randint(12, 18); hh = random.randint(10, 15)
            for _att in range(150):
                hx = random.randint(wx0, max(wx0, wx1 - hw))
                hy = random.randint(wy0, max(wy0, wy1 - hh))
                if self._rect_clear(hx, hy, hw, hh, pad=3):
                    self._stamp_house(hx, hy, hw, hh)
                    ware_centres.append((hx + hw // 2, hy + hh // 2)); break

        # residential: a touching terrace + a spaced block
        self._stamp_complex(int(Wg * 0.06), int(Hg * 0.05), 5, 2, 13, 12, 0, pad=0)
        self._stamp_complex(int(Wg * 0.56), int(Hg * 0.05), 4, 2, 13, 12, 5)

        hcx, hcy = L['HILL_C']
        hill_cells = self._stamp_hill(hcx, hcy, L['HILL_R'])

        dy0, dy1 = L['DOCK_Y']
        for dxc in (int(Wg * 0.20), int(Wg * 0.42), int(Wg * 0.78)):
            hx, hy, hw, hh = dxc - 5, dy0 - 2, 10, 8
            if self._rect_clear(hx, hy, hw, hh, pad=2):
                self._stamp_house(hx, hy, hw, hh)
            for py in range(dy1, min(Hg, dy1 + 14)):
                for dx in (-1, 0):
                    x = dxc + dx
                    if 0 <= x < Wg and g[x][py]["t"] == T_WATER:
                        g[x][py]["t"] = T_BRIDGE

        self._fix_land_pinches(min_corridor=4)

        self.reactor_center = (rcx, rcy)
        self.reactor_half   = L['REACTOR_HALF']
        self.reactor_inner  = max(4, L['REACTOR_HALF'] // 2)
        self.chem_center    = (ccx, ccy)
        self.chem_wind      = (1, 0)
        self.coast_map      = coast
        self.dock_yband     = (dy0, dy1)
        self.ware_bbox      = L['WARE_BBOX']
        self.hill_cells     = hill_cells
        self.hill_center    = (hcx, hcy)
        self.hill_r         = L['HILL_R']
        fires = [(tx, ty, 95.0, 4.5) for (tx, ty) in tank_xy]
        if ware_centres:
            wcx, wcy = ware_centres[0]
            fires.append((wcx, wcy, 80.0, 5.0))
        ex0 = L['ESTUARY_X']; rh = L['REACTOR_HALF']
        fires += [
            (ex0 + 10, int(Hg * 0.42), 80.0, 5.0),
            (ex0 - 11, int(Hg * 0.64), 75.0, 5.0),
            (rcx,          rcy + rh + 7, 82.0, 5.0),
            (rcx + rh + 6, rcy - 10,     70.0, 4.5),
        ]
        self.fire_sources = fires

    M.GridWorld._generate = _extreme_generate

    # ── hazard fields ─────────────────────────────────────────────────────────
    def _extreme_radiation(self):
        rcx, rcy = self.reactor_center
        AMP, SIG, BG = 430.0, 6.5, 2.0
        leak = [(rcx + k * 7, rcy + k * 5, 110.0 * math.exp(-0.28 * k), 7.0 + 0.8 * k)
                for k in range(1, 6)]
        for x in range(self.w):
            for y in range(self.h):
                d2 = (x - rcx) ** 2 + (y - rcy) ** 2
                v = BG + AMP * math.exp(-d2 / (2 * SIG * SIG))
                for lx, ly, la, ls in leak:
                    v += la * math.exp(-((x - lx) ** 2 + (y - ly) ** 2) / (2 * ls * ls))
                if self.grid[x][y]["t"] in (T_WATER, T_BRIDGE):
                    v = 0.0
                self.grid[x][y]["rad"] = v

    def _extreme_temperature(self):
        ccx, ccy = self.chem_center; wx, wy = self.chem_wind
        AMBIENT = 15.0
        puffs = [((ccx + 8 + int(k * 5 * wx), ccy + int(k * 5 * wy) + random.randint(-1, 1)),
                  6.0 + 0.6 * k, 175.0 * math.exp(-0.16 * k)) for k in range(14)]
        fires = getattr(self, 'fire_sources', [])
        for x in range(self.w):
            for y in range(self.h):
                v = AMBIENT
                for (mx, my), s, a in puffs:
                    v += a * math.exp(-((x - mx) ** 2 + (y - my) ** 2) / (2 * s * s))
                for fx, fy, fa, fs in fires:
                    v += fa * math.exp(-((x - fx) ** 2 + (y - fy) ** 2) / (2 * fs * fs))
                if self.grid[x][y]["t"] in (T_WATER, T_BRIDGE):
                    v = 5.0
                self.grid[x][y]["temp"] = v

    M.GridWorld._init_radiation   = _extreme_radiation
    M.GridWorld._init_temperature = _extreme_temperature

    # ── radio-shadow: exempt the rocky hill (applies to ALL sims via FleetSim) ──
    _orig_build_shadow = M.FleetSim._build_radio_shadow

    def _patched_build_radio_shadow(self):
        wd = self.world
        hill = list(getattr(wd, 'hill_cells', []) or [])
        for (x, y) in hill:
            wd.grid[x][y]["t"] = T_FREE
        _orig_build_shadow(self)
        if hill:
            for (x, y) in hill:
                wd.grid[x][y]["t"] = T_STAIRS
                self.radio_shadow[x, y] = False
            rs = self.radio_shadow
            self._shadow_cells_arr = np.argwhere(rs)
            rs_i = rs.astype(np.uint8)
            nbr = (np.roll(rs_i, 1, 0) | np.roll(rs_i, -1, 0) |
                   np.roll(rs_i, 1, 1) | np.roll(rs_i, -1, 1)).astype(bool)
            nbr[0, :] &= rs[1, :]; nbr[-1, :] &= rs[-2, :]
            nbr[:, 0] &= rs[:, 1]; nbr[:, -1] &= rs[:, -2]
            self._shadow_border_mask_cache = (~rs) & nbr
            self._shadow_border_cells_arr  = np.argwhere(self._shadow_border_mask_cache)

    M.FleetSim._build_radio_shadow = _patched_build_radio_shadow

    # ── reveal radius ─────────────────────────────────────────────────────────
    def _wrap_reveal(cls):
        _orig = cls.__init__
        def _init(self, *a, **k):
            _orig(self, *a, **k); self.terrain_R = REVEAL
        cls.__init__ = _init
    _wrap_reveal(M.Robot)
    if M_gnf is not None and hasattr(M_gnf, 'GNFRobot'):
        _wrap_reveal(M_gnf.GNFRobot)
    if M_rl is not None and hasattr(M_rl, 'GreedyRLRobot'):
        _wrap_reveal(M_rl.GreedyRLRobot)

    # ── robot spawn (land staging areas; all baselines delegate here) ───────────
    def _patched_build_robots(self):
        templates = [
            ("Legged", {M.Capability.LAND, M.Capability.STAIRS}, np.array([10., 10.]), (M.TEMP_LIMIT, M.RAD_LIMIT)),
            ("Drone",  {M.Capability.AIR},   np.array([10., 10.]), (M.TEMP_LIMIT, M.RAD_LIMIT)),
            ("Boat",   {M.Capability.WATER}, np.array([0.,  0.]),  (9999., 9999.)),
            ("Rover",  {M.Capability.LAND},  np.array([-2., -2.]), (9999., 9999.)),
        ]
        tpl = {n: (n, c, w, l) for n, c, w, l in templates}
        spawn = []
        for t, n in ROBOTS.items():
            spawn += [t] * n
        random.shuffle(spawn)
        Wg, Hg = M.GRID_W, M.GRID_H; g = self.world.grid
        clusters = [
            (int(Wg*0.18), int(Hg*0.12)), (int(Wg*0.50), int(Hg*0.12)), (int(Wg*0.82), int(Hg*0.12)),
            (int(Wg*0.20), int(Hg*0.45)), (int(Wg*0.80), int(Hg*0.45)),
            (int(Wg*0.30), int(Hg*0.62)), (int(Wg*0.78), int(Hg*0.70)),
        ]
        water_cells = [(x, y) for x in range(Wg) for y in range(Hg) if g[x][y]["t"] == M.T_WATER]

        def nearest_land(cx, cy, caps):
            best = None; bestd = 1e9
            for x in range(Wg):
                for y in range(Hg):
                    tt = g[x][y]["t"]
                    if (tt == M.T_FREE or (tt == M.T_STAIRS and M.Capability.STAIRS in caps)) \
                            and not self.radio_shadow[x, y]:
                        d = (x - cx) ** 2 + (y - cy) ** 2
                        if d < bestd:
                            bestd = d; best = (x, y)
            return best or (cx, cy)

        self.robots = []
        for i, tname in enumerate(spawn):
            _, caps, weights, (tlim, rlim) = tpl[tname]
            name = f"{tname}{i}"; center = clusters[i % len(clusters)]
            if tname == "Boat" and water_cells:
                ns = [c for c in water_cells if not self.radio_shadow[c[0], c[1]]]
                sx, sy = random.choice(ns or water_cells)
            else:
                sx, sy = center; placed = False
                for _ in range(60):
                    cx = max(1, min(Wg - 2, center[0] + random.randint(-12, 12)))
                    cy = max(1, min(Hg - 2, center[1] + random.randint(-12, 12)))
                    tt = g[cx][cy]["t"]
                    if (tt == M.T_FREE or (tt == M.T_STAIRS and M.Capability.STAIRS in caps)) \
                            and not self.radio_shadow[cx, cy]:
                        sx, sy = cx, cy; placed = True; break
                if not placed:
                    sx, sy = nearest_land(center[0], center[1], caps)
            self.robots.append(M.Robot(name, sx, sy, caps, self.world, self, weights, tlim, rlim))

    M.FleetSim._build_robots = _patched_build_robots

    # ── survivors (regional, survivable, reachable, hill excluded) ──────────────
    def _patched_build_survivors(self):
        Wg, Hg = M.GRID_W, M.GRID_H
        g = self.world.grid; wd = self.world
        rcx, rcy = wd.reactor_center
        half = wd.reactor_half; inner = wd.reactor_inner
        TL, RL = M.TEMP_LIMIT, M.RAD_LIMIT
        hill = set(getattr(wd, 'hill_cells', []) or [])

        def safe(x, y):
            return g[x][y]["temp"] < TL * 0.9 and g[x][y]["rad"] < RL * 0.9

        reactor_corridor, industrial, waterfront, open_cells = [], [], [], []
        sea_adj = set()
        for x in range(Wg):
            for y in range(Hg):
                if g[x][y]["t"] == T_WATER:
                    for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
                        nx, ny = x+dx, y+dy
                        if 0 <= nx < Wg and 0 <= ny < Hg and g[nx][ny]["t"] in (T_FREE, T_BRIDGE):
                            sea_adj.add((nx, ny))
        for x in range(Wg):
            for y in range(Hg):
                if (x, y) in hill:
                    continue
                t = g[x][y]["t"]; d = math.hypot(x - rcx, y - rcy)
                in_reactor = (abs(x - rcx) <= half and abs(y - rcy) <= half)
                if t == T_STAIRS and in_reactor:
                    if inner + 1 < d < half - 1 and safe(x, y):
                        reactor_corridor.append((x, y))
                elif t == T_STAIRS:
                    if safe(x, y):
                        industrial.append((x, y))
                elif t in (T_FREE, T_BRIDGE):
                    if not safe(x, y):
                        continue
                    if (x, y) in sea_adj or wd.dock_yband[0] <= y <= wd.dock_yband[1]:
                        waterfront.append((x, y))
                    else:
                        open_cells.append((x, y))

        from collections import deque as _dq
        PASS = {T_FREE, T_STAIRS, T_BRIDGE}
        vis = np.zeros((Wg, Hg), dtype=bool); best = set()
        for sx0 in range(Wg):
            for sy0 in range(Hg):
                if g[sx0][sy0]["t"] in PASS and not vis[sx0, sy0]:
                    q = _dq([(sx0, sy0)]); vis[sx0, sy0] = True; comp = [(sx0, sy0)]
                    while q:
                        a, b = q.popleft()
                        for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
                            nx, ny = a+dx, b+dy
                            if 0 <= nx < Wg and 0 <= ny < Hg and not vis[nx, ny] and g[nx][ny]["t"] in PASS:
                                vis[nx, ny] = True; q.append((nx, ny)); comp.append((nx, ny))
                    if len(comp) > len(best): best = set(comp)
        reactor_corridor = [c for c in reactor_corridor if c in best]
        industrial       = [c for c in industrial       if c in best]
        waterfront       = [c for c in waterfront       if c in best]
        open_cells       = [c for c in open_cells       if c in best]

        n = NSURV
        want = {'open': round(n*0.30), 'industrial': round(n*0.40), 'reactor': round(n*0.20),
                'waterfront': n - round(n*0.30) - round(n*0.40) - round(n*0.20)}

        def take(pool, k):
            random.shuffle(pool); return pool[:max(0, k)]

        chosen = []
        chosen += take(reactor_corridor, want['reactor'])
        chosen += take(industrial,       want['industrial'])
        chosen += take(waterfront,       want['waterfront'])
        k_open = want['open']; gk = max(1, round(k_open ** 0.5) + 1)
        cw, ch = Wg // gk, Hg // gk; picked = []; buckets = {}
        for (x, y) in open_cells:
            buckets.setdefault((x // cw, y // ch), []).append((x, y))
        for key in list(buckets.keys()):
            if len(picked) >= k_open: break
            picked.append(random.choice(buckets[key]))
        if len(picked) < k_open:
            picked += take(open_cells, k_open - len(picked))
        chosen += picked

        seen = set(); uniq = []
        for c in chosen:
            if c not in seen: seen.add(c); uniq.append(c)
        backup = industrial + open_cells + waterfront + reactor_corridor
        random.shuffle(backup)
        for c in backup:
            if len(uniq) >= n: break
            if c not in seen: seen.add(c); uniq.append(c)
        self.survivors = uniq[:n]

    M.FleetSim._build_survivors = _patched_build_survivors
    if M_gnf is not None and hasattr(M_gnf, 'GNFSim'):
        M_gnf.GNFSim._build_survivors = _patched_build_survivors

    M._extreme_scenario_installed = True
    return dict(config=cfg, layout=LAYOUT)