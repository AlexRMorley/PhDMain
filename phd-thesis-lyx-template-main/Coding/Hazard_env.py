"""
hazard_env.py — hazard-dense randomized environment patch for Framework M.

Applied to a loaded framework module (same pattern as the benches' other
patches; the framework file itself is never modified). Changes vs the plain
world:

  BUILDINGS   4-8 (was: exactly 4, one per quadrant). The original quadrant
              pass runs unchanged, then 0-4 extra buildings are stamped at
              random clear locations, so every world still has the baseline
              4 spread out plus a random surplus.

  TEMPERATURE 3-4 LARGE hot zones (sigma 12-18, peak 150-260) — genuinely
              lethal regions Legged/Drone (limit 120) must route around —
              plus 12-20 SMALL hotspots (sigma 2.5-5.5, peak 90-220): dense
              scatter that makes straight-line paths expensive everywhere.

  RADIATION   2-3 large zones + 10-16 small hotspots on the same pattern
              (limit 150; Boats/Rovers are immune by design, so a hazard-
              dense map is what makes capability heterogeneity matter).

  SOLVABILITY GUARDRAILS (so "hard" never silently becomes "impossible"):
    - No hotspot centre within 14 cells of the 9 spawn cluster centres —
      robots must never die on tick 1.
    - Large-zone centres kept (sigma + 4) away from building rectangles,
      small centres 6 cells away: buildings sit at most in a warm fringe
      (costly to enter, never sealed off from stair-capable robots).

All randomness flows through the global `random` module, so runs remain
fully reproducible under the benches' per-run seeding.
"""
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