"""ACHORD-inspired baseline: droppable radios + comms-aware exploration.

    Saboia, M.; Clark, L.; Thangavelu, V.; et al. (JPL CoSTAR)
    "ACHORD: Communication-Aware Multi-Robot Coordination with
     Intermittent Connectivity." IEEE RA-L / IROS, 2022.
    arXiv:2206.02245

ACHORD is a multi-layer SYSTEMS architecture from the DARPA Subterranean
Challenge; a full reimplementation (bandwidth stratification, flow control,
low-layer radio metrics) is neither possible nor meaningful in a grid-world
simulator. This module implements its COORDINATION CORE, honestly labelled
"ACHORD-inspired":

  1. IRM-lite (Information RoadMap). Each robot's traversed positions are
     tagged with an estimated signal quality (comms checkpoints); the base
     station and dropped radios are network nodes. Mirrors the paper's
     incrementally built network-topology map.
  2. DROPPABLE RADIOS — PREDICTIVE. The fleet carries a budget of
     RADIO_BUDGET static radio nodes. ACHORD's comms modelling is
     predictive: the network is extended BEFORE connectivity is lost, not
     recovered after. Grid analogue: when a robot stands on a shadow-border
     cell whose interior beyond is not yet relay-covered (it is about to
     enter a blackout region), and no radio sits within MIN_RADIO_SPACING,
     it deploys a radio AT that border cell — a breadcrumb at the cave
     mouth — and continues exploring. (A reactive trigger-then-backtrack
     design was tried first and validated as broken: by the time the
     windowed estimate degrades the robot is deep in uncovered shadow, which
     the substrate's shadow gate makes un-pathable for air/stair robots, so
     drops never complete.) This is ACHORD's defining difference from
     robot-as-relay schemes (CARA, our framework): the network is extended
     by hardware, not by sacrificing an explorer.
  3. CONNECTIVITY RESTORATION. A robot continuously blacked out (uncovered
     shadow) for RESTORE_PATIENCE ticks walks back to its last strong-signal
     position to re-link, then resumes — the paper's restore-connectivity
     behaviour, applied conservatively so it does not fight exploration
     (ACHORD tolerates intermittency by design).

DECLARED ADAPTATIONS / ASYMMETRIES (for the information-access audit):
  * CAPABILITY ASYMMETRY IN ACHORD'S FAVOUR: static radios cost nothing and
    never die; Framework M/N's relays are working robots. This is
    deliberate — a gold-standard comparator should hold its real-world
    hardware advantage.
  * Signal quality is the same declared proxy used by CARA-2022 (1.0 open
    air; 0.9 covered shadow; exp(-0.12*depth) in uncovered shadow), standing
    in for the paper's SNR/loss-rate telemetry.
  * Radios only extend coverage from shadow-border cells (substrate rule for
    all models); the drop-site selector therefore targets border cells.
  * Dropped radios do NOT sense survivors — they are radios. (The dropping
    robot stood on that exact cell when deploying, so no information is
    gained or lost by this rule.)
  * Bandwidth prioritisation / data stratification are out of scope: the
    substrate's comms are binary on coverage.

Everything else (frontier exploration, movement, battery, graded hazard
dose) is the shared GNF substrate, identical to every other baseline.

TWO VARIANTS are exported:
  make_achord_sim       — ACHORD-insp: hazard-BLIND pathing (the raw GNF
                          substrate, matching GNF / Greedy-Oracle / CARA-2022).
  make_achord_risk_sim  — ACHORD-Risk: adds belief-based RISK-AWARE pathing,
                          the same planner treatment CARA-EL has (max-pooled
                          hazard risk from the robot's OWN temp/rad beliefs,
                          its own graded limits, soft-cost hazard avoidance).
                          Belief-only: nothing oracle. This makes the
                          capability ladder explicit in the results table:
                          substrate-blind ACHORD vs risk-aware ACHORD vs
                          risk-aware CARA-EL.
"""
import numpy as np
from collections import deque

RADIO_BUDGET     = 10    # fleet-wide droppable radio nodes
SLIDING_WINDOW   = 10    # ticks per link-estimate window
DROP_LOWER       = 0.8   # deploy a radio when windowed estimate < this
GOOD_SIGNAL      = 0.9   # a trail cell qualifies as a drop/restore site at >= this
MIN_RADIO_SPACING = 12   # Chebyshev cells between radios (no wasteful stacking)
TRAIL_LEN        = 40    # per-robot trail memory (positions + signal)
RESTORE_PATIENCE = 60    # continuous blackout ticks before walking back out
RESTORE_TIMEOUT  = 40    # give up on an unreachable restore target after this
ATTEN_ALPHA      = 0.12  # uncovered-shadow signal decay per penetration cell


class _RadioStub:
    """Static radio node. Participates in the substrate's border flood-fill
    coverage exactly like a stationed robot; inert for everything else."""
    __slots__ = ('name', 'pos', 'active', 'caps_mask', 'dose_T', 'dose_R',
                 'battery', 'temp_limit', 'rad_limit', 'hazard_killed',
                 'death_reason')

    def __init__(self, idx, pos):
        self.name = f"Radio{idx}"
        self.pos = pos
        self.active = True
        self.caps_mask = 0
        self.dose_T = 0.0; self.dose_R = 0.0
        self.battery = float('inf')
        self.temp_limit = 9999.0; self.rad_limit = 9999.0
        self.hazard_killed = False
        self.death_reason = None

    def tick(self, *a, **k):     # radios do nothing per tick
        return


def make_achord_sim(gnf_module, M_module):
    """Return an AchordSim class (same factory pattern as the other baselines)."""
    GNFSim   = gnf_module.GNFSim
    GNFRobot = gnf_module.GNFRobot
    M        = M_module

    class ACHORDBot(GNFRobot):
        __slots__ = ('net_target', 'net_action', 'blackout_ticks', 'net_set_tick')

        def _nearest_frontier(self):
            # A pending network action (walk to drop site / restore point)
            # takes priority; afterwards the robot resumes normal exploration.
            if self.net_target is not None and self.pos != self.net_target:
                return self.net_target
            return super()._nearest_frontier()

    class AchordSim(GNFSim):

        def _build_robots(self):
            super()._build_robots()
            wrapped = []
            for r in self.robots:
                b = ACHORDBot(r.name, r.pos[0], r.pos[1], r.caps, r.caps_mask,
                              self.world, self, r.temp_limit, r.rad_limit)
                b.net_target = None
                b.net_action = None       # 'restore' (drops are instantaneous)
                b.net_set_tick = 0
                b.blackout_ticks = 0
                wrapped.append(b)
            self.robots = wrapped
            self._sig_win  = {r.name: deque(maxlen=SLIDING_WINDOW) for r in self.robots}
            self._trail    = {r.name: deque(maxlen=TRAIL_LEN) for r in self.robots}
            self.radios    = []          # _RadioStub list (the IRM's dropped nodes)
            self.radio_budget = RADIO_BUDGET
            self.radios_dropped = []     # (tick, robot, pos) — for reporting
            self._shadow_depth = self._build_shadow_depth()

        # ── IRM-lite signal machinery (same proxy family as CARA-2022) ────────
        def _build_shadow_depth(self):
            W, H = M.GRID_W, M.GRID_H
            shadow = self.radio_shadow
            depth = np.full((W, H), 32767, dtype=np.int32)
            q = deque()
            for x in range(W):
                for y in range(H):
                    if shadow[x, y]:
                        for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
                            nx, ny = x+dx, y+dy
                            if 0 <= nx < W and 0 <= ny < H and not shadow[nx, ny]:
                                depth[x, y] = 1
                                q.append((x, y))
                                break
            while q:
                cx, cy = q.popleft()
                for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
                    nx, ny = cx+dx, cy+dy
                    if (0 <= nx < W and 0 <= ny < H and shadow[nx, ny]
                            and depth[nx, ny] > depth[cx, cy] + 1):
                        depth[nx, ny] = depth[cx, cy] + 1
                        q.append((nx, ny))
            return depth

        def _signal(self, pos):
            x, y = pos
            if not self.radio_shadow[x, y]:
                return 1.0
            if self._relay_ok[x, y]:
                return 0.9
            return float(np.exp(-ATTEN_ALPHA * float(self._shadow_depth[x, y])))

        def _near_existing_radio(self, cell):
            for rad in self.radios:
                if max(abs(rad.pos[0]-cell[0]), abs(rad.pos[1]-cell[1])) < MIN_RADIO_SPACING:
                    return True
            return False

        def _best_trail_site(self, name, need_border):
            """Nearest recent trail position with strong signal; optionally
            restricted to shadow-border cells (radio drop sites must bridge)."""
            border = self._shadow_border_mask_cache
            shadow = self.radio_shadow
            me = next(r for r in self.robots if r.name == name)
            best, best_d = None, 1 << 30
            for (px, py), sig in self._trail[name]:
                if sig < GOOD_SIGNAL:
                    continue
                if need_border and (shadow[px, py] or not border[px, py]):
                    continue
                d = abs(px - me.pos[0]) + abs(py - me.pos[1])
                if d < best_d:
                    best, best_d = (px, py), d
            return best

        # ── ACHORD orchestration ──────────────────────────────────────────────
        def _uncovered_interior_beyond(self, x, y):
            """True if a shadow cell adjacent to (x,y) is not relay-covered —
            i.e. stepping in from this border cell would enter a blackout."""
            for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
                nx, ny = x+dx, y+dy
                if 0 <= nx < M.GRID_W and 0 <= ny < M.GRID_H \
                        and self.radio_shadow[nx, ny] and not self._relay_ok[nx, ny]:
                    return True
            return False

        def _achord_bookkeeping(self):
            border = self._shadow_border_mask_cache
            for r in self.robots:
                if not r.active:
                    continue
                sig = self._signal(r.pos)
                self._sig_win[r.name].append(sig)
                self._trail[r.name].append((r.pos, sig))
                x, y = r.pos
                # blackout accounting for connectivity restoration
                if self.radio_shadow[x, y] and not self._relay_ok[x, y]:
                    r.blackout_ticks += 1
                else:
                    r.blackout_ticks = 0

                # 1) PREDICTIVE network extension: standing at the mouth of an
                #    uncovered blackout region -> breadcrumb here, keep going.
                if (self.radio_budget > 0
                        and not self.radio_shadow[x, y] and border[x, y]
                        and self._uncovered_interior_beyond(x, y)
                        and not self._near_existing_radio(r.pos)):
                    self._drop_radio(r)

                # restoration arrival / timeout
                if r.net_target is not None:
                    if r.pos == r.net_target or \
                            (self.timestep - r.net_set_tick) > RESTORE_TIMEOUT:
                        r.net_target = None
                        r.net_action = None
                        r.goal = None; r.path = []
                    else:
                        continue                 # still en route

                # 2) connectivity restoration: long blackout -> walk back out
                if r.blackout_ticks > RESTORE_PATIENCE:
                    site = self._best_trail_site(r.name, need_border=False)
                    if site is not None and site != r.pos:
                        r.net_target = site
                        r.net_action = 'restore'
                        r.net_set_tick = self.timestep
                        r.goal = None; r.path = []

        # Radio coverage integration: _relay_ok is a property whose setter
        # ORs the radios' persistent flood coverage into every assignment the
        # substrate makes. The substrate recomputes robot coverage inside
        # step() and robots tick against it immediately afterwards — with the
        # setter, radio coverage is present at that exact moment, with no
        # roster pollution and no dependence on the substrate's internals.
        @property
        def _relay_ok(self):
            return self.__dict__.get('_relay_ok_val')

        @_relay_ok.setter
        def _relay_ok(self, arr):
            rc = self.__dict__.get('_radio_cov')
            if rc is not None and arr is not None:
                arr = arr | rc
            self.__dict__['_relay_ok_val'] = arr

        def _flood_from_border(self, cell, into):
            """Stamp a static radio's coverage: shadow cells within Euclidean
            RELAY_COVERAGE_RADIUS_CELLS of the radio — the SAME bounded-disk
            model as the Fleet's relays (was an unbounded flood; renamed
            semantics kept for call-site stability)."""
            R_cov = getattr(M, 'RELAY_COVERAGE_RADIUS_CELLS', 30)
            rr = int(R_cov)
            x, y = cell
            ax = np.arange(-rr, rr + 1)
            tpl = (ax[:, None] ** 2 + ax[None, :] ** 2) <= R_cov * R_cov
            x0, x1 = max(0, x - rr), min(M.GRID_W, x + rr + 1)
            y0, y1 = max(0, y - rr), min(M.GRID_H, y + rr + 1)
            tx0 = x0 - (x - rr); ty0 = y0 - (y - rr)
            win = tpl[tx0:tx0 + (x1 - x0), ty0:ty0 + (y1 - y0)]
            into[x0:x1, y0:y1] |= win & self.radio_shadow[x0:x1, y0:y1]

        def _drop_radio(self, robot):
            stub = _RadioStub(len(self.radios), robot.pos)
            self.radios.append(stub)
            self.radio_budget -= 1
            self.radios_dropped.append((self.timestep, robot.name, robot.pos))
            if self.__dict__.get('_radio_cov') is None:
                self.__dict__['_radio_cov'] = np.zeros((M.GRID_W, M.GRID_H), dtype=bool)
            self._flood_from_border(robot.pos, self.__dict__['_radio_cov'])
            # refresh the live mask immediately (setter ORs the new coverage)
            self._relay_ok = self.__dict__['_relay_ok_val']

        def step(self) -> bool:
            self._achord_bookkeeping()
            return super().step()

    return AchordSim


def make_achord_risk_sim(gnf_module, M_module):
    """ACHORD-Risk: identical droppable-radio coordination, plus risk-aware
    path planning (belief-based, own graded limits) — the same planner
    treatment CARA-EL's execution layer applies. See module docstring."""
    M = M_module
    AchordSim = make_achord_sim(gnf_module, M_module)

    # The base bot class is whatever make_achord_sim wrapped robots into; we
    # patch planning at the class level of a subclass sim's robots instead:
    class AchordRiskSim(AchordSim):

        def _build_robots(self):
            super()._build_robots()
            # Patch the (factory-local) bot class once. Each make_achord_*
            # call closes over its own ACHORDBot class, so this cannot leak
            # into the hazard-blind ACHORD-insp variant.
            if self.robots:
                cls = type(self.robots[0])
                if not getattr(cls, '_risk_planner_installed', False):
                    _install_risk_planner(cls, M)
                    cls._risk_planner_installed = True

    return AchordRiskSim


def _install_risk_planner(bot_cls, M):
    """Replace this bot CLASS's _plan_to with a risk-aware version (belief-only).
    Parameters mirror CARA-EL's planner call exactly: the robot's own
    max-pooled hazard-belief risk map, its own graded limits, and soft-cost
    hazard shaping (alpha/beta=1.0, soft_frac=0.85) — versus the substrate's
    zero risk map, limits=9999, alpha/beta=0."""

    def _recompute_chunked(self):
        nW = M.GRID_W // M.CHUNK_SIZE; nH = M.GRID_H // M.CHUNK_SIZE
        mT = np.zeros((M.GRID_W, M.GRID_H), dtype=np.float32)
        mR = np.zeros((M.GRID_W, M.GRID_H), dtype=np.float32)
        km = self.known_mask
        mT[km] = self.temp_belief[km]; mR[km] = self.rad_belief[km]
        np.nan_to_num(mT, copy=False); np.nan_to_num(mR, copy=False)
        self.chunked[0] = mT.reshape(nW, M.CHUNK_SIZE, nH, M.CHUNK_SIZE).max(axis=(1, 3))
        self.chunked[1] = mR.reshape(nW, M.CHUNK_SIZE, nH, M.CHUNK_SIZE).max(axis=(1, 3))

    def _plan_to(self, goal) -> bool:
        self._recompute_chunked()
        _zero_traffic = np.zeros((M.GRID_W, M.GRID_H), dtype=np.uint16)
        can_enter = bool(self.caps_mask & (M.CAP_STAIRS | M.CAP_AIR))
        relay_ok = self.sim._relay_ok
        gx, gy = goal
        path = M.AStar.search(
            start=self.pos, goal=goal,
            caps_mask=self.caps_mask,
            terrain_u8=self.terrain_belief,
            temp_f32=self.temp_belief, rad_f32=self.rad_belief,
            chunked_risk=self.chunked,                       # belief risk map
            temp_limit=self.temp_limit, rad_limit=self.rad_limit,
            radio_shadow=self.sim.radio_shadow,
            relay_ok_fn=(lambda z: bool(relay_ok[gx, gy])) if can_enter
                        else (lambda z: False),
            cell_to_zone_fn=self.sim.cell_to_zone,
            global_cov=self.sim.global_cov,
            unk_pen=0.3, info_w=0.1, unk_prior=0.25,
            # soft_frac=0.45: soft hazard cost begins at the DOSE-ACCRUAL
            # onset (0.5x limit) rather than near the instant-death limit —
            # under the graded dose model, the fringe itself is the killer.
            alpha_mult=1.0, beta_mult=1.0, soft_frac=0.45,
            traffic_u16=_zero_traffic, traffic_w=0.0,
            shadow_border=self.sim._shadow_border_mask_cache,
        )
        if not path:
            self.failed_goals[goal] = self.sim.timestep + 60
            return False
        self.goal = goal
        self.path = path
        return True

    bot_cls._recompute_chunked = _recompute_chunked
    bot_cls._plan_to = _plan_to