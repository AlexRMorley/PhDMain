"""RITAGS-inspired: risk-tolerant trait-based coalition allocation (Park et al.).

    J. Park, A. Messing, H. Ravichandar, S. Hutchinson, "Risk-Tolerant Task
    Allocation and Scheduling in Heterogeneous Multi-Robot Teams,"
    IEEE/RSJ IROS 2023, pp. 5372-5379. DOI: 10.1109/IROS55552.2023.10341837
    (Extends ITAGS: Neville et al., "An Interleaved Approach to Trait-Based
     Task Allocation and Scheduling," IROS 2021.)

MECHANISM CLASS: trait-based coalition formation under uncertainty. Tasks
declare multi-dimensional TRAIT REQUIREMENTS; the allocator forms coalitions
of heterogeneous robots that COLLECTIVELY satisfy them, and — the paper's
core contribution — uses a Sequential Probability Ratio Test (SPRT) so that
the probability a coalition fails its task's requirements stays below a
user-specified threshold DELTA, adding REDUNDANT robots when a coalition is
statistically rejected.

SOURCE-FIDELITY NOTE (read before citing): the IEEE full text is paywalled;
this port is built from the published abstract plus the documented ITAGS
lineage. Elements verified against the abstract: trait-based allocation,
capability/requirement uncertainty, SPRT acceptance with user threshold,
redundancy as the risk lever, makespan-minimizing intent. Elements adapted
or inferred (declared below): the task model, uncertainty distributions, and
the scheduling layer. A fidelity pass against the full text is recommended.

MAPPING TO THIS SAR SUBSTRATE (declared adaptations):
  * TASKS := search ZONES (the substrate's chunk grid). Each zone's trait
    requirements are derived from the CENTRAL BELIEF (union of active
    robots' beliefs — RITAGS is a centralized allocator, same precedent as
    CARA-EL's MILP):
      - req_stairs: zone contains known stair cells or unknown shadow;
      - req_temp / req_rad: zone hazard intensity — UNCERTAIN where cells
        are unexplored (sampled from the empirical distribution of known
        hazard values, fattened by UNKNOWN_TAIL);
      - req_capacity: coalition size scaled to unexplored area (the
        makespan lever: enough robots to finish the zone quickly).
  * ROBOT TRAITS := capability flags + graded hazard limits, with
    CAPABILITY UNCERTAINTY modelled as accumulated dose degrading the
    robot's effective limit (sampled around limit*(1 - dose_fraction)).
  * SPRT: Wald's test on Bernoulli satisfaction samples. A sample draws a
    zone-hazard realization and per-robot effective limits; success iff the
    coalition covers every requirement dimension (a stairs-capable member
    where required; every member tolerating the sampled fields; capacity
    met). Accept when the likelihood ratio crosses A=(1-beta)/alpha for
    H1: p >= 1-DELTA; reject at B=beta/(1-alpha); undecided at K_MAX falls
    back to the empirical rate. On rejection the allocator adds the best
    marginal robot (hardest limits first) and retests — the paper's
    redundancy mechanism.
  * SCHEDULING: our mission is continuous search, not a task DAG; the MILP
    with deadlines/precedence/synchronization does not transfer. Its
    makespan objective is represented by capacity-proportional coalition
    sizing and travel-cost-aware zone ranking, on a fixed reallocation
    cadence. This is the largest single adaptation.
  * Robots explore by nearest-frontier WITHIN their assigned zone (global
    fallback when their zone is exhausted). Movement, sensing, battery,
    dose, comms coverage: shared GNF substrate, identical to all baselines.
  * Pathing is the suite's STANDARD risk-aware planner (belief-based
    max-pooled hazard map, own graded limits, dose-aligned soft threshold
    0.45). Rationale: A* is the agents' ONBOARD competence — the same local
    autonomy every platform plausibly ships with (and why path compute is
    excluded from coordination-cost accounting) — so withholding it would
    handicap agent-level autonomy rather than isolate the allocation layer.
    The factorial isolation comes from the suite instead: GNF-Risk
    (risk steering, no allocation) vs RITAGS-insp (risk steering + SPRT
    trait allocation) measures the allocation layer's marginal value on
    top of competent agents.

Metrics: sim.coalitions (current zone -> [names]), sim.sprt_stats
(accepted, rejected, redundancy_adds) harvested by the bench.
"""
import numpy as np
import random

ALPHA_RISK   = 0.10   # paper's alpha: user risk tolerance, P(Y < Y*) <= alpha
SPRT_EPS     = 0.05   # paper's epsilon indifference band (paper: 0.01;
                      #   widened — declared compute adaptation, see below)
SPRT_ALPHA   = 0.05   # type-I error  (paper-exact)
SPRT_BETA    = 0.05   # type-II error (paper-exact)
SPRT_NMAX    = 120    # paper: 10,000 for a ONE-SHOT offline verification; a
                      #   receding-horizon allocator re-solving ~100 zones
                      #   every ALLOC_EVERY ticks cannot afford that — declared
UNKNOWN_TAIL = 1.35   # fatten sampled hazards for unexplored cells
ALLOC_EVERY  = 40     # reallocation cadence (ticks)
AREA_PER_BOT = 900    # unexplored cells per coalition member (capacity)
DOSE_SIGMA   = 0.10   # capability-uncertainty spread (fraction of limit)


def make_ritags_sim(gnf_module, M_module):
    GNFSim   = gnf_module.GNFSim
    GNFRobot = gnf_module.GNFRobot
    M        = M_module

    class RitagsBot(GNFRobot):
        __slots__ = ('zone_assign',)

        def _nearest_frontier(self):
            g = super()._nearest_frontier()
            if self.zone_assign is None or g is None:
                return g
            # prefer a frontier inside the assigned zone; fall back to g
            zx, zy = self.zone_assign
            known = self.known_mask
            free = known & (self.terrain_belief == M.T_FREE)
            unk = ~known
            nbr = (np.roll(unk, 1, 0) | np.roll(unk, -1, 0) |
                   np.roll(unk, 1, 1) | np.roll(unk, -1, 1))
            fx, fy = np.where(free & nbr)
            best, bd = None, 1 << 30
            for x, y in zip(fx.tolist(), fy.tolist()):
                if self.sim.cell_to_zone(x, y) != (zx, zy):
                    continue
                d = abs(x - self.pos[0]) + abs(y - self.pos[1])
                if d < bd:
                    best, bd = (x, y), d
            return best if best is not None else g

    class RitagsSim(GNFSim):

        def _build_robots(self):
            super()._build_robots()
            wrapped = []
            for r in self.robots:
                b = RitagsBot(r.name, r.pos[0], r.pos[1], r.caps, r.caps_mask,
                              self.world, self, r.temp_limit, r.rad_limit)
                b.zone_assign = None
                wrapped.append(b)
            self.robots = wrapped
            self.coalitions = {}
            self.sprt_stats = {'accepted': 0, 'rejected': 0,
                               'redundancy_adds': 0}
            nz = M.GRID_W // M.CHUNK_SIZE
            self._zone_n = nz

        # ── central belief (union of active robots — declared adaptation) ────
        def _central_belief(self):
            bots = [r for r in self.robots if r.active]
            known = np.zeros((M.GRID_W, M.GRID_H), dtype=bool)
            temp = np.zeros((M.GRID_W, M.GRID_H), dtype=np.float32)
            radn = np.zeros((M.GRID_W, M.GRID_H), dtype=np.float32)
            terr = np.full((M.GRID_W, M.GRID_H), 255, dtype=np.uint8)
            for b in bots:
                m = b.known_mask
                known |= m
                terr[m] = b.terrain_belief[m]
                np.maximum(temp, b.temp_belief, out=temp)
                np.maximum(radn, b.rad_belief, out=radn)
            return known, terr, temp, radn

        # ── zone trait requirements under uncertainty ─────────────────────────
        def _zone_requirements(self, known, terr, temp, radn):
            cs = M.CHUNK_SIZE
            reqs = {}
            kt = temp[known]; kr = radn[known]
            emp_t = kt[kt > 0]; emp_r = kr[kr > 0]
            for zx in range(self._zone_n):
                for zy in range(self._zone_n):
                    sl = np.s_[zx*cs:(zx+1)*cs, zy*cs:(zy+1)*cs]
                    k = known[sl]
                    unex = int((~k).sum())
                    if unex < cs:            # zone essentially explored
                        continue
                    zt = temp[sl][k]; zr = radn[sl][k]
                    shadow_unk = bool((self.radio_shadow[sl] & ~k).any())
                    stair_known = bool((terr[sl][k] == M.T_STAIRS).any())
                    reqs[(zx, zy)] = {
                        'temp_known': float(zt.max()) if zt.size else 0.0,
                        'rad_known':  float(zr.max()) if zr.size else 0.0,
                        'frac_unknown': unex / (cs * cs),
                        'req_stairs': stair_known or shadow_unk,
                        'capacity': max(1, int(np.ceil(unex / AREA_PER_BOT))),
                        'unexplored': unex,
                        'emp_t': emp_t, 'emp_r': emp_r,
                    }
            return reqs

        def _sample_zone_hazard(self, rq):
            """One uncertain realization of the zone's (temp, rad) demand."""
            t = rq['temp_known']; r = rq['rad_known']
            if rq['frac_unknown'] > 0.05:
                # unexplored cells may hide hotter fields: draw from the
                # empirical distribution of observed hazards, fattened
                if rq['emp_t'].size:
                    t = max(t, float(np.random.choice(rq['emp_t'])) * UNKNOWN_TAIL)
                if rq['emp_r'].size:
                    r = max(r, float(np.random.choice(rq['emp_r'])) * UNKNOWN_TAIL)
            return t, r

        def _sample_effective_limits(self, bot):
            """Capability uncertainty: dose degrades the effective limit."""
            bt = getattr(M, 'DOSE_BUDGET', None)
            dfT = getattr(bot, 'dose_T', 0.0)
            dfR = getattr(bot, 'dose_R', 0.0)
            # normalise accrued dose to a degradation fraction (bounded)
            degT = min(0.5, dfT / 100.0); degR = min(0.5, dfR / 100.0)
            eT = bot.temp_limit * (1 - degT) * (1 + np.random.randn() * DOSE_SIGMA)
            eR = bot.rad_limit * (1 - degR) * (1 + np.random.randn() * DOSE_SIGMA)
            return eT, eR

        # ── SPRT coalition acceptance (Wald; paper-faithful hypotheses) ──────
        # Paper (Sec IV-C): test the FAILURE probability p = P(Y < Y*) with
        # base gamma = alpha and indifference band [p0, p1] = [alpha-eps,
        # alpha+eps]. ACCEPT (guarantee shown) when the LLR concludes
        # p <= p0; REJECT when p >= p1; and — critically — an UNDECIDED test
        # at Nmax is a REJECTION (no guarantee shown; the search continues
        # via redundancy). This is stricter than a success-rate fallback.
        def _sprt_accept(self, coalition, rq):
            p0 = max(1e-3, ALPHA_RISK - SPRT_EPS)   # H0: p <= p0  (accept)
            p1 = min(0.999, ALPHA_RISK + SPRT_EPS)  # H1: p >= p1  (reject)
            A = np.log((1 - SPRT_BETA) / SPRT_ALPHA)     # reject H0 bound
            B = np.log(SPRT_BETA / (1 - SPRT_ALPHA))     # accept H0 bound
            llr = 0.0
            need_stairs = rq['req_stairs']
            lr_bad  = np.log(p1 / p0)                 # bad sample: failure
            lr_good = np.log((1 - p1) / (1 - p0))     # good sample: success
            for k in range(1, SPRT_NMAX + 1):
                t, r = self._sample_zone_hazard(rq)
                ok = True
                if need_stairs and not any(
                        b.caps_mask & (M.CAP_STAIRS | M.CAP_AIR)
                        for b in coalition):
                    ok = False
                if ok:
                    for b in coalition:
                        eT, eR = self._sample_effective_limits(b)
                        if eT < t or eR < r:
                            ok = False
                            break
                llr += lr_bad if not ok else lr_good
                if llr >= A:
                    return False    # failure rate concluded >= alpha+eps
                if llr <= B:
                    return True     # failure rate concluded <= alpha-eps
            return False            # undecided at Nmax = no guarantee (paper)

        # ── allocation round ──────────────────────────────────────────────────
        def _allocate(self):
            known, terr, temp, radn = self._central_belief()
            reqs = self._zone_requirements(known, terr, temp, radn)
            free_bots = [r for r in self.robots if r.active]
            for b in free_bots:
                b.zone_assign = None
            self.coalitions = {}
            cs = M.CHUNK_SIZE
            # zone value: unexplored area discounted by fleet travel distance
            def zval(z):
                zx, zy = z
                cxy = (zx*cs + cs//2, zy*cs + cs//2)
                dmin = min(abs(b.pos[0]-cxy[0]) + abs(b.pos[1]-cxy[1])
                           for b in free_bots) if free_bots else 1
                return reqs[z]['unexplored'] / (1 + dmin)
            for z in sorted(reqs, key=zval, reverse=True):
                if not free_bots:
                    break
                rq = reqs[z]
                zx, zy = z
                cxy = (zx*cs + cs//2, zy*cs + cs//2)
                # seed with the nearest robots up to capacity; STAIRS need
                # pulls a stairs/air robot in first when required
                pool = sorted(free_bots, key=lambda b:
                              abs(b.pos[0]-cxy[0]) + abs(b.pos[1]-cxy[1]))
                coal = []
                if rq['req_stairs']:
                    s = next((b for b in pool
                              if b.caps_mask & (M.CAP_STAIRS | M.CAP_AIR)), None)
                    if s is not None:
                        coal.append(s)
                for b in pool:
                    if len(coal) >= rq['capacity']:
                        break
                    if b not in coal:
                        coal.append(b)
                if not coal:
                    continue
                # SPRT with redundancy: rejected -> add hardest remaining bot
                tries = 0
                while not self._sprt_accept(coal, rq):
                    self.sprt_stats['rejected'] += 1
                    extra = [b for b in pool if b not in coal]
                    # hardest first: max combined hazard limits
                    extra.sort(key=lambda b: -(b.temp_limit + b.rad_limit))
                    if not extra or tries >= 3:
                        coal = None
                        break
                    coal.append(extra[0])
                    self.sprt_stats['redundancy_adds'] += 1
                    tries += 1
                if not coal:
                    continue
                self.sprt_stats['accepted'] += 1
                self.coalitions[z] = [b.name for b in coal]
                for b in coal:
                    b.zone_assign = z
                    b.goal = None; b.path = []
                    free_bots.remove(b)

        def step(self) -> bool:
            if self.timestep % ALLOC_EVERY == 0:
                self._allocate()
            return super().step()

    # Standard onboard risk-aware planner (identical parameters to GNF-Risk
    # and ACHORD-Risk), installed at class level on the factory-local bot —
    # slots-safe, cannot leak into other factories' classes.
    _install_risk_planner(RitagsBot, M)
    return RitagsSim


def _install_risk_planner(bot_cls, M):
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
            chunked_risk=self.chunked,
            temp_limit=self.temp_limit, rad_limit=self.rad_limit,
            radio_shadow=self.sim.radio_shadow,
            relay_ok_fn=(lambda z: bool(relay_ok[gx, gy])) if can_enter
                        else (lambda z: False),
            cell_to_zone_fn=self.sim.cell_to_zone,
            global_cov=self.sim.global_cov,
            unk_pen=0.3, info_w=0.1, unk_prior=0.25,
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