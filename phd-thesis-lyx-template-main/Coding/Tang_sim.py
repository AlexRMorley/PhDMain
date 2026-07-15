"""Tang-CM: connectivity-maintained decentralized exploration (Tang et al. port).

    W. Tang, C. Li, J. Wu, Q. Zhu, "Decentralized Communication-Maintained
    Coordination for Multi-Robot Exploration: Achieving Connectivity and
    Adaptability," IEEE/RSJ IROS 2024, pp. 3417-3424.
    DOI: 10.1109/IROS58592.2024.10802832

MECHANISM CLASS: connectivity-MAINTAINED exploration — the comms graph
(pairwise links within range rho, multi-hop chains allowed) must stay
connected at all times. The opposite pole to Framework M/N's
connectivity-RESTORED design (break comms deliberately, dispatch relays).

WHAT IS PORTED (the paper's expert policy — its imitation target):
  * Leader-follower team: one leader runs greedy nearest-frontier (the
    paper's leader policy verbatim); followers choose goals decentrally.
  * Candidate goals per follower decision: frontier goals from the shared
    belief + "regular goals" uniformly sampled in explored free space
    (repositioning moves that exist purely to maintain connectivity).
  * One-step forward prediction: teammates advanced along their remaining
    paths to the follower's estimated arrival time; the candidate is scored
    on the PREDICTED formation graph.
  * Expert rule: candidates keeping the robot connected to the leader's
    component score 0; disconnecting candidates score minus the movement
    distance needed to reconnect. Followers pick among connected candidates.
  * Group belief sharing: the paper states explicitly that, since the group
    stays connected, each robot's map is shared with the whole group. We
    merge beliefs within actual connected components every SYNC_EVERY ticks.

DECLARED ADAPTATIONS (information-access audit):
  * LEARNED POLICY NOT PORTED. The GNN+attention network trained via
    DAgger + MCTS refines exploration efficiency ON TOP of the expert rule.
    We stand in for that refinement with utility-based frontier selection
    among the CONNECTED candidates: nearest-frontier cost plus a DISPERSION
    bonus (distance to the nearest predicted teammate position), the
    classical coordination utility the paper's related work builds on
    (utility reduction of targets chosen by teammates). Without dispersion
    the shared-belief group degenerates into a blob that explores almost
    nothing — validated: found 2/45 at t=200 with pure nearest-frontier.
    Tang-CM represents the mechanism class, not the trained artefact;
    results may understate the learned system's efficiency. (A trained
    entry belongs to the MARVEL/learning slot.)
  * Pairwise range rho := RELAY_COVERAGE_RADIUS_CELLS (default 30) — the
    same radio-physics constant every model in the suite uses. Links are
    range-only (work through walls at short range), consistent with the
    substrate's relay-coverage precedent.
  * The paper's expert "logit equal to reconnection distance" is
    implemented signed (logit = -distance) — required for the stated
    softmax preference of connected goals to hold.
  * Teammates whose predicted path ends before the follower's arrival hold
    at their goal (the paper extrapolates away from the group centroid).
  * Arrival times estimated by Euclidean distance at unit speed (full A*
    only for the chosen goal), and follower decisions are re-evaluated when
    the current goal completes or expires — the paper's asynchronous
    decide-on-arrival scheme under our synchronous-round simulator.
  * No hazard concept exists in the paper: pathing is the hazard-blind
    substrate planner, same as GNF / CARA-2022 / ACHORD-insp.
  * WATER PLATFORMS EXEMPT: the paper's fleet is homogeneous ground
    robots; a Boat physically cannot hold position in a land chain, so
    forcing it into the constraint yields a useless robot, not a fair test.
    Boats keep their normal spawns and explore independently (plain
    substrate behaviour); the connectivity mechanism governs the land+air
    team it can meaningfully govern. conn_ratio is measured over that team.
  * CONNECTED START: the paper deploys the team together; the bench's
    default 9-cluster scatter would test our deployment, not their
    mechanism. Tang-CM therefore respawns the team in a single cluster
    around the leader (the world itself is untouched — spawn happens after
    world generation). Boats are placed at the water cells NEAREST the
    leader: water-bound platforms may make full initial connectivity
    physically impossible in this heterogeneous fleet, which is itself an
    honest finding about connectivity-maintained schemes under platform
    heterogeneity.

Metric: sim.conn_ratio — fraction of ticks the active-robot comms graph was
fully connected (the paper's connectivity-ratio metric).

TWO VARIANTS are exported:
  make_tang_sim       — Tang-CM: hazard-BLIND pathing (paper-faithful; the
                        paper has no hazard concept).
  make_tang_risk_sim  — Tang-Risk: adds the suite's standard belief-based
                        risk-aware planner (own graded limits, dose-aligned
                        soft threshold 0.45), the same treatment GNF-Risk
                        and ACHORD-Risk received, for the capability ladder.
"""
import numpy as np
import random

SYNC_EVERY       = 5     # belief-merge cadence within connected components
N_FRONTIER_GOALS = 8     # frontier candidates per decision
N_REGULAR_GOALS  = 12    # repositioning candidates sampled in explored space
LOGIT_CONNECTED  = 0.0
TETHER_MARGIN    = 0.85  # goals vetted at 0.85*rho: Euclidean prediction
                         # underestimates real A* detours, so plan with slack
RECALL_MARGIN    = 0.95  # preventive recall before a link actually snaps


def make_tang_sim(gnf_module, M_module):
    GNFSim   = gnf_module.GNFSim
    GNFRobot = gnf_module.GNFRobot
    M        = M_module

    def _is_waterbound(r):
        return (M.Capability.WATER in r.caps
                and M.Capability.AIR not in r.caps)

    class TangBot(GNFRobot):
        __slots__ = ('is_leader',)

        def _nearest_frontier(self):
            if self.is_leader or _is_waterbound(self):
                # Leader: paper's greedy closest-frontier. Boats: exempt from
                # the land-chain constraint (see docstring) — plain substrate
                # exploration of the water network.
                return super()._nearest_frontier()
            g = self.sim._tang_choose_goal(self)
            if g is None:
                return super()._nearest_frontier()
            return g

    class TangSim(GNFSim):

        def _build_robots(self):
            super()._build_robots()
            wrapped = []
            for r in self.robots:
                b = TangBot(r.name, r.pos[0], r.pos[1], r.caps, r.caps_mask,
                            self.world, self, r.temp_limit, r.rad_limit)
                b.is_leader = False
                wrapped.append(b)
            self.robots = wrapped
            # Leader: the paper assumes better comms/compute; we designate the
            # first hardened ground robot (Rover) if present, else robot 0.
            leader = next((b for b in self.robots if b.name.startswith('Rover')),
                          self.robots[0])
            leader.is_leader = True
            self._tang_leader = leader
            self._tang_stage_team(leader)
            self._rho = float(getattr(M, 'RELAY_COVERAGE_RADIUS_CELLS', 30))
            self._conn_ticks = 0
            self._total_ticks = 0
            self.conn_ratio = 1.0

        def _tang_stage_team(self, leader):
            """Deploy the team as ONE cluster (paper assumption), at the
            SAFEST standard staging point: the connected-start requirement
            means a single bad staging choice exposes the whole team at
            once, so the staging cluster is chosen by minimum local hazard
            + shadow (a deployment decision, not agent knowledge —
            disclosed). Boats go to the nearest water cells instead."""
            W, H = M.GRID_W, M.GRID_H
            centers = [(W//6, H//6), (W//2, H//6), (5*W//6, H//6),
                       (W//6, H//2), (W//2, H//2), (5*W//6, H//2),
                       (W//6, 5*H//6), (W//2, 5*H//6), (5*W//6, 5*H//6)]
            def _danger(cx, cy):
                s = 0.0
                for x in range(max(0, cx-30), min(W, cx+31), 3):
                    for y in range(max(0, cy-30), min(H, cy+31), 3):
                        c = self.world.grid[x][y]
                        s += c.get("temp", 0.0) + c.get("rad", 0.0)
                        if self.radio_shadow[x, y]:
                            s += 40.0
                return s
            lx, ly = min(centers, key=lambda c: _danger(*c))
            leader.pos = (lx, ly)
            free = []
            for rad in range(2, 40):
                for dx in range(-rad, rad + 1):
                    for dy in range(-rad, rad + 1):
                        if max(abs(dx), abs(dy)) != rad:
                            continue
                        x, y = lx + dx, ly + dy
                        if not (0 < x < W - 1 and 0 < y < H - 1):
                            continue
                        c = self.world.grid[x][y]
                        # spawn placement is experimental setup: avoid dropping
                        # the cluster inside a hazard field (temp/rad hot zones)
                        if (c["t"] == M.T_FREE and not self.radio_shadow[x, y]
                                and c.get("temp", 0) < 50 and c.get("rad", 0) < 50):
                            free.append((x, y))
                if len(free) > len(self.robots) * 2:
                    break
            fi = 0
            for b in self.robots:
                if b is leader or _is_waterbound(b):
                    continue          # boats keep their original water spawns
                if fi < len(free):
                    b.pos = free[fi]; fi += 1
                # else: keep original spawn — never stack robots on one cell

        # ── comms graph ───────────────────────────────────────────────────────
        def _components(self, positions):
            """Connected components of the range-rho graph over positions
            (list of (x, y)). Returns a label per index."""
            n = len(positions)
            if n == 0:
                return []
            P = np.asarray(positions, dtype=np.float64)
            d2 = ((P[:, None, :] - P[None, :, :]) ** 2).sum(axis=2)
            adj = d2 <= self._rho * self._rho
            label = [-1] * n
            comp = 0
            for s in range(n):
                if label[s] >= 0:
                    continue
                stack = [s]
                label[s] = comp
                while stack:
                    u = stack.pop()
                    for v in np.where(adj[u])[0]:
                        if label[v] < 0:
                            label[v] = comp
                            stack.append(v)
                comp += 1
            return label

        # ── group belief sharing (paper: maps shared within the group) ───────
        def _tang_sync_beliefs(self):
            bots = [r for r in self.robots if r.active]
            if len(bots) < 2:
                return
            labels = self._components([b.pos for b in bots])
            for comp in set(labels):
                members = [b for b, l in zip(bots, labels) if l == comp]
                if len(members) < 2:
                    continue
                known = members[0].known_mask.copy()
                for b in members[1:]:
                    known |= b.known_mask
                terr = members[0].terrain_belief.copy()
                temp = members[0].temp_belief.copy()
                radn = members[0].rad_belief.copy()
                for b in members[1:]:
                    m = b.known_mask
                    terr[m] = b.terrain_belief[m]
                    np.maximum(temp, b.temp_belief, out=temp)
                    np.maximum(radn, b.rad_belief, out=radn)
                for b in members:
                    b.known_mask[:] = known
                    b.terrain_belief[:] = terr
                    b.temp_belief[:] = temp
                    b.rad_belief[:] = radn

        # ── follower decision (the paper's expert policy) ─────────────────────
        def _tang_candidates(self, bot):
            known = bot.known_mask
            terr = bot.terrain_belief
            free = known & (terr == M.T_FREE)
            unk = ~known
            nbr_unk = (np.roll(unk, 1, 0) | np.roll(unk, -1, 0) |
                       np.roll(unk, 1, 1) | np.roll(unk, -1, 1))
            fx, fy = np.where(free & nbr_unk)
            cands = []
            if len(fx):
                d = np.abs(fx - bot.pos[0]) + np.abs(fy - bot.pos[1])
                order = np.argsort(d)
                step = max(1, len(order) // N_FRONTIER_GOALS)
                for k in order[::step][:N_FRONTIER_GOALS]:
                    cands.append(((int(fx[k]), int(fy[k])), True))
            ex, ey = np.where(free)
            if len(ex):
                idx = np.random.choice(len(ex),
                                       size=min(N_REGULAR_GOALS, len(ex)),
                                       replace=False)
                for k in idx:
                    cands.append(((int(ex[k]), int(ey[k])), False))
            return cands

        def _tang_choose_goal(self, bot):
            others = [r for r in self.robots
                      if r.active and r is not bot and not _is_waterbound(r)]
            if not others:
                return None
            cands = self._tang_candidates(bot)
            if not cands:
                return None
            leader = self._tang_leader

            best_conn = None      # (dist, goal) among connected frontier goals
            best_conn_reg = None  # fallback: connected regular goal
            best_logit = None     # (logit, goal) if nothing keeps connection

            for (gx, gy), is_frontier in cands:
                eta = int(((gx - bot.pos[0]) ** 2 + (gy - bot.pos[1]) ** 2) ** 0.5)
                pred = []
                for o in others:
                    path = getattr(o, 'path', None) or []
                    if path:
                        pred.append(tuple(path[min(eta, len(path) - 1)]))
                    else:
                        pred.append(o.pos)
                nodes = pred + [(gx, gy)]
                _rho_full = self._rho
                self._rho = _rho_full * TETHER_MARGIN
                labels = self._components(nodes)
                self._rho = _rho_full
                li = others.index(leader) if leader in others else None
                # bot may BE the leader only in degenerate cases; guard:
                leader_label = labels[li] if li is not None else labels[-1]
                me_label = labels[-1]
                if me_label == leader_label:
                    d = abs(gx - bot.pos[0]) + abs(gy - bot.pos[1])
                    # dispersion: stretch the chain — prefer frontiers away
                    # from teammates' predicted positions (anti-overlap)
                    sep = min(((gx - px) ** 2 + (gy - py) ** 2) ** 0.5
                              for px, py in pred)
                    util = -d + 1.5 * min(sep, self._rho)
                    if is_frontier:
                        if best_conn is None or util > best_conn[0]:
                            best_conn = (util, (gx, gy))
                    else:
                        if best_conn_reg is None or util > best_conn_reg[0]:
                            best_conn_reg = (util, (gx, gy))
                else:
                    # expert: logit = -(movement needed to reconnect)
                    comp_nodes = [nodes[i] for i, l in enumerate(labels[:-1])
                                  if l == leader_label]
                    if comp_nodes:
                        dmin = min(((gx - cx) ** 2 + (gy - cy) ** 2) ** 0.5
                                   for cx, cy in comp_nodes)
                        logit = -(max(0.0, dmin - self._rho))
                    else:
                        logit = -1e9
                    if best_logit is None or logit > best_logit[0]:
                        best_logit = (logit, (gx, gy))

            if best_conn is not None:
                return best_conn[1]          # greedy among connected frontiers
            if best_conn_reg is not None:
                return best_conn_reg[1]      # reposition to stay connected
            if best_logit is not None:
                return best_logit[1]         # least reconnection distance
            return None

        def step(self) -> bool:
            if self.timestep % SYNC_EVERY == 0:
                self._tang_sync_beliefs()
            # conn_ratio measures GROUND+AIR connectivity: water-only
            # platforms (Boats) physically cannot hold position in a land
            # chain, so including them caps the metric at an unreachable
            # value in a heterogeneous fleet (declared adaptation — the
            # paper's fleet is homogeneous). Enforcement below still covers
            # every robot.
            bots = [r for r in self.robots if r.active
                    and not _is_waterbound(r)]
            if len(bots) >= 2:
                labels = self._components([b.pos for b in bots])
                self._total_ticks += 1
                if len(set(labels)) == 1:
                    self._conn_ticks += 1
                else:
                    # Continuous enforcement (the paper's RL penalises
                    # disconnection every step, not only at decision time):
                    # followers that drifted out of the leader's component
                    # abort their current goal and re-decide immediately —
                    # the expert rule then returns them toward the team.
                    leader_idx = bots.index(self._tang_leader) \
                        if self._tang_leader in bots else 0
                    lead_lab = labels[leader_idx]
                    for b, l in zip(bots, labels):
                        if l != lead_lab and not b.is_leader:
                            b.goal = None; b.path = []
                # PREVENTIVE recall: a follower whose nearest teammate in the
                # leader's component is beyond RECALL_MARGIN*rho aborts before
                # the link snaps (keeps the team together, per the constraint).
                leader_idx = bots.index(self._tang_leader) \
                    if self._tang_leader in bots else 0
                lead_lab = labels[leader_idx]
                comp_pos = [b.pos for b, l in zip(bots, labels) if l == lead_lab]
                thr2 = (self._rho * RECALL_MARGIN) ** 2
                for b, l in zip(bots, labels):
                    if b.is_leader or l != lead_lab or not b.goal:
                        continue
                    d2 = min((b.pos[0]-px)**2 + (b.pos[1]-py)**2
                             for px, py in comp_pos if (px, py) != b.pos) \
                         if len(comp_pos) > 1 else 0
                    if d2 > thr2:
                        b.goal = None; b.path = []
                self.conn_ratio = self._conn_ticks / max(1, self._total_ticks)
            return super().step()

    return TangSim


def make_tang_risk_sim(gnf_module, M_module):
    """Tang-Risk: identical connectivity-maintained coordination, plus
    belief-based risk-aware path planning (see module docstring)."""
    M = M_module
    TangSim = make_tang_sim(gnf_module, M_module)

    class TangRiskSim(TangSim):
        def _build_robots(self):
            super()._build_robots()
            # Patch the factory-local bot class once (slots-safe; each
            # make_tang_* call closes over its own TangBot class, so the
            # hazard-blind Tang-CM variant cannot be contaminated).
            if self.robots:
                cls = type(self.robots[0])
                if not getattr(cls, '_risk_planner_installed', False):
                    _install_risk_planner(cls, M)
                    cls._risk_planner_installed = True

    return TangRiskSim


def _install_risk_planner(bot_cls, M):
    """Belief-only risk-aware _plan_to (same parameters as GNF-Risk and
    ACHORD-Risk: own hazard-belief risk map, own graded limits, soft-cost
    onset aligned to the dose-accrual threshold)."""

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