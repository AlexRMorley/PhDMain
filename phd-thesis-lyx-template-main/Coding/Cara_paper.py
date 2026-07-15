"""CARA-2022: faithful implementation of the published CARA mechanism.

    Abu-Aisheh, R.; Bronzino, F.; Salaun, L.; Watteyne, T.
    "CARA: Connectivity-Aware Relay Algorithm for Multi-Robot Expeditions."
    Sensors 2022, 22(23), 9042. doi:10.3390/s22239042

Decision structure reproduced EXACTLY as published (Algorithm 1):
  1. A relay check is due every SLIDING_WINDOW ticks.
  2. A new relay is needed when a robot's windowed PDR estimate drops below
     LOWER_THRESHOLD (paper: 0.8).
  3. THAT robot (the one whose PDR degraded) becomes the relay.
  4. It is stationed at the CLOSEST position in its own communications-history
     map where its estimated PDR was >= UPPER_THRESHOLD (paper: 0.9), and it
     remains a relay for the rest of the mission (the paper never demotes).

DECLARED ADAPTATIONS (must be disclosed wherever results are reported):
  * PDR proxy. The paper estimates PDR from 500 ms heartbeats under a
    Pister-Hack RSSI model with concurrent-transmission flooding. This
    framework's radio model is binary (radio shadow + relay coverage), so no
    continuous link quality exists to threshold. We substitute a declared,
    physically-motivated proxy evaluated at the robot's cell each tick:
        1.0                       outside radio shadow (line-of-sight comms)
        0.9                       inside shadow WITH relay coverage
        max(0, 1 - 0.1 * depth)   inside uncovered shadow, where depth is the
                                  BFS penetration distance from the shadow
                                  border (deeper = weaker link)
    The windowed estimate is the mean over the last SLIDING_WINDOW ticks,
    mirroring the paper's sliding-window packet count.
  * Exploration substrate. The paper couples CARA to Atlas 2.0 (each robot
    sent to a random cell among the frontier cells closest to it). Here CARA
    is coupled to the shared GNF nearest-frontier explorer so that every
    baseline in the comparison runs on the same substrate (same robots,
    sensors, movement, hazard exposure). Same algorithm family; disclosed.
  * Fleet. The paper's fleet is homogeneous; this benchmark's fleet is the
    shared heterogeneous roster. CARA-2022 remains capability-BLIND exactly
    as published: any robot whose PDR degrades becomes a relay, including
    stair-capable robots whose locomotion is unique — this is a genuine
    mechanistic difference vs. the capability-aware CARA-EL variant, and
    observing its cost is part of the point of including both.

Everything else (movement, scanning, battery, hazard dose, survivor
detection) is inherited unchanged from the shared GNF substrate.
"""
import numpy as np
from collections import deque

SLIDING_WINDOW  = 10    # ticks per PDR estimation window (paper: sliding window)
LOWER_THRESHOLD = 0.8   # paper: place a relay when estimated PDR < this
UPPER_THRESHOLD = 0.9   # paper: station it where its history PDR >= this
DEPTH_DECAY     = 0.1   # v1 proxy: PDR loss per cell of shadow penetration

# ── PDR estimation model ──────────────────────────────────────────────────────
# 'v1' — deterministic proxy averaged over the window (original port).
# 'v2' — three fidelity upgrades toward the paper's actual estimator:
#   (a) BERNOULLI HEARTBEATS. The paper's orchestrator estimates PDR by
#       counting received heartbeat packets over a sliding window — an
#       empirical frequency estimate of a stochastic channel. v2 samples one
#       heartbeat per tick with success probability equal to the instantaneous
#       link quality; the windowed estimate is the received fraction. This
#       reproduces the paper's estimator INCLUDING its sampling noise, so
#       near-threshold robots trigger with the same jitter real packet loss
#       produces (v1's deterministic average could never show this).
#   (b) EXPONENTIAL WALL ATTENUATION. Cascaded obstruction losses are
#       multiplicative in link budget terms, so uncovered-shadow link quality
#       decays exponentially with penetration depth: exp(-ATTEN_ALPHA*depth)
#       (Pister-Hack-style path loss), replacing v1's ad-hoc linear ramp.
#       Calibration: depth 1 ~ 0.89, depth 2 ~ 0.79 (crosses the 0.8 trigger).
#   (c) CONCURRENT-TRANSMISSION MULTIPATH (simplified). The paper's CT
#       flooding combines independent paths: PDR = 1 - prod(1 - PDR_path).
#       v2 approximates one alternate path via the best-connected active
#       robot within CT_NEIGH_RANGE cells:
#           p = 1 - (1-p_self) * (1 - CT_RELAY_EFF * p_best_neighbour)
#       with CT_RELAY_EFF discounting the extra hop. Declared approximation.
PDR_MODEL     = 'v2'
ATTEN_ALPHA   = 0.12   # exponential depth-attenuation coefficient
CT_NEIGH_RANGE = 12    # Chebyshev range for the CT alternate path (cells)
CT_RELAY_EFF   = 0.8   # per-extra-hop efficiency of the CT path


def make_cara_paper_sim(gnf_module, M_module):
    """Return a CARAPaperSim class (same factory pattern as make_cara_sim)."""
    GNFSim   = gnf_module.GNFSim
    GNFRobot = gnf_module.GNFRobot
    M        = M_module

    class CARAPaperBot(GNFRobot):
        __slots__ = ('is_relay', 'relay_station')

        def _nearest_frontier(self):
            # Relays navigate to their station, then hold there permanently
            # (the paper never demotes a relay). Explorers behave exactly as
            # the substrate's nearest-frontier robots.
            if self.is_relay:
                if self.relay_station is not None and self.pos != self.relay_station:
                    return self.relay_station
                return None          # stationed — hold position
            return super()._nearest_frontier()

    class CARAPaperSim(GNFSim):

        def _build_robots(self):
            super()._build_robots()
            wrapped = []
            for r in self.robots:
                b = CARAPaperBot(r.name, r.pos[0], r.pos[1], r.caps, r.caps_mask,
                                 self.world, self, r.temp_limit, r.rad_limit)
                b.is_relay = False
                b.relay_station = None
                wrapped.append(b)
            self.robots = wrapped
            # per-robot sliding window of instantaneous proxy values
            self._pdr_win  = {r.name: deque(maxlen=SLIDING_WINDOW) for r in self.robots}
            # communications-history map: name -> {position: estimated PDR}
            # (paper: "[Robot_id, position, estimated_PDR]" entries)
            self._comm_hist = {r.name: {} for r in self.robots}
            self._shadow_depth = self._build_shadow_depth()
            self.relays_placed = []          # (tick, name, station) — for reporting

        def _build_shadow_depth(self):
            """BFS penetration depth of every shadow cell from the shadow
            border (depth 1 = adjacent to open air). Shadows are static, so
            this is computed once."""
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

        def _instant_proxy(self, pos):
            x, y = pos
            if not self.radio_shadow[x, y]:
                return 1.0
            if self._relay_ok[x, y]:
                return 0.9
            d = float(self._shadow_depth[x, y])
            if PDR_MODEL == 'v2':
                return float(np.exp(-ATTEN_ALPHA * d))
            return max(0.0, 1.0 - DEPTH_DECAY * d)

        def _link_quality(self, r, actives):
            """Instantaneous link quality for robot r, including the v2
            concurrent-transmission alternate path via the best-connected
            nearby active robot (paper: CT flooding multipath)."""
            p = self._instant_proxy(r.pos)
            if PDR_MODEL != 'v2' or p >= 1.0:
                return p
            best = 0.0
            rx, ry = r.pos
            for o in actives:
                if o is r:
                    continue
                if max(abs(o.pos[0] - rx), abs(o.pos[1] - ry)) <= CT_NEIGH_RANGE:
                    q = self._instant_proxy(o.pos)
                    if q > best:
                        best = q
            return 1.0 - (1.0 - p) * (1.0 - CT_RELAY_EFF * best)

        def step(self) -> bool:
            # ── CARA orchestrator bookkeeping (paper Algorithm 1) ─────────────
            # Uses the previous tick's coverage state — a one-tick estimation
            # lag, consistent with the paper's periodic estimation.
            _actives = [r for r in self.robots if r.active]
            for r in _actives:
                p = self._link_quality(r, _actives)
                win = self._pdr_win[r.name]
                if PDR_MODEL == 'v2':
                    # one heartbeat per tick, received with probability p —
                    # the windowed estimate is the received fraction, exactly
                    # the paper's packet-counting estimator (with its noise).
                    import random as _rnd
                    win.append(1.0 if _rnd.random() < p else 0.0)
                else:
                    win.append(p)
                est = sum(win) / len(win)
                # history is recorded at the position of estimation (paper)
                self._comm_hist[r.name][r.pos] = est

            if self.timestep % SLIDING_WINDOW == 0:          # step 1: check due
                for r in self.robots:
                    if not r.active or r.is_relay:
                        continue
                    win = self._pdr_win[r.name]
                    if not win:
                        continue
                    est = sum(win) / len(win)
                    if est < LOWER_THRESHOLD:                # step 2: relay needed
                        r.is_relay = True                    # step 3: THIS robot
                        r.goal = None; r.path = []
                        # step 4: closest own-history position with PDR >= upper
                        cands = [(abs(px - r.pos[0]) + abs(py - r.pos[1]), (px, py))
                                 for (px, py), v in self._comm_hist[r.name].items()
                                 if v >= UPPER_THRESHOLD]
                        if cands:
                            r.relay_station = min(cands)[1]
                        else:
                            # No qualifying history yet (early mission edge
                            # case the paper does not specify): fall back to
                            # the best-PDR position seen so far.
                            hist = self._comm_hist[r.name]
                            r.relay_station = max(hist.items(), key=lambda kv: kv[1])[0] \
                                              if hist else r.pos
                        self.relays_placed.append((self.timestep, r.name, r.relay_station))

            return super().step()

    return CARAPaperSim