"""
Price-of-Anarchy / Efficiency Benchmark for the Fleet Role-Selection Game
=========================================================================

WHAT THIS MEASURES (and what it deliberately does NOT)
------------------------------------------------------
The full search-and-rescue mission is a decentralised, partially-observed,
sequential decision problem (a Dec-POMDP), whose social optimum is NEXP-hard and
therefore intractable to compute exactly.  A literal mission-level Price of
Anarchy (PoA) is not well defined here, and any "optimum" from a static MILP
(e.g. the CARA baseline) is itself only an optimum of an *incomplete* model, not
the true social optimum — so dividing by it would overstate efficiency.

What IS well defined, exactly computable, and theory-backed is the **stage game**:
at a fixed tick, with belief / zones / relay-coverage frozen, every robot chooses
a role in {SCOUT, SCAN, LOITER, RELAY} and receives the utility defined by
`FleetSim._role_utility_pg`.  That stage game is a finite normal-form game; the
paper claims it is an exact potential game (Monderer & Shapley 1996) so best-
response converges to a pure Nash equilibrium.  For that game we can, on
tractable sub-instances, enumerate:

    * the social optimum            W* = max_a  Σ_i u_i(a)
    * the full set of pure NE        (best-response-stable joint actions)
    * the equilibrium our own best-response dynamics actually reach (BR)

and report the standard efficiency quantities used in the congestion-games
literature:

    Price of Anarchy   PoA = W* / min_{NE} W      (worst equilibrium)
    Price of Stability  PoS = W* / max_{NE} W      (best equilibrium)
    Realised efficiency  η  = W(BR) / W*           (what our dynamics achieve)

This is the same empirical recipe recent multi-robot work uses — measuring the
ratio of equilibrium welfare to a tractably-computed optimum on small instances
(see refs [9,10]) — but here the optimum is the *exact* brute-force optimum of
the real game, not a MILP surrogate, because the sub-instances are kept small
enough (K robots, 4^K joint actions) to enumerate.

WHY THE STAGE GAME IS THE RIGHT OBJECT
--------------------------------------
Roughgarden's smoothness / "robust PoA" framework [2] proves PoA bounds via a
(lambda, mu)-smoothness inequality giving PoA <= lambda / (1 - mu), and — the key
point — any bound proven this way applies *not only* to pure NE but to mixed and
coarse-correlated equilibria and to the empirical play of **no-regret learning
dynamics**.  Best-response role selection is exactly such a dynamic, so a
stage-game PoA bound certifies the equilibria our dynamics reach.  The relevant
known constants for orientation:
    * affine atomic congestion games:   PoA = 5/2                         [4,5]
    * submodular / valid-utility games:  PoA <= 2  (welfare >= 1/2 of opt)  [11]
    * CBBA read as a distributed welfare game:  PoA = 1/2, PoS = 1          [8-analysis]
The relay role is a public good and the same-capability congestion term is a
congestion cost, so the Fleet stage game sits in the submodular/valid-utility
family; an empirical PoA at or below ~2 (efficiency >= ~0.5) is the expected,
theory-consistent result, and PoS near 1 indicates a good equilibrium exists.

REFERENCES
----------
[1]  E. Koutsoupias and C. Papadimitriou, "Worst-case equilibria,"
     STACS 1999.  (origin of the price of anarchy)
[2]  T. Roughgarden, "Intrinsic robustness of the price of anarchy,"
     STOC 2009; J. ACM 62(5), 2015.  (smoothness framework; robust PoA that
     extends to coarse-correlated equilibria and no-regret dynamics)
[3]  T. Roughgarden and F. Schoppmann, "Local smoothness and the price of
     anarchy in splittable congestion games," SODA 2011; J. Econ. Theory 2015.
[4]  G. Christodoulou and E. Koutsoupias, "The price of anarchy of finite
     congestion games," STOC 2005.  (PoA = 5/2, affine atomic)
[5]  B. Awerbuch, Y. Azar, A. Epstein, "The price of routing unsplittable
     flow," STOC 2005.
[6]  R. W. Rosenthal, "A class of games possessing pure-strategy Nash
     equilibria," Int. J. Game Theory 2(1), 1973.  (congestion games /
     exact potential)
[7]  D. Monderer and L. S. Shapley, "Potential games," Games and Economic
     Behavior 14(1), 1996.
[8]  H.-L. Choi, L. Brunet, J. P. How, "Consensus-based decentralized auctions
     for robust task allocation," IEEE Trans. Robotics 25(4), 2009.  (CBBA)
[8a] "Potential Game-Theoretic Analysis of a Market-Based Decentralized Task
     Allocation Algorithm," in Distributed Autonomous Robotic Systems,
     Springer Tracts in Advanced Robotics vol. 112, 2016.  (shows CBBA
     converges to a pure NE of a distributed welfare game with PoA = 1/2,
     PoS = 1)
[9]  Multi-robot prize-collecting / orienteering games measuring PoA as the
     ratio of worst-case equilibrium reward to a MILP optimum (e.g. R. Nagi
     et al., "Prize-collecting multi-agent orienteering: price of anarchy
     bounds"; and "Stochastic Prize-Collecting Games," 2025), reporting
     equilibria at ~87-95% of optimum.
[10] A. Vetta, "Nash equilibria in competitive societies, with applications to
     facility location, traffic routing and auctions," FOCS 2002.  (valid
     utility games; submodular welfare => PoA <= 2)

USAGE
-----
    python PoABench.py [--seeds 0,1,2,3,4] [--snapshots 6] [--K 6] [--out DIR]

K is the sub-game size (number of mutually-interacting robots whose roles are
jointly enumerated); 4^K joint actions, so keep K <= 7 for tractability.
"""

import sys, os, json, time, random, argparse, itertools
from unittest.mock import MagicMock
from collections import defaultdict

# headless pygame
_pg = MagicMock(); _pg.SRCALPHA = 0
for _m in ['pygame', 'pygame.display', 'pygame.font']:
    sys.modules.setdefault(_m, _pg)

import numpy as np
import importlib.util
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))


def _find(*names):
    for name in names:
        for folder in (_HERE, '/mnt/user-data/outputs', os.path.join(_HERE, '..')):
            p = os.path.join(folder, name)
            if os.path.exists(p):
                return os.path.abspath(p)
    raise FileNotFoundError(f"none of {names} found")


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


M = _load(_find('hetero_robot_fleet_sim.py', '2DFleetFrameworkI.py'), 'fleet')
Role = M.Role
ROLES = [Role.SCOUT, Role.SCAN, Role.LOITER, Role.RELAY]


# ── Stage-game extraction + exact PoA on a K-robot sub-game ─────────────────────
def _eligible_active(sim):
    """Mirror the best-response eligibility filter in _pg_best_response_roles."""
    t = sim.timestep; shadow = sim.radio_shadow
    out = []
    for r in sim.robots:
        if not r.active:
            continue
        if t < getattr(r, 'role_locked_until', 0):
            continue
        if shadow[r.pos[0], r.pos[1]] and not sim.relay_ok_extended(
                sim.cell_to_zone(r.pos[0], r.pos[1])):
            continue           # stranded in uncovered shadow — must evacuate, not play
        out.append(r)
    return out


def _build_cluster_info(sim):
    clusters = getattr(sim, '_last_clusters', []) or []
    info = {}
    for cid, cl in enumerate(clusters):
        covered = any(sim._relay_ok_flood.get(z, False) for z in cl)
        info[cid] = (cl, covered)
    return info


def _select_subgame(sim, eligible, K):
    """Pick K mutually-interacting robots: the K eligible robots closest to a
    randomly-chosen building (shadow) cluster, so their role choices genuinely
    couple (relay public good + same-capability congestion). Falls back to the
    K nearest-together robots if no building is present."""
    if len(eligible) <= K:
        return eligible
    clusters = getattr(sim, '_last_clusters', []) or []
    anchor = None
    stair = [cl for cl in clusters
             if sim._shadow_zone_type.get(cl[0]) in ('stair', 'disc')]
    if stair:
        cl = random.choice(stair)
        zx, zy = cl[0]
        anchor = (zx * sim.zone_w_cells + sim.zone_w_cells // 2,
                  zy * sim.zone_h_cells + sim.zone_h_cells // 2)
    if anchor is None:
        anchor = eligible[0].pos
    eligible = sorted(eligible,
                      key=lambda r: abs(r.pos[0] - anchor[0]) + abs(r.pos[1] - anchor[1]))
    return eligible[:K]


def measure_stage_poa(sim, K):
    """Enumerate the exact K-robot role sub-game using the REAL utility.
    Returns a dict of efficiency quantities, or None if the sub-game is degenerate
    (fewer than 2 eligible robots, or non-positive optimum welfare)."""
    active = [r for r in sim.robots if r.active]
    eligible = _eligible_active(sim)
    if len(eligible) < 2:
        return None
    subset = _select_subgame(sim, eligible, K)
    k = len(subset)
    if k < 2:
        return None
    info = _build_cluster_info(sim)
    saved = {r.name: r.role for r in subset}

    def util(r):
        return float(sim._role_utility_pg(r, r.role, active, r.task_zone, info))

    def set_assignment(asn):
        for r, a in zip(subset, asn):
            r.role = a

    def welfare(asn):
        set_assignment(asn)
        return sum(util(r) for r in subset)

    def is_nash(asn):
        set_assignment(asn)
        for i, r in enumerate(subset):
            base = util(r)
            for alt in ROLES:
                if alt == asn[i]:
                    continue
                r.role = alt
                u = util(r)
                r.role = asn[i]
                if u > base + 1e-9:
                    return False
        return True

    # Exhaustive enumeration: social optimum + full pure-NE set
    best_W = -1e18
    ne_welfares = []
    for asn in itertools.product(ROLES, repeat=k):
        W = welfare(asn)
        if W > best_W:
            best_W = W
        if is_nash(asn):
            ne_welfares.append(W)

    # Equilibrium our own best-response dynamics reach (random start, like the sim)
    br_W = _local_best_response(sim, subset, util)

    # restore original roles — leave sim state untouched
    for r in subset:
        r.role = saved[r.name]

    if not ne_welfares:
        return None  # no pure NE found in sub-game (should not happen for a potential game)
    worst_ne = min(ne_welfares); best_ne = max(ne_welfares)

    out = {
        'k': k, 'n_ne': len(ne_welfares),
        'W_opt': best_W, 'W_worst_ne': worst_ne, 'W_best_ne': best_ne, 'W_br': br_W,
    }
    # Ratios are only meaningful when welfare is positive (standard PoA assumption).
    if best_W > 1e-9 and worst_ne > 1e-9:
        out['PoA'] = best_W / worst_ne
        out['PoS'] = best_W / best_ne if best_ne > 1e-9 else None
        out['eff_br'] = (br_W / best_W) if best_W > 1e-9 else None   # realised efficiency
        out['ratio_ok'] = True
    else:
        out['ratio_ok'] = False
    return out


def _local_best_response(sim, subset, util, max_iter=25):
    """Run the same kind of asynchronous best-response the sim uses, scoped to the
    sub-game, from a random initial role profile.  Returns the welfare of the
    equilibrium reached."""
    for r in subset:
        r.role = random.choice(ROLES)
    for _ in range(max_iter):
        changed = False
        order = subset[:]; random.shuffle(order)
        for r in order:
            best_role = r.role; best_u = util(r)
            for alt in ROLES:
                if alt == r.role:
                    continue
                cur = r.role; r.role = alt; u = util(r); r.role = cur
                if u > best_u + 1e-9:
                    best_u = u; best_role = alt
            if best_role != r.role:
                r.role = best_role; changed = True
        if not changed:
            break
    return sum(util(r) for r in subset)


# ── Main ────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=str, default='0,1,2,3,4')
    ap.add_argument('--snapshots', type=int, default=6,
                    help='stage games sampled per seed (at spread-out ticks)')
    ap.add_argument('--K', type=int, default=6, help='sub-game size (4^K actions)')
    ap.add_argument('--warmup', type=int, default=40,
                    help='steps before first snapshot (let belief/relays form)')
    ap.add_argument('--spacing', type=int, default=35, help='steps between snapshots')
    ap.add_argument('--out', type=str, default=os.path.join(_HERE, 'poa_results'))
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(',')]
    os.makedirs(args.out, exist_ok=True)

    print(f"\nPrice-of-Anarchy benchmark — Fleet role-selection stage game")
    print(f"seeds={seeds}  snapshots/seed={args.snapshots}  K={args.K} "
          f"(4^{args.K}={4**args.K} joint actions)  → exact enumeration")
    print("-" * 70)

    records = []
    t0 = time.time()
    for seed in seeds:
        random.seed(seed); np.random.seed(seed)
        sim = M.FleetSim()
        # step to warmup, then sample stage games at spaced ticks
        step = 0
        next_snap = args.warmup
        taken = 0
        while taken < args.snapshots and step < args.warmup + args.spacing * args.snapshots + 5:
            sim.step(); step += 1
            if step >= next_snap:
                rec = measure_stage_poa(sim, args.K)
                next_snap += args.spacing
                if rec is not None:
                    rec['seed'] = seed; rec['step'] = step
                    cov = float(np.mean(sim.union_belief != M.T_UNKNOWN))
                    rec['cov'] = round(100 * cov, 1)
                    records.append(rec)
                    taken += 1
                    tag = (f"PoA={rec['PoA']:.3f} PoS={rec['PoS']:.3f} "
                           f"eff_BR={rec['eff_br']:.3f}") if rec.get('ratio_ok') else \
                          "(welfare<=0, ratio skipped)"
                    print(f"  seed {seed} step {step:4d} cov={rec['cov']:4.0f}%  "
                          f"k={rec['k']} NE={rec['n_ne']:3d}  {tag}")
    wall = time.time() - t0

    ratio = [r for r in records if r.get('ratio_ok')]
    print("-" * 70)
    if ratio:
        poa = np.array([r['PoA'] for r in ratio])
        pos = np.array([r['PoS'] for r in ratio if r['PoS'] is not None])
        effb = np.array([r['eff_br'] for r in ratio if r['eff_br'] is not None])
        print(f"  stage games measured: {len(ratio)}  (over {len(seeds)} seeds)")
        print(f"  Price of Anarchy (W*/worst-NE):  mean={poa.mean():.3f}  "
              f"median={np.median(poa):.3f}  worst={poa.max():.3f}  "
              f"(theory: <=2 for valid-utility/submodular [10])")
        print(f"  Price of Stability (W*/best-NE): mean={pos.mean():.3f}  "
              f"worst={pos.max():.3f}  (theory: =1 for CBBA welfare game [8a])")
        print(f"  Realised efficiency (W_BR/W*):   mean={effb.mean():.3f}  "
              f"min={effb.min():.3f}  (1.0 = our dynamics reach the optimum)")
        frac_opt = float(np.mean(np.isclose(effb, 1.0, atol=1e-6)))
        print(f"  fraction of stage games where BR reaches the social optimum: "
              f"{100*frac_opt:.0f}%")
    else:
        print("  No positive-welfare stage games sampled — try a larger instance "
              "or more snapshots.")

    summary = {
        'config': {'seeds': seeds, 'snapshots_per_seed': args.snapshots,
                   'K': args.K, 'joint_actions': 4 ** args.K,
                   'warmup': args.warmup, 'spacing': args.spacing},
        'n_stage_games': len(records),
        'n_with_ratio': len(ratio),
        'wall_s': round(wall, 1),
    }
    if ratio:
        poa = [r['PoA'] for r in ratio]
        pos = [r['PoS'] for r in ratio if r['PoS'] is not None]
        effb = [r['eff_br'] for r in ratio if r['eff_br'] is not None]
        summary['PoA'] = {'mean': float(np.mean(poa)), 'median': float(np.median(poa)),
                          'worst': float(np.max(poa)), 'std': float(np.std(poa))}
        summary['PoS'] = {'mean': float(np.mean(pos)), 'worst': float(np.max(pos))}
        summary['efficiency_BR'] = {'mean': float(np.mean(effb)), 'min': float(np.min(effb))}
        summary['theory_bounds'] = {
            'valid_utility_submodular_PoA_le': 2.0,
            'affine_atomic_congestion_PoA': 2.5,
            'CBBA_welfare_game_PoA': 0.5, 'CBBA_welfare_game_PoS': 1.0,
        }
    summary['records'] = records
    with open(os.path.join(args.out, 'poa_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=float)

    # ── plots ──
    if ratio:
        poa = np.array([r['PoA'] for r in ratio])
        effb = np.array([r['eff_br'] for r in ratio if r['eff_br'] is not None])
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        axes[0].hist(poa, bins=min(20, max(4, len(poa)//2)), color='#2166ac',
                     alpha=0.8, edgecolor='white')
        axes[0].axvline(2.0, color='red', ls='--', label='valid-utility bound (2.0) [10]')
        axes[0].axvline(2.5, color='orange', ls=':', label='affine congestion (2.5) [4,5]')
        axes[0].axvline(poa.mean(), color='black', ls='-', lw=1,
                        label=f'mean {poa.mean():.2f}')
        axes[0].set_xlabel('Price of Anarchy  (W* / worst-NE)')
        axes[0].set_ylabel('stage games'); axes[0].legend(fontsize=8)
        axes[0].set_title('Empirical PoA distribution'); axes[0].grid(alpha=0.3)

        axes[1].hist(effb, bins=min(20, max(4, len(effb)//2)), color='#4dac26',
                     alpha=0.8, edgecolor='white')
        axes[1].axvline(0.5, color='red', ls='--', label='½-optimal (CBBA PoA bound) [8a]')
        axes[1].axvline(effb.mean(), color='black', ls='-', lw=1,
                        label=f'mean {effb.mean():.2f}')
        axes[1].set_xlabel('Realised efficiency  (W_BR / W*)')
        axes[1].set_ylabel('stage games'); axes[1].legend(fontsize=8)
        axes[1].set_title('Efficiency of reached equilibria'); axes[1].grid(alpha=0.3)
        fig.suptitle('Fleet role-selection stage game — efficiency vs exact optimum',
                     fontsize=12)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, 'poa_distribution.png'), dpi=140)
        plt.close(fig)

    print(f"\n  saved -> {args.out}/poa_summary.json"
          f"{' , poa_distribution.png' if ratio else ''}")
    print(f"  wall {wall:.0f}s")


if __name__ == '__main__':
    main()