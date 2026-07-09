"""CARA-EL: Capability-Aware Relay-constrained Assignment with Execution Layer."""


import random, time
from collections import deque
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csc_matrix

ALLOCATION_CADENCE = 50
ZONE_CAPACITY      = 2
MAX_BUNDLE         = 8


# -----------------------------------------------------------------------------
# CARABot
# -----------------------------------------------------------------------------
class CARABot:
    __slots__ = (
        'name', 'pos', 'caps_mask', 'caps', 'world', 'sim',
        'temp_limit', 'rad_limit',
        'terrain_belief', 'known_mask', 'temp_belief', 'rad_belief',
        'chunked', 'active', 'battery', 'death_reason', 'hazard_killed',
        'goal', 'path', 'stuck_steps', 'failed_goals',
        'dose_T', 'dose_R', 'terrain_R',
        '_reachable_arr', '_reachable_tick',
        'scan_age', 'confidence',
        'outbox', 'inbox', '_inbox_dirty', '_scan_dirty',
        'personally_scanned',
        'assigned_zones',
        'is_relay',
        'relay_cluster',
        # -- execution-layer state --
        'exec_state',        # 'waiting'|'entering'|'inside'|'ejecting'|'free'
        'exit_goal',         # border cell to head for when ejecting
        'role', 'task_zone',
    )

    def __init__(self, name, x, y, caps, caps_mask, world, sim,
                 temp_limit, rad_limit):
        self.name=name; self.pos=(x,y); self.caps=caps; self.caps_mask=caps_mask
        self.world=world; self.sim=sim
        self.temp_limit=temp_limit; self.rad_limit=rad_limit
        M=sim.M
        self.terrain_belief=np.full((M.GRID_W,M.GRID_H),M.T_UNKNOWN,dtype=np.uint8)
        self.known_mask=np.zeros((M.GRID_W,M.GRID_H),dtype=bool)
        self.temp_belief=np.full((M.GRID_W,M.GRID_H),np.nan,dtype=np.float32)
        self.rad_belief=np.full((M.GRID_W,M.GRID_H),np.nan,dtype=np.float32)
        self.chunked=np.zeros((2,M.GRID_W//M.CHUNK_SIZE,M.GRID_H//M.CHUNK_SIZE),dtype=np.float32)
        self.scan_age=np.full((M.GRID_W,M.GRID_H),32767,dtype=np.int16)
        self.confidence=np.zeros((M.GRID_W,M.GRID_H),dtype=np.float32)
        self.personally_scanned=np.zeros((M.GRID_W,M.GRID_H),dtype=bool)
        self.outbox=[]; self.inbox=[]; self._inbox_dirty=False; self._scan_dirty=False
        self.active=True; self.battery=M.MAX_BATTERY
        self.death_reason=None; self.hazard_killed=False
        self.goal=None; self.path=[]; self.stuck_steps=0; self.failed_goals={}
        self.dose_T=0.0; self.dose_R=0.0
        self.terrain_R=max(3,round(24/M.CELL_SIZE))
        self._reachable_arr=None; self._reachable_tick=-999
        self.assigned_zones=[]; self.is_relay=False; self.relay_cluster=-1
        self.exec_state='free'; self.exit_goal=None
        _Role=getattr(M,'Role',None)
        self.role=_Role.SCAN if _Role else None
        self.task_zone=None
        self._scan()

    def _scan(self):
        M=self.sim.M; x0,y0=self.pos; R=self.terrain_R
        W,H=self.world.w,self.world.h
        inside=(self.world.grid[x0][y0]["t"]==M.T_STAIRS)
        for dx in range(-R,R+1):
            for dy in range(-R,R+1):
                if dx*dx+dy*dy>R*R: continue
                nx,ny=x0+dx,y0+dy
                if not (0<=nx<W and 0<=ny<H): continue
                if (dx or dy) and not self.sim._has_los(x0,y0,nx,ny,inside): continue
                self.personally_scanned[nx,ny]=True
                self.scan_age[nx,ny]=0; self.confidence[nx,ny]=1.0
                if not self.known_mask[nx,ny]:
                    self.known_mask[nx,ny]=True
                    self.terrain_belief[nx,ny]=self.world.grid[nx][ny]["t"]
                    self.temp_belief[nx,ny]=self.world.grid[nx][ny]["temp"]
                    self.rad_belief[nx,ny]=self.world.grid[nx][ny]["rad"]

    def reachable(self):
        M=self.sim.M; t=self.sim.timestep
        if self._reachable_tick==t and self._reachable_arr is not None:
            return self._reachable_arr
        from scipy import ndimage as _ndi
        W,H=self.world.w,self.world.h; tb=self.terrain_belief
        mask=self.caps_mask; mask4=mask&0xF
        passable=np.zeros((W,H),dtype=bool)
        for tc in range(6):
            if M._TRAV_LUT[tc][mask4]: passable|=(tb==tc)
        is_land=bool(mask&M.CAP_LAND) and not bool(mask&(M.CAP_AIR|M.CAP_WATER))
        is_boat=bool(mask&M.CAP_WATER) and not bool(mask&M.CAP_AIR)
        if is_land:
            unk=(tb==M.T_UNKNOWN)
            wn=_ndi.binary_dilation(tb==M.T_WATER,structure=np.ones((3,3),dtype=bool))
            bn=_ndi.binary_dilation(tb==M.T_BRIDGE,structure=np.ones((3,3),dtype=bool))
            passable&=~(unk&wn&~bn)
            if (self.world.grid[self.pos[0]][self.pos[1]]["t"]==M.T_STAIRS
                    and bool(mask&(M.CAP_STAIRS|M.CAP_AIR))):
                sn=_ndi.binary_dilation(tb==M.T_STAIRS,structure=np.ones((3,3),dtype=bool))
                passable|=(unk&sn)
        elif is_boat:
            unk=(tb==M.T_UNKNOWN)
            wbn=_ndi.binary_dilation((tb==M.T_WATER)|(tb==M.T_BRIDGE),
                                      structure=np.ones((3,3),dtype=bool))
            blk=unk&~wbn; blk[self.pos[0],self.pos[1]]=False; passable&=~blk
        can_enter=bool(mask&(M.CAP_STAIRS|M.CAP_AIR))
        shd=self.sim.radio_shadow; brd=self.sim._shadow_border_mask_cache
        interior=shd&~brd
        if np.any(interior):
            rok=self.sim._relay_ok
            passable&=~(interior&~rok) if can_enter else ~interior
        sx,sy=self.pos
        if not passable[sx,sy]:
            arr=np.zeros((W,H),dtype=bool); arr[sx,sy]=True
        else:
            from scipy.ndimage import label as _lbl
            lab,_=_lbl(passable); arr=(lab==lab[sx,sy])
        self._reachable_arr=arr; self._reachable_tick=t; return arr

    def _frontier_in_zone(self, zone):
        M=self.sim.M; W,H=self.world.w,self.world.h
        zx,zy=zone
        x0=zx*self.sim.zone_w_cells; x1=min(x0+self.sim.zone_w_cells,W)
        y0=zy*self.sim.zone_h_cells; y1=min(y0+self.sim.zone_h_cells,H)
        union=self.sim.union_belief; reach=self.reachable()
        vis=np.zeros((W,H),dtype=bool); q=deque([self.pos]); vis[self.pos[0],self.pos[1]]=True
        while q:
            cx,cy=q.popleft()
            in_zone=(x0<=cx<x1 and y0<=cy<y1)
            if in_zone and self.sim.timestep>=self.failed_goals.get((cx,cy),0):
                if union[cx,cy]==M.T_UNKNOWN and reach[cx,cy]:
                    return (cx,cy)
                if (union[cx,cy]!=M.T_UNKNOWN and reach[cx,cy]
                        and any(union[nx2,ny2]==M.T_UNKNOWN
                                for nx2,ny2 in self.world.neighbours((cx,cy)))):
                    return (cx,cy)
            for nx2,ny2 in self.world.neighbours((cx,cy)):
                if not vis[nx2,ny2] and reach[nx2,ny2]:
                    vis[nx2,ny2]=True; q.append((nx2,ny2))
        return None

    def _border_goal_for_cluster(self, cluster_zones):
        """Nearest shadow-border cell adjacent to cluster."""
        M=self.sim.M; W,H=self.world.w,self.world.h
        border=self.sim._shadow_border_mask_cache; reach=self.reachable()
        cz_set=set(cluster_zones); best=None; best_d=1e9
        border_pts=np.argwhere(border)
        for pt in border_pts:
            bx,by=int(pt[0]),int(pt[1])
            if not reach[bx,by]: continue
            for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
                nx,ny=bx+dx,by+dy
                if (0<=nx<W and 0<=ny<H and self.sim.radio_shadow[nx,ny]
                        and self.sim.cell_to_zone(nx,ny) in cz_set):
                    d=abs(bx-self.pos[0])+abs(by-self.pos[1])
                    if d<best_d: best_d=d; best=(bx,by)
                    break
        return best

    def _nearest_exit(self):
        """Nearest reachable shadow-border cell."""
        border=self.sim._shadow_border_mask_cache
        reach=self.reachable()
        best=None; best_d=1e9
        for pt in np.argwhere(border):
            bx,by=int(pt[0]),int(pt[1])
            if not reach[bx,by]: continue
            d=abs(bx-self.pos[0])+abs(by-self.pos[1])
            if d<best_d: best_d=d; best=(bx,by)
        return best

    def _nearest_exit_bfs(self):
        """
        BFS to nearest shadow-border cell ignoring relay_ok — used during eject
        so a robot always finds a way out even when coverage is dropped.
        Works from the robot's current position regardless of shadow state.
        """
        M=self.sim.M; W,H=self.world.w,self.world.h
        shadow=self.sim.radio_shadow; border=self.sim._shadow_border_mask_cache
        tb=self.terrain_belief; mask=self.caps_mask; mask4=mask&0xF
        # Build traversability ignoring shadow gate
        passable=np.zeros((W,H),dtype=bool)
        for tc in range(6):
            if M._TRAV_LUT[tc][mask4]: passable|=(tb==tc)
        passable[self.pos[0],self.pos[1]]=True  # always start from current pos
        vis=np.zeros((W,H),dtype=bool); q=deque([self.pos])
        vis[self.pos[0],self.pos[1]]=True
        while q:
            cx,cy=q.popleft()
            if border[cx,cy]: return (cx,cy)  # first border cell found = nearest
            for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
                nx,ny=cx+dx,cy+dy
                if 0<=nx<W and 0<=ny<H and not vis[nx,ny] and passable[nx,ny]:
                    vis[nx,ny]=True; q.append((nx,ny))
        return None

    def _plan_to_ignoring_shadow(self, goal):
        """A* to goal with shadow gate disabled — for emergency exit routing."""
        M=self.sim.M
        _zc=np.zeros((2,M.GRID_W//M.CHUNK_SIZE,M.GRID_H//M.CHUNK_SIZE),dtype=np.float32)
        _zt=np.zeros((M.GRID_W,M.GRID_H),dtype=np.uint16)
        _no_shadow=np.zeros((M.GRID_W,M.GRID_H),dtype=bool)  # blank shadow = no gate
        path=M.AStar.search(
            start=self.pos, goal=goal, caps_mask=self.caps_mask,
            terrain_u8=self.terrain_belief,
            temp_f32=self.temp_belief, rad_f32=self.rad_belief,
            chunked_risk=_zc, temp_limit=9999.0, rad_limit=9999.0,
            radio_shadow=_no_shadow, relay_ok_fn=lambda z:True,
            cell_to_zone_fn=self.sim.cell_to_zone,
            global_cov=self.sim.global_cov, unk_pen=0.3, info_w=0.0, unk_prior=0.25,
            alpha_mult=0.0, beta_mult=0.0, soft_frac=1.0,
            traffic_u16=_zt, traffic_w=0.0)
        if not path: self.failed_goals[goal]=self.sim.timestep+30; return False
        self.goal=goal; self.path=path; return True

    def _plan_to(self, goal):
        M=self.sim.M
        _zc=np.zeros((2,M.GRID_W//M.CHUNK_SIZE,M.GRID_H//M.CHUNK_SIZE),dtype=np.float32)
        _zt=np.zeros((M.GRID_W,M.GRID_H),dtype=np.uint16)
        can_enter=bool(self.caps_mask&(M.CAP_STAIRS|M.CAP_AIR))
        rok=self.sim._relay_ok; gx,gy=goal
        path=M.AStar.search(
            start=self.pos, goal=goal, caps_mask=self.caps_mask,
            terrain_u8=self.terrain_belief,
            temp_f32=self.temp_belief, rad_f32=self.rad_belief,
            chunked_risk=_zc, temp_limit=9999.0, rad_limit=9999.0,
            radio_shadow=self.sim.radio_shadow,
            relay_ok_fn=(lambda z:bool(rok[gx,gy])) if can_enter else (lambda z:False),
            cell_to_zone_fn=self.sim.cell_to_zone,
            global_cov=self.sim.global_cov, unk_pen=0.3, info_w=0.1, unk_prior=0.25,
            alpha_mult=0.0, beta_mult=0.0, soft_frac=1.0,
            traffic_u16=_zt, traffic_w=0.0,
            shadow_border=self.sim._shadow_border_mask_cache)
        if not path: self.failed_goals[goal]=self.sim.timestep+60; return False
        self.goal=goal; self.path=path; return True

    def _cluster_relay_ok(self):
        """True if relay coverage exists for this robot's assigned stair cluster."""
        if not self.assigned_zones: return False
        M=self.sim.M
        for z in self.assigned_zones:
            if self.sim._shadow_zone_type.get(z,'none')=='stair':
                cid=self.sim._cara_cluster_id.get(z,-1)
                if cid>=0:
                    czones=[z2 for z2,c in self.sim._cara_cluster_id.items() if c==cid]
                    if any(self.sim._relay_ok_flood.get(z2,False) for z2 in czones):
                        return True
        # No stair zones assigned - open terrain, always ok
        return not any(self.sim._shadow_zone_type.get(z,'none')=='stair'
                       for z in self.assigned_zones)

    def tick(self, occupied):
        M=self.sim.M
        if not self.active or self.battery<=0: self.active=False; return

        # -- Relay role: navigate to cluster border and hold ------------------
        if self.is_relay and self.relay_cluster>=0:
            czones=[z for z,c in self.sim._cara_cluster_id.items() if c==self.relay_cluster]
            if not self.path or self.pos==self.goal:
                tgt=self._border_goal_for_cluster(czones)
                if tgt and self._plan_to(tgt):
                    pass
                else:
                    # Cannot reach this cluster's border — demote to explorer
                    self.is_relay=False; self.relay_cluster=-1
                    _Role=getattr(self.sim.M,'Role',None)
                    self.role=_Role.SCAN if _Role else None
                    self.exec_state='waiting'; self.path=[]; self.goal=None

        # -- Explorer ---------------------------------------------------------
        else:
            in_shadow=self.sim.radio_shadow[self.pos[0],self.pos[1]]
            covered=self.sim._relay_ok[self.pos[0],self.pos[1]]

            if not self.sim.use_exec_layer:
                # CARA Base: no hold gate, no eject — go directly to assigned zone
                if not self.assigned_zones: return
                if not self.path or self.pos==self.goal:
                    tgt=None
                    for z in list(self.assigned_zones):
                        tgt=self._frontier_in_zone(z)
                        if tgt: break
                        else: self.assigned_zones.remove(z)
                    if tgt is None: return
                    self._plan_to(tgt)

            else:
                # CARA Dynamic: full execution-layer state machine

                # EJECT: relay lost while inside --------------------------------
                if in_shadow and not covered and self.exec_state=='inside':
                    self.exec_state='ejecting'
                    self.path=[]; self.goal=None; self.exit_goal=None
                    exit_cell=self._nearest_exit_bfs()
                    if exit_cell:
                        self.exit_goal=exit_cell
                        self._plan_to_ignoring_shadow(exit_cell)

                # Currently ejecting -------------------------------------------
                elif self.exec_state=='ejecting':
                    if not in_shadow or covered:
                        self.exec_state='waiting'; self.exit_goal=None
                        self.path=[]; self.goal=None
                    elif not self.path:
                        exit_cell=self._nearest_exit_bfs()
                        if exit_cell:
                            self.exit_goal=exit_cell
                            self._plan_to_ignoring_shadow(exit_cell)

                # HOLD GATE: waiting for relay confirmation --------------------
                elif self.exec_state in ('waiting','free'):
                    has_stair=any(self.sim._shadow_zone_type.get(z,'none')=='stair'
                                  for z in self.assigned_zones)
                    if has_stair:
                        if self._cluster_relay_ok():
                            self.exec_state='entering'; self.path=[]; self.goal=None
                        else:
                            tgt=None
                            for z in list(self.assigned_zones):
                                if self.sim._shadow_zone_type.get(z,'none')=='stair':
                                    continue
                                tgt=self._frontier_in_zone(z)
                                if tgt: break
                            if tgt and (not self.path or self.pos==self.goal):
                                self._plan_to(tgt)
                            elif not tgt:
                                self.path=[]; self.goal=None
                    else:
                        self.exec_state='entering'

                # Entering / inside: normal exploration -----------------------
                elif self.exec_state=='entering':
                    if in_shadow: self.exec_state='inside'
                    if not self.assigned_zones: self.exec_state='free'; return
                    if not self.path or self.pos==self.goal:
                        tgt=None
                        for z in list(self.assigned_zones):
                            tgt=self._frontier_in_zone(z)
                            if tgt: break
                            else: self.assigned_zones.remove(z)
                        if tgt is None: self.exec_state='free'; return
                        self._plan_to(tgt)

                elif self.exec_state=='inside':
                    if in_shadow and not covered:
                        self.exec_state='ejecting'; self.path=[]; self.goal=None
                        self.exit_goal=None
                        exit_cell=self._nearest_exit_bfs()
                        if exit_cell:
                            self.exit_goal=exit_cell
                            self._plan_to_ignoring_shadow(exit_cell)
                    else:
                        if not self.assigned_zones: self.exec_state='free'; return
                        if not self.path or self.pos==self.goal:
                            tgt=None
                            for z in list(self.assigned_zones):
                                tgt=self._frontier_in_zone(z)
                                if tgt: break
                                else: self.assigned_zones.remove(z)
                            if tgt is None: self.exec_state='free'; return
                            self._plan_to(tgt)

        # -- Move one step ----------------------------------------------------
        if not self.path: return
        nc=self.path[0]; tt=self.world.grid[nc[0]][nc[1]]["t"]
        if tt==M.T_OBS: self.path=[]; self.goal=None; return
        is_land_only=not bool(self.caps_mask&(M.CAP_AIR|M.CAP_WATER))
        if is_land_only and tt==M.T_WATER:
            self.terrain_belief[nc[0],nc[1]]=tt; self.known_mask[nc[0],nc[1]]=True
            self.path=[]; self.goal=None; return
        is_boat=bool(self.caps_mask&M.CAP_WATER) and not bool(self.caps_mask&M.CAP_AIR)
        if is_boat and tt not in (M.T_WATER,M.T_BRIDGE):
            self.terrain_belief[nc[0],nc[1]]=tt; self.known_mask[nc[0],nc[1]]=True
            self.path=[]; self.goal=None; return
        occupied.discard(self.pos); self.pos=self.path.pop(0); occupied.add(self.pos)
        self._scan()
        drain={"Legged":1.0,"Drone":2.0,"Boat":2.0,"Rover":0.4}
        rt=next((t for t in ("Legged","Drone","Boat","Rover") if self.name.startswith(t)),"Legged")
        self.battery-=drain.get(rt,1.0)
        c=self.world.grid[self.pos[0]][self.pos[1]]
        self.dose_T+=max(0.0,c["temp"])*0.01; self.dose_R+=max(0.0,c["rad"])*0.01
        if c["temp"] > self.temp_limit or c["rad"] > self.rad_limit:
            reasons = []
            if c["temp"] > self.temp_limit: reasons.append(f"temp({c['temp']:.0f}>{self.temp_limit:.0f})")
            if c["rad"]  > self.rad_limit:  reasons.append(f"rad({c['rad']:.0f}>{self.rad_limit:.0f})")
            self.active = False; self.hazard_killed = True
            self.death_reason = " & ".join(reasons)
        if self.battery<=0: self.active=False; self.death_reason="battery depleted"


# -----------------------------------------------------------------------------
# CARASim
# -----------------------------------------------------------------------------
def make_cara_sim(gnf_module, M_module, use_exec_layer=True):
    """
    Returns a CARASim class.

    use_exec_layer=True  (CARA Dynamic):
        Adds three runtime safety checks every tick:
        - Hold gate: explorers wait at shadow border until relay is physically confirmed
        - Eject: explorers exit immediately if relay coverage drops mid-exploration
        - Emergency re-election: nearest robot promoted to relay without waiting
          for the next MILP cadence when explorers are stranded

    use_exec_layer=False  (CARA Base):
        Pure MILP allocation. Robots navigate directly to assigned zones with
        no per-tick safety checking. Relay constraints are enforced logically at
        solve time only — if a relay hasn't arrived yet, explorers may enter
        shadow uncovered and become temporarily stranded.
    """
    GNFSim=gnf_module.GNFSim

    class CARASim(GNFSim):

        def _build_robots(self):
            super()._build_robots()
            self.robots=[
                CARABot(r.name,r.pos[0],r.pos[1],r.caps,r.caps_mask,
                        self.world,self,r.temp_limit,r.rad_limit)
                for r in self.robots]
            self._build_cara_clusters()
            self.milp_solve_times=[]
            self.last_milp_time_ms=0.0
            self.use_exec_layer=use_exec_layer   # stored so CARABot.tick can read it
            self.eject_events=[]
            self.hold_ticks={}

        def _build_cara_clusters(self):
            # Build clusters for BOTH stair and disc shadow zones.
            # Each connected component of same-type shadow zones becomes one cluster.
            # Disc and stair are always separate clusters (never merged) because
            # they have different relay requirements (land vs any robot).
            self._cara_cluster_id={}
            self._cara_cluster_type={}   # cid -> 'stair' | 'disc'
            visited=set(); cid=0
            for shadow_type in ('stair', 'disc'):
                type_zones={z for z,t in self._shadow_zone_type.items() if t==shadow_type}
                for sz in sorted(type_zones):
                    if sz in visited: continue
                    q=deque([sz]); visited.add(sz)
                    while q:
                        z2=q.popleft(); self._cara_cluster_id[z2]=cid
                        for nz in self.zone_neighbors4(z2):
                            if nz not in visited and nz in type_zones:
                                visited.add(nz); q.append(nz)
                    self._cara_cluster_type[cid]=shadow_type
                    cid+=1
            self._cara_n_clusters=cid

        def _solve_milp(self):
            M=self.M
            robots=[r for r in self.robots if r.active and r.battery>0]
            zones=[(zx,zy) for zx in range(self.zone_nx)
                           for zy in range(self.zone_ny)]
            R=len(robots); Z=len(zones); C=self._cara_n_clusters
            if R==0: return {}, {}, 0.0

            zuf={}
            for z in zones:
                zx,zy=z; x0=zx*self.zone_w_cells; x1=min(x0+self.zone_w_cells,M.GRID_W)
                y0=zy*self.zone_h_cells; y1=min(y0+self.zone_h_cells,M.GRID_H)
                zuf[z]=float(np.mean(self.union_belief[x0:x1,y0:y1]==M.T_UNKNOWN))

            n_x=R*Z; n_y=R*C; n_vars=n_x+n_y
            c_vec=np.zeros(n_vars)
            for ri,r in enumerate(robots):
                can_stair=bool(r.caps_mask&(M.CAP_STAIRS|M.CAP_AIR))
                can_relay=not can_stair
                for zi,z in enumerate(zones):
                    zt=self._shadow_zone_type.get(z,'none')
                    is_shadow_zone=(zt in ('stair','disc'))
                    # cap_ok: stair zones require stair/air capability
                    cap_ok=(zt!='stair') or can_stair
                    # relay_ok_now: check any cell in zone area covered, not just centre
                    if is_shadow_zone:
                        zx2,zy2=z
                        x0z=zx2*self.zone_w_cells; x1z=min(x0z+self.zone_w_cells,M.GRID_W)
                        y0z=zy2*self.zone_h_cells; y1z=min(y0z+self.zone_h_cells,M.GRID_H)
                        relay_ok_now=bool(np.any(self._relay_ok[x0z:x1z,y0z:y1z]))
                    else:
                        relay_ok_now=True
                    shd_factor=1.0 if (not is_shadow_zone or relay_ok_now or can_stair) else 0.1
                    c_vec[ri*Z+zi]=-(zuf[z]*cap_ok*shd_factor)
                for ci in range(C):
                    ct=self._cara_cluster_type.get(ci,'stair')
                    # Stair clusters: only land-only robots (Rover) as relay
                    # Disc clusters: any non-stair-capable (Rover, Boat) as relay
                    relay_eligible = can_relay or (ct=='disc' and not can_stair)
                    c_vec[n_x+ri*C+ci]=-0.5 if relay_eligible else 0.0

            rows=[]; cols=[]; data=[]; b_lo_l=[]; b_up_l=[]; n_con=0
            for zi in range(Z):
                for ri in range(R):
                    rows.append(n_con); cols.append(ri*Z+zi); data.append(1.0)
                b_lo_l.append(-np.inf); b_up_l.append(float(ZONE_CAPACITY)); n_con+=1
            for ri in range(R):
                for zi in range(Z):
                    rows.append(n_con); cols.append(ri*Z+zi); data.append(1.0)
                for ci in range(C):
                    rows.append(n_con); cols.append(n_x+ri*C+ci); data.append(1.0)
                b_lo_l.append(-np.inf); b_up_l.append(float(MAX_BUNDLE)); n_con+=1
            for ri,r in enumerate(robots):
                can_stair=bool(r.caps_mask&(M.CAP_STAIRS|M.CAP_AIR))
                if not can_stair:
                    for zi,z in enumerate(zones):
                        if self._shadow_zone_type.get(z,'none')=='stair':
                            rows.append(n_con); cols.append(ri*Z+zi); data.append(1.0)
                            b_lo_l.append(-np.inf); b_up_l.append(0.0); n_con+=1
            z_to_idx={z:i for i,z in enumerate(zones)}
            for ri,r in enumerate(robots):
                can_stair=bool(r.caps_mask&(M.CAP_STAIRS|M.CAP_AIR))
                if not can_stair: continue
                for z,ci in self._cara_cluster_id.items():
                    if ci<0 or ci>=C: continue
                    zi=z_to_idx.get(z)
                    if zi is None: continue
                    rows.append(n_con); cols.append(ri*Z+zi); data.append(1.0)
                    for ri2 in range(R):
                        rows.append(n_con); cols.append(n_x+ri2*C+ci); data.append(-1.0)
                    b_lo_l.append(-np.inf); b_up_l.append(0.0); n_con+=1
            # Stair-capable robots (Legged, Drone) must NOT relay stair clusters
            # — they are the only types that can enter buildings, so relay duty
            # wastes their unique capability. They CAN relay disc clusters if needed.
            for ri,r in enumerate(robots):
                can_stair=bool(r.caps_mask&(M.CAP_STAIRS|M.CAP_AIR))
                if can_stair:
                    for ci in range(C):
                        if self._cara_cluster_type.get(ci,'stair')=='stair':
                            rows.append(n_con); cols.append(n_x+ri*C+ci); data.append(1.0)
                            b_lo_l.append(-np.inf); b_up_l.append(0.0); n_con+=1

            if not rows:
                return {r.name:[] for r in robots},{r.name:-1 for r in robots},0.0

            A=csc_matrix((data,(rows,cols)),shape=(n_con,n_vars))
            constraints=LinearConstraint(A,np.array(b_lo_l),np.array(b_up_l))
            t0=time.perf_counter()
            result=milp(c_vec,constraints=constraints,
                       integrality=np.ones(n_vars),bounds=Bounds(0,1))
            elapsed=(time.perf_counter()-t0)*1000

            assignment={r.name:[] for r in robots}
            relay={r.name:-1 for r in robots}
            if result.success and result.x is not None:
                x=result.x
                for ri,r in enumerate(robots):
                    for zi,z in enumerate(zones):
                        if x[ri*Z+zi]>0.5: assignment[r.name].append(z)
                    for ci in range(C):
                        if x[n_x+ri*C+ci]>0.5: relay[r.name]=ci
            return assignment,relay,elapsed

        def _apply_allocation(self, assignment, relay):
            """Apply a MILP solution to all robots.

            Stability rule: if a robot is currently serving as relay at its
            cluster's border (relay_ok_flood=True), do NOT reassign it even if
            the MILP chose a different robot this solve.  This prevents relay
            churning where a serving relay gets replaced every cadence tick,
            guaranteeing no relay holds its border long enough for explorers to
            actually enter.  The serving relay stays until:
              (a) the cluster is fully explored (all stair/disc zones done), OR
              (b) the robot's battery drops below 15%, OR
              (c) an explorer is inside and the relay is about to leave (safety).
            """
            M=self.M
            _Role=getattr(M,'Role',None)
            # Identify which clusters have a committed relay that should not
            # be interrupted.  Two cases warrant stability:
            #   A) Relay physically at border with relay_ok confirmed (serving)
            #   B) Relay en route (has a path, battery ok) — interrupting it
            #      wastes the travel already done and causes relay churning
            # A relay is only replaced if: battery critical (<15%), OR the
            # cluster is done (all zones explored), OR another relay is already
            # serving the same cluster (duplicate).
            serving_clusters={}  # cid -> robot_name
            for r in self.robots:
                if not r.active or not r.is_relay: continue
                if r.relay_cluster<0: continue
                cid=r.relay_cluster
                if cid in serving_clusters: continue  # another relay already committed
                czones=[z for z,c in self._cara_cluster_id.items() if c==cid]
                is_serving=any(self._relay_ok_flood.get(z,False) for z in czones)
                is_enroute=(r.path or r.goal is not None)  # has navigation intent
                committed=is_serving or is_enroute
                if committed:
                    batt_ok=(r.battery/M.MAX_BATTERY)>0.15
                    explorer_inside=any(
                        rr.active and not rr.is_relay
                        and self.radio_shadow[rr.pos[0],rr.pos[1]]
                        and self._cara_cluster_id.get(
                            self.cell_to_zone(rr.pos[0],rr.pos[1]),-1)==cid
                        for rr in self.robots
                    )
                    if batt_ok or explorer_inside:
                        serving_clusters[cid]=r.name

            for r in self.robots:
                if not r.active: continue
                was_relay=r.is_relay
                new_cluster=relay.get(r.name,-1)
                # Stability: if this robot is serving a cluster, keep it as relay
                # regardless of what the new MILP solve chose
                if was_relay and r.relay_cluster in serving_clusters:
                    if serving_clusters[r.relay_cluster]==r.name:
                        # Keep relay role; update zone assignments only
                        r.assigned_zones=list(assignment.get(r.name,[]))
                        r.task_zone=r.assigned_zones[0] if r.assigned_zones else None
                        continue  # do NOT change relay role
                r.assigned_zones=list(assignment.get(r.name,[]))
                r.relay_cluster=new_cluster
                r.is_relay=(r.relay_cluster>=0)
                r.role=(_Role.RELAY if r.is_relay else _Role.SCAN) if _Role else None
                r.task_zone=r.assigned_zones[0] if r.assigned_zones else None
                if r.is_relay and not was_relay:
                    r.goal=None; r.path=[]
                    r.exec_state='free'
                elif not r.is_relay and r.exec_state=='free':
                    r.exec_state='waiting'

        def _emergency_relay_election(self):
            """
            Execution-layer fix: when a relay is lost and explorers are inside
            shadow with no coverage, immediately promote the nearest available
            robot as relay - without waiting for the next MILP cadence.

            This is greedy (not globally optimal) but prevents indefinite
            stranding. The next scheduled MILP solve will produce a better
            global assignment.
            """
            M=self.M
            _Role=getattr(M,'Role',None)
            # Find clusters with explorers inside but no relay coverage
            for cid in range(self._cara_n_clusters):
                czones=[z for z,c in self._cara_cluster_id.items() if c==cid]
                already_covered=any(self._relay_ok_flood.get(z,False) for z in czones)
                if already_covered: continue
                # Any explorer inside this cluster's shadow?
                explorers_inside=[
                    r for r in self.robots
                    if r.active and not r.is_relay
                    and self.radio_shadow[r.pos[0],r.pos[1]]
                    and self._cara_cluster_id.get(
                        self.cell_to_zone(r.pos[0],r.pos[1]),-1)==cid
                ]
                if not explorers_inside: continue
                # Elect nearest non-relay, non-shadow robot as emergency relay
                cluster_cx=int(np.mean([z[0]*self.zone_w_cells+self.zone_w_cells//2
                                        for z in czones]))
                cluster_cy=int(np.mean([z[1]*self.zone_h_cells+self.zone_h_cells//2
                                        for z in czones]))
                candidates=[
                    r for r in self.robots
                    if r.active and not r.is_relay
                    and not self.radio_shadow[r.pos[0],r.pos[1]]
                    and not bool(r.caps_mask&(M.CAP_STAIRS|M.CAP_AIR))
                ]
                if not candidates:
                    # No ideal candidate - use any non-relay robot outside shadow
                    candidates=[
                        r for r in self.robots
                        if r.active and not r.is_relay
                        and not self.radio_shadow[r.pos[0],r.pos[1]]
                    ]
                if not candidates: continue
                best=min(candidates,key=lambda r:
                         abs(r.pos[0]-cluster_cx)+abs(r.pos[1]-cluster_cy))
                # Promote to relay
                best.is_relay=True
                best.relay_cluster=cid
                best.role=_Role.RELAY if _Role else None
                best.goal=None; best.path=[]
                best.exec_state='free'
                # Record eject events and trigger immediate exit planning
                for exp in explorers_inside:
                    self.eject_events.append((self.timestep, exp.name))
                    exp.exec_state='ejecting'
                    exp.path=[]; exp.goal=None; exp.exit_goal=None
                    exit_cell=exp._nearest_exit_bfs()
                    if exit_cell:
                        exp.exit_goal=exit_cell
                        exp._plan_to_ignoring_shadow(exit_cell)

        def step(self):
            self.timestep+=1
            M=self.M

            # -- Rebuild relay coverage ---------------------------------------
            border=self._shadow_border_mask_cache; shadow=self.radio_shadow
            relay_ok=np.zeros((M.GRID_W,M.GRID_H),dtype=bool)
            for r in self.robots:
                if not r.active: continue
                rx,ry=r.pos
                if not shadow[rx,ry] and border[rx,ry]:
                    q2=deque()
                    for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
                        nx2,ny2=rx+dx,ry+dy
                        if 0<=nx2<M.GRID_W and 0<=ny2<M.GRID_H and shadow[nx2,ny2]:
                            if not relay_ok[nx2,ny2]:
                                relay_ok[nx2,ny2]=True; q2.append((nx2,ny2))
                    while q2:
                        cx2,cy2=q2.popleft()
                        for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
                            nx2,ny2=cx2+dx,cy2+dy
                            if (0<=nx2<M.GRID_W and 0<=ny2<M.GRID_H
                                    and shadow[nx2,ny2] and not relay_ok[nx2,ny2]):
                                relay_ok[nx2,ny2]=True; q2.append((nx2,ny2))
            self._relay_ok=relay_ok
            self._relay_ok_flood={}
            for zx in range(self.zone_nx):
                for zy in range(self.zone_ny):
                    x0=zx*self.zone_w_cells; x1=min(x0+self.zone_w_cells,M.GRID_W)
                    y0=zy*self.zone_h_cells; y1=min(y0+self.zone_h_cells,M.GRID_H)
                    if np.any(relay_ok[x0:x1,y0:y1]):
                        self._relay_ok_flood[(zx,zy)]=True

            # -- Execution layer: check for stranded explorers ----------------
            # Runs EVERY tick when enabled. Detects relay loss and triggers
            # emergency re-election immediately without waiting for MILP cadence.
            if self.use_exec_layer:
                self._emergency_relay_election()

            # -- Track hold-gate waiting time (telemetry) ---------------------
            if self.use_exec_layer:
                for r in self.robots:
                    if r.active and r.exec_state=='waiting':
                        self.hold_ticks[r.name]=self.hold_ticks.get(r.name,0)+1

            # -- MILP allocation at cadence -----------------------------------
            if self.timestep==1 or self.timestep%ALLOCATION_CADENCE==0:
                assignment,relay,elapsed=self._solve_milp()
                self.milp_solve_times.append(elapsed)
                self.last_milp_time_ms=elapsed
                self._apply_allocation(assignment,relay)

            # -- Robot ticks --------------------------------------------------
            occupied={r.pos for r in self.robots if r.active}
            for r in self.robots:
                if not r.active: continue
                if r.battery<=0:
                    r.active=False; r.death_reason="battery depleted"
                    self.dead_robots.append((r.name,r.death_reason))
                    # Trigger emergency re-election next tick (not immediate -
                    # relay_ok rebuild happens at top of next step)
                    continue
                r.tick(occupied)

            # -- Replan on death (rate-limited) --------------------------------
            # Re-solve at most once per ALLOCATION_CADENCE//2 ticks after a death.
            # Without this, every robot death fires an immediate re-solve which
            # elects a different relay, causing relay churning every 1-2 ticks.
            new_deaths=any(not r.active for r in self.robots
                           if not getattr(r,'_death_handled',False))
            if new_deaths:
                for r in self.robots:
                    if not r.active:
                        object.__setattr__(r,'_death_handled',True) if hasattr(r,'__dict__') else None
                last_death_solve=getattr(self,'_last_death_solve_tick',-999)
                if self.timestep-last_death_solve >= ALLOCATION_CADENCE//2:
                    self._last_death_solve_tick=self.timestep
                    assignment,relay,elapsed=self._solve_milp()
                    self.milp_solve_times.append(elapsed)
                    self.last_milp_time_ms=elapsed
                    self._apply_allocation(assignment,relay)

            # -- Union belief + survivor detection ---------------------------
            self.union_belief=self._union_terrain()
            self.union_T=self._union_temp()
            self.union_R=self._union_rad()
            self.global_cov=float(np.mean(self.union_belief!=M.T_UNKNOWN))
            for r in self.robots:
                if not r.active: continue
                R2=r.terrain_R; rx,ry=r.pos
                r_inside=self.world.grid[rx][ry]["t"]==M.T_STAIRS
                for s in self.survivors:
                    if s in self.found: continue
                    sx,sy=s
                    if (rx-sx)**2+(ry-sy)**2>R2*R2: continue
                    if self._has_los(rx,ry,sx,sy,r_inside): self.found.add(s)
            if len(self.found)>=len(self.survivors): return False
            return any(r.active and r.battery>0 for r in self.robots)

    return CARASim

import random, time
from collections import deque
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csc_matrix

ALLOCATION_CADENCE = 50   # re-solve every N ticks (matches CBBA cadence)
ZONE_CAPACITY      = 2
MAX_BUNDLE         = 8


# -----------------------------------------------------------------------------
# CARABot - same movement logic as GNFRobot, but goal zone comes from MILP
# -----------------------------------------------------------------------------