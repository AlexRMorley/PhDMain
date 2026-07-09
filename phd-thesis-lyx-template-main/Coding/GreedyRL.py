"""
Single-Step Greedy RL Baseline with Shadow Awareness (γ=0)
===========================================================
Reward:
  W_COV * coverage_gain + W_CRIT * stair_first_entry
  + W_BORDER * relay_border_bonus - W_DIST * distance

Shadow model is now ACTIVE. Relay coverage built each step from border
positions. Rovers earn W_BORDER per unknown stair cell they would enable
by parking at the shadow border — this replaces explicit role election.
"""

import sys, os
from collections import deque
import numpy as np
from scipy import ndimage as _ndi

W_COV    = 1.0
W_CRIT   = 15.0
W_BORDER = 0.5
W_DIST   = 0.05
ACTION_RADIUS = 8


class GreedyRLRobot:
    __slots__ = (
        'name','pos','caps_mask','caps','world','sim',
        'temp_limit','rad_limit',
        'terrain_belief','known_mask','temp_belief','rad_belief',
        'chunked','active','battery','death_reason','hazard_killed',
        'goal','path','stuck_steps','failed_goals',
        'dose_T','dose_R','terrain_R',
        '_reachable_arr','_reachable_tick',
        'scan_age','confidence',
        'outbox','inbox','_inbox_dirty','_scan_dirty',
        'personally_scanned','_visited_stair',
        'role','task_zone',
    )

    def __init__(self, name, x, y, caps, caps_mask, world, sim, temp_limit, rad_limit):
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
        self._visited_stair=set()
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
            wbn=_ndi.binary_dilation((tb==M.T_WATER)|(tb==M.T_BRIDGE),structure=np.ones((3,3),dtype=bool))
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
            lab,_=_ndi.label(passable); arr=(lab==lab[sx,sy])
        self._reachable_arr=arr; self._reachable_tick=t; return arr

    def _reward(self, gx, gy):
        M=self.sim.M; union=self.sim.union_belief
        W,H=self.world.w,self.world.h; R=self.terrain_R
        x0=max(0,gx-R); x1=min(W,gx+R+1); y0=max(0,gy-R); y1=min(H,gy+R+1)
        xs=np.arange(x0,x1); ys=np.arange(y0,y1)
        xx,yy=np.meshgrid(xs,ys,indexing='ij')
        cov=int(np.count_nonzero(((xx-gx)**2+(yy-gy)**2<=R*R)&(union[x0:x1,y0:y1]==M.T_UNKNOWN)))
        is_stair=self.world.grid[gx][gy]["t"]==M.T_STAIRS
        crit=W_CRIT if is_stair and (gx,gy) not in self._visited_stair else 0.0
        border_bonus=0.0
        can_enter=bool(self.caps_mask&(M.CAP_STAIRS|M.CAP_AIR))
        brd=self.sim._shadow_border_mask_cache; shd=self.sim.radio_shadow
        if not can_enter and brd[gx,gy]:
            if not hasattr(self.sim,'_world_stair_arr'):
                self.sim._world_stair_arr=np.array(
                    [[self.world.grid[x][y]["t"]==M.T_STAIRS
                      for y in range(H)] for x in range(W)],dtype=bool)
            unk_stair=int(np.count_nonzero(shd&(union==M.T_UNKNOWN)&self.sim._world_stair_arr))
            border_bonus=W_BORDER*unk_stair
            if any(r.active and r is not self and bool(r.caps_mask&(M.CAP_STAIRS|M.CAP_AIR))
                   and shd[r.pos[0],r.pos[1]] for r in self.sim.robots):
                border_bonus*=2.0
        return W_COV*cov+crit+border_bonus-W_DIST*(abs(gx-self.pos[0])+abs(gy-self.pos[1]))

    def _greedy_goal(self):
        M=self.sim.M; W,H=self.world.w,self.world.h
        reach=self.reachable(); union=self.sim.union_belief
        px,py=self.pos; unk_mask=(union==M.T_UNKNOWN)
        adj_unk=_ndi.binary_dilation(unk_mask,structure=np.ones((3,3),dtype=bool))
        brd=self.sim._shadow_border_mask_cache
        can_enter=bool(self.caps_mask&(M.CAP_STAIRS|M.CAP_AIR))
        best=None; best_s=-1e9
        vis=np.zeros((W,H),dtype=bool); q=deque([(px,py,0)]); vis[px,py]=True
        while q:
            x,y,d=q.popleft()
            if d>ACTION_RADIUS: continue
            if (x,y)!=(px,py) and self.sim.timestep>=self.failed_goals.get((x,y),0):
                is_boat=bool(self.caps_mask&M.CAP_WATER) and not bool(self.caps_mask&M.CAP_AIR)
                is_stair_new=self.world.grid[x][y]["t"]==M.T_STAIRS and (x,y) not in self._visited_stair
                is_border=not can_enter and brd[x,y]
                if is_boat:
                    ok=union[x,y] in (M.T_WATER,M.T_BRIDGE) and unk_mask[x,y]
                else:
                    ok=unk_mask[x,y] or adj_unk[x,y] or is_stair_new or is_border
                if ok:
                    s=self._reward(x,y)
                    if s>best_s: best_s=s; best=(x,y)
            if d<ACTION_RADIUS:
                for nx,ny in self.world.neighbours((x,y)):
                    if not vis[nx,ny] and reach[nx,ny]:
                        vis[nx,ny]=True; q.append((nx,ny,d+1))
        return best

    def _plan_to(self, goal):
        M=self.sim.M
        _zc=np.zeros((2,M.GRID_W//M.CHUNK_SIZE,M.GRID_H//M.CHUNK_SIZE),dtype=np.float32)
        _zt=np.zeros((M.GRID_W,M.GRID_H),dtype=np.uint16)
        can_enter=bool(self.caps_mask&(M.CAP_STAIRS|M.CAP_AIR))
        rok=self.sim._relay_ok; gx,gy=goal
        path=M.AStar.search(
            start=self.pos,goal=goal,caps_mask=self.caps_mask,
            terrain_u8=self.terrain_belief,temp_f32=self.temp_belief,rad_f32=self.rad_belief,
            chunked_risk=_zc,temp_limit=9999.0,rad_limit=9999.0,
            radio_shadow=self.sim.radio_shadow,
            relay_ok_fn=(lambda z:bool(rok[gx,gy])) if can_enter else (lambda z:False),
            cell_to_zone_fn=self.sim.cell_to_zone,
            global_cov=self.sim.global_cov,unk_pen=0.3,info_w=0.1,unk_prior=0.25,
            alpha_mult=0.0,beta_mult=0.0,soft_frac=1.0,
            traffic_u16=_zt,traffic_w=0.0,
            shadow_border=self.sim._shadow_border_mask_cache)
        if not path: self.failed_goals[goal]=self.sim.timestep+60; return False
        self.goal=goal; self.path=path; return True

    def tick(self, occupied):
        M=self.sim.M
        if not self.active or self.battery<=0: self.active=False; return
        tgt=self._greedy_goal()
        if tgt is None:
            self.stuck_steps+=1
            if self.stuck_steps>20: self.failed_goals={}; self.stuck_steps=0
            return
        self.stuck_steps=0
        if tgt!=self.goal or not self.path:
            if not self._plan_to(tgt): return
        if not self.path: return
        nc=self.path[0]; tt=self.world.grid[nc[0]][nc[1]]["t"]
        if tt==M.T_OBS: self.path=[]; self.goal=None; return
        is_boat=bool(self.caps_mask&M.CAP_WATER) and not bool(self.caps_mask&M.CAP_AIR)
        if is_boat and tt not in (M.T_WATER,M.T_BRIDGE):
            self.terrain_belief[nc[0],nc[1]]=tt; self.known_mask[nc[0],nc[1]]=True
            self.path=[]; self.goal=None; return
        occupied.discard(self.pos); self.pos=self.path.pop(0); occupied.add(self.pos)
        if self.world.grid[self.pos[0]][self.pos[1]]["t"]==M.T_STAIRS:
            self._visited_stair.add(self.pos)
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


def make_greedy_rl_sim(gnf_module, M_module):
    """
    Returns GreedyRLSim subclassing GNFSim.
    Inherits shadow infrastructure and the step() relay-rebuild loop.
    Only the per-robot policy differs.
    """
    GNFSim=gnf_module.GNFSim

    class GreedyRLSim(GNFSim):
        def _build_robots(self):
            super()._build_robots()
            self.robots=[
                GreedyRLRobot(r.name,r.pos[0],r.pos[1],r.caps,r.caps_mask,
                               self.world,self,r.temp_limit,r.rad_limit)
                for r in self.robots]

    return GreedyRLSim