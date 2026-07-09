"""
2-D heterogeneous robot fleet exploration simulator with GUI controls.

Robots: Legged, Drone, Boat, Rover
Environment: grid cells with semantic terrain (STAIRS band & WATER pool & RIVER; rest FREE/OBSTACLE).
Planner: A* on each robot’s current known map.
GUI: pygame grid + faint cell lines + sidebar with clickable Start, Show Map, Show Survivors, Show Heat, Show Rad buttons + robot key + coverage % + time step + checklist of survivors.
Robots reveal a 3×3 area around themselves when they move (and at start).

Run:
    pip install pygame
    python hetero_robot_fleet_sim.py
"""

import pygame
from enum import Enum
import heapq
import math
import random
import sys
from collections import deque
import numpy as np

# ---------- Configuration ----------
CELL_SIZE    = 10
GRID_W,GRID_H= 32,32
FPS          = 10
SIDEBAR_WIDTH= 200
MAX_BATTERY  = 1000
TEMP_LIMIT   = 85      # raised so robots only break down nearer centre
RAD_LIMIT    = 100     # raised so radiation breakdown nearer centre
# -----------------------------------


class Terrain(Enum):
    UNKNOWN=0; FREE=1; OBSTACLE=2; STAIRS=3; WATER=4

class Capability(Enum):
    LAND=1; STAIRS=2; WATER=3; AIR=4

TERRAIN_COLOUR = {
    Terrain.UNKNOWN:  (200,200,200),
    Terrain.FREE:     (255,255,255),
    Terrain.OBSTACLE: (  0,  0,  0),
    Terrain.STAIRS:   (255,255,  0),
    Terrain.WATER:    (  0,  0,255),
}
ROBOT_COLOUR = {
    "Legged": (  0,255,  0),
    "Drone":  (255,  0,255),
    "Boat":   (  0,255,255),
    "Rover":  (255,165,  0),
}
SURVIVOR_COLOUR = (255,  0,  0)

# ---------- Grid & Cells ----------
class Cell:
    def __init__(self, true_terrain=None):
        self.true_terrain = true_terrain or Terrain.FREE
        self.terrain      = Terrain.UNKNOWN
        # temperature & radiation
        self.temperature  = 0.0
        self.radiation    = 0.0

    def cost(self, caps:set)->float:
        t=self.true_terrain
        if t is Terrain.OBSTACLE: return math.inf
        if t is Terrain.STAIRS and Capability.STAIRS not in caps and Capability.AIR not in caps:
            return math.inf
        if t is Terrain.WATER and Capability.WATER not in caps and Capability.AIR not in caps:
            return math.inf
        if t is Terrain.FREE and Capability.LAND not in caps and Capability.AIR not in caps:
            return math.inf
        return 1.0

class GridWorld:
    def __init__(self,w,h):
        self.w,self.h=w,h
        self.grid=[[Cell() for _ in range(h)] for _ in range(w)]
        self._generate_demo_world()
        self._initialize_temperature()
        self._initialize_radiation()

    def _generate_demo_world(self):
        # exactly as before—random obstacles, pool, river, bridge, building, rest FREE
        for x in range(self.w):
            for y in range(self.h):
                if random.random()<0.05:
                    self.grid[x][y]=Cell(Terrain.OBSTACLE)


    def _initialize_temperature(self):
        # four Gaussian heat sources
        sources = [
            ((20, 31), 10, 100),
            ((45, 15), 8, 90),
            ((15, 55), 5, 120),
            ((50, 55), 7, 55),
        ]
        for x in range(self.w):
            for y in range(self.h):
                temp = 0.0
                for (mx, my), sigma, amplitude in sources:
                    temp += amplitude * math.exp(-((x-mx)**2+(y-my)**2)/(2*sigma**2))
                if self.grid[x][y].true_terrain == Terrain.WATER:
                    temp = 5.0
                self.grid[x][y].temperature = temp

    def _initialize_radiation(self):
        # radiation sources, one overlapping a heat spot
        rad_sources = [
            ((20, 31), 5, 100),
            ((40, 50), 8, 80),
            ((10, 10), 6, 90),
            ((50, 10), 5, 120),  # smaller sigma, more potent
        ]
        for x in range(self.w):
            for y in range(self.h):
                rad = 0.0
                for (mx, my), sigma, amplitude in rad_sources:
                    rad += amplitude * math.exp(-((x-mx)**2+(y-my)**2)/(2*sigma**2))
                if self.grid[x][y].true_terrain == Terrain.WATER:
                    rad = 0.0
                self.grid[x][y].radiation = rad

    def neighbours(self,u):
        x,y=u
        for dx,dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx,ny=x+dx,y+dy
            if 0<=nx<self.w and 0<=ny<self.h:
                yield (nx,ny)

    def cost(self,u,v,caps):
        return self.grid[v[0]][v[1]].cost(caps)

# ---------- A* Planner ----------
def heuristic(a,b):
    return abs(a[0]-b[0])+abs(a[1]-b[1])

class AStar:
    def __init__(self, world, start, goal, caps):
        self.world,self.start,self.goal,self.caps=world,start,goal,caps

    def search(self):
        open_set = [(0, self.start)]
        came_from = {}
        g_score = {self.start: 0}
        f_score = {self.start: heuristic(self.start, self.goal)}

        while open_set:
            _, current = heapq.heappop(open_set)
            if current == self.goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                return path[::-1]

            for nbr in self.world.neighbours(current):
                cell = self.world.grid[nbr[0]][nbr[1]]
                cost = cell.cost(self.caps)
                if math.isinf(cost): continue

                tentative_g = g_score[current] + cost
                if tentative_g < g_score.get(nbr, math.inf):
                    came_from[nbr] = current
                    g_score[nbr]   = tentative_g
                    f_score[nbr]   = tentative_g + heuristic(nbr, self.goal)
                    heapq.heappush(open_set, (f_score[nbr], nbr))
        return []

# ---------- Robot (single class) ----------
class Robot:
    def __init__(self,name,x,y,caps,world):
        self.name,self.pos,self.caps,self.world=name,(x,y),caps,world
        self.goal,self.path=None,[]
        self.active=True
        self.battery=MAX_BATTERY
        self.death_reason=None
        self.reveal()
        
        # ← new: choose "astar" or "greedy"

        self.strategy = "astar"

    def reveal(self):
        x0,y0=self.pos
        for dx in (-1,0,1):
            for dy in (-1,0,1):
                nx,ny=x0+dx,y0+dy
                if 0<=nx<self.world.w and 0<=ny<self.world.h:
                    self.world.grid[nx][ny].terrain=self.world.grid[nx][ny].true_terrain

    def set_goal(self,tgt):
        self.goal=tgt
        self.path=AStar(self.world,self.pos,tgt,self.caps).search()

    def step(self):
        if not self.path or self.battery <= 0:
            self.active = False
            if self.battery <= 0:
                self.death_reason = "battery depleted"
            return
        self.pos = self.path.pop(0)
        self.reveal()
        # battery consumption
        if self.name == "Drone":
            self.battery -= 3
        elif self.name == "Legged":
            self.battery -= 2
        elif self.name == "Boat":
            self.battery -= 2
        else:
            self.battery -= 0.5
        # environmental kill for Legged & Drone
        if self.name in ("Legged","Drone"): 
            cell = self.world.grid[self.pos[0]][self.pos[1]]
            over_temp = cell.temperature > TEMP_LIMIT
            over_rad  = cell.radiation > RAD_LIMIT
            if over_temp or over_rad:
                self.active = False
                reasons = []
                if over_temp: reasons.append("high temperature")
                if over_rad:  reasons.append("high radiation")
                self.death_reason = " & ".join(reasons)

# ---------- Simulation Controller ----------
class FleetSim:
    def __init__(self):
        self.world=GridWorld(GRID_W,GRID_H)
        starts=[(1,1),(GRID_W-2,1),(1,GRID_H-2),(GRID_W-2,GRID_H-2)]
        names=["Rover","Rover","Rover","Rover"]
        caps_l=[{Capability.LAND,Capability.STAIRS},{Capability.AIR},
                {Capability.WATER},{Capability.LAND}]
        self.robots=[Robot(names[i],*starts[i],caps_l[1],self.world)
                     for i in range(4)]

        free_cells=[(x,y) for x in range(self.world.w) for y in range(self.world.h)
                    if self.world.grid[x][y].true_terrain==Terrain.FREE]
        self.survivors = random.sample(free_cells, 3)
        self.found = set()
        self.dead_robots = []

    def step(self):
        any_active=False
        for r in self.robots:
            if not r.active:
                continue

            # detect survivors
            for s in self.survivors:
                if s not in self.found and abs(r.pos[0]-s[0])<=1 and abs(r.pos[1]-s[1])<=1:
                    self.found.add(s)

            # choose frontier
            dq=deque([r.pos]); reachable={r.pos}
            while dq:
                u=dq.popleft()
                for v in self.world.neighbours(u):
                    if v in reachable: continue
                    if self.world.grid[v[0]][v[1]].cost(r.caps)<math.inf:
                        reachable.add(v); dq.append(v)
            frontiers=[(x,y)
                for x in range(self.world.w)
                for y in range(self.world.h)
                if self.world.grid[x][y].terrain==Terrain.UNKNOWN and (x,y) in reachable]
            if not frontiers:
                r.active=False
                if r.death_reason:
                    self.dead_robots.append((r.name, r.death_reason))
                continue

            any_active=True
            if r.goal is None or r.pos==r.goal or not r.path:
                tgt=min(frontiers,key=lambda c:heuristic(r.pos,c))
                r.set_goal(tgt)
            r.step()
            if not r.active and r.death_reason:
                self.dead_robots.append((r.name, r.death_reason))

        # end condition
        if len(self.found) == len(self.survivors):
            return False
        return any_active

# ---------- GUI routines ----------
def draw_grid(scr,world,show_map,show_survivors,show_heat,show_rad):
    for x in range(world.w):
        for y in range(world.h):
            pos=(x,y)
            if pos in sim.found or (show_survivors and pos in sim.survivors):
                clr=SURVIVOR_COLOUR
            elif show_rad:
                # radiation map: water black, high rad bright green
                r = world.grid[x][y].radiation
                if world.grid[x][y].true_terrain == Terrain.WATER:
                    clr = (0,0,0)
                else:
                    v = min(r/100,1)
                    clr = (0, int(255*v), 0)
            elif show_heat:
                t = world.grid[x][y].temperature
                t_norm = min(max((t-0)/(100-0),0),1)
                clr = (int(255*t_norm), 0, int(255*(1-t_norm)))
            else:
                t=(world.grid[x][y].true_terrain if show_map
                   else world.grid[x][y].terrain)
                clr=TERRAIN_COLOUR[t]
            rect=pygame.Rect(x*CELL_SIZE,y*CELL_SIZE,
                             CELL_SIZE,CELL_SIZE)
            pygame.draw.rect(scr,clr,rect)
            pygame.draw.rect(scr,(150,150,150),rect,1)

def draw_robots(scr,robots):
    for r in robots:
        x,y=r.pos; clr=ROBOT_COLOUR[r.name]
        pygame.draw.rect(scr,clr,
                         (x*CELL_SIZE,y*CELL_SIZE,
                          CELL_SIZE,CELL_SIZE))

def gui_loop():
    global sim
    pygame.init()
    screen=pygame.display.set_mode(
        (GRID_W*CELL_SIZE+SIDEBAR_WIDTH,GRID_H*CELL_SIZE))
    pygame.display.set_caption("Heterogeneous Robot Fleet Simulator")
    font=pygame.font.SysFont(None,24)
    sim=FleetSim()
    running=False; show_map=False; show_survivors=False
    show_heat=False; show_rad=False; time_step=0

    start_btn=pygame.Rect(GRID_W*CELL_SIZE+10,10,80,30)
    map_btn  =pygame.Rect(GRID_W*CELL_SIZE+10,50,100,30)
    surv_btn =pygame.Rect(GRID_W*CELL_SIZE+10,90,100,30)
    heat_btn =pygame.Rect(GRID_W*CELL_SIZE+10,130,100,30)
    rad_btn  =pygame.Rect(GRID_W*CELL_SIZE+10,170,100,30)
    clock=pygame.time.Clock()

    while True:
        for ev in pygame.event.get():
            if ev.type==pygame.QUIT:
                pygame.quit(); sys.exit()
            if ev.type==pygame.MOUSEBUTTONDOWN and ev.button==1:
                if start_btn.collidepoint(ev.pos):
                    running=not running
                if map_btn.collidepoint(ev.pos):
                    show_map=not show_map
                if surv_btn.collidepoint(ev.pos):
                    show_survivors=not show_survivors
                if heat_btn.collidepoint(ev.pos):
                    show_heat=not show_heat
                    if show_heat: show_rad=False
                if rad_btn.collidepoint(ev.pos):
                    show_rad=not show_rad
                    if show_rad: show_heat=False

        if running:
            if not sim.step():
                running=False
                print("Exploration Complete!")
                if sim.dead_robots:
                    print("Robots incapacitated during run:")
                    for name, reason in sim.dead_robots:
                        print(f" - {name} stopped due to {reason}")
            time_step+=1

        draw_grid(screen,sim.world,show_map,show_survivors,show_heat,show_rad)
        draw_robots(screen,sim.robots)

        # draw legends
        legend_x = GRID_W*CELL_SIZE + 10
        legend_y = rad_btn.y + 220
        if show_heat:
            pygame.draw.rect(screen,(0,0,0),(legend_x,legend_y,20,100),1)
            for i in range(100):
                tnorm = 1 - i/99
                clr = (int(255*tnorm),0,int(255*(1-tnorm)))
                pygame.draw.line(screen,clr,(legend_x,legend_y+i),(legend_x+20,legend_y+i))
            screen.blit(font.render("Temp",True,(0,0,0)),(legend_x,legend_y+105))
            screen.blit(font.render("0",True,(0,0,0)),(legend_x+25,legend_y+90))
            screen.blit(font.render("100",True,(0,0,0)),(legend_x+25,legend_y))
        if show_rad:
            rx = legend_x + 40
            pygame.draw.rect(screen,(0,0,0),(rx,legend_y,20,100),1)
            for i in range(100):
                rnorm = i/99
                clr = (0,int(255*rnorm),0)
                pygame.draw.line(screen,clr,(rx,legend_y+i),(rx+20,legend_y+i))
            screen.blit(font.render("Rad",True,(0,0,0)),(rx,legend_y+105))
            screen.blit(font.render("0",True,(0,0,0)),(rx+25,legend_y+90))
            screen.blit(font.render("100",True,(0,0,0)),(rx+25,legend_y))

        pygame.draw.rect(screen,(255,255,255),
                         (GRID_W*CELL_SIZE,0,
                          SIDEBAR_WIDTH,GRID_H*CELL_SIZE))
        pygame.draw.rect(screen,(200,200,200),start_btn)
        screen.blit(font.render('Pause' if running else 'Start',
                                True,(0,0,0)),
                    (start_btn.x+10,start_btn.y+5))
        pygame.draw.rect(screen,(200,200,200),map_btn)
        screen.blit(font.render('Hide Map' if show_map else 'Show Map',
                                True,(0,0,0)),
                    (map_btn.x+10,map_btn.y+5))
        pygame.draw.rect(screen,(200,200,200),surv_btn)
        screen.blit(font.render('Hide Survi' if show_survivors else 'Show Survi',
                                True,(0,0,0)),
                    (surv_btn.x+10,surv_btn.y+5))
        pygame.draw.rect(screen,(200,200,200),heat_btn)
        screen.blit(font.render('Hide Heat' if show_heat else 'Show Heat',
                                True,(0,0,0)),
                    (heat_btn.x+10,heat_btn.y+5))
        pygame.draw.rect(screen,(200,200,200),rad_btn)
        screen.blit(font.render('Hide Rad' if show_rad else 'Show Rad',
                                True,(0,0,0)),
                    (rad_btn.x+10,rad_btn.y+5))

        screen.blit(font.render(f"Step: {time_step}",True,(0,0,0)),
                    (GRID_W*CELL_SIZE+10,rad_btn.y+50))
        total=sim.world.w*sim.world.h
        disc=sum(1 for x in range(sim.world.w)
                 for y in range(sim.world.h)
                 if sim.world.grid[x][y].terrain!=Terrain.UNKNOWN)
        pct=disc/total*100
        screen.blit(font.render(f"Coverage: {pct:.1f}%",True,(0,0,0)),
                    (GRID_W*CELL_SIZE+10,rad_btn.y+80))
        y=rad_btn.y+110
        screen.blit(font.render("Battery:",True,(0,0,0)),
                    (GRID_W*CELL_SIZE+10,y)); y+=25
        for r in sim.robots:
            screen.blit(font.render(f"{r.name}: {r.battery:.1f}",True,(0,0,0)),
                        (GRID_W*CELL_SIZE+10,y))
            y+=25
        screen.blit(font.render("Robot Key:",True,(0,0,0)),
                    (GRID_W*CELL_SIZE+10,y)); y+=25
        for nm,clr in ROBOT_COLOUR.items():
            pygame.draw.rect(screen,clr,
                             (GRID_W*CELL_SIZE+10,y,20,20))
            screen.blit(font.render(nm,True,(0,0,0)),
                        (GRID_W*CELL_SIZE+40,y))
            y+=25
        screen.blit(font.render("Survivors:",True,(0,0,0)),
                    (GRID_W*CELL_SIZE+10,y)); y+=25
        for idx,pos in enumerate(sim.survivors, start=1):
            found = pos in sim.found
            mark = "✔" if found else "✖"
            col  = (0,200,0) if found else (200,0,0)
            screen.blit(font.render(f"{mark} S{idx} {pos}",True,col),
                        (GRID_W*CELL_SIZE+10,y))
            y+=25

        pygame.display.flip()
        clock.tick(FPS)

if __name__=="__main__":
    gui_loop()
