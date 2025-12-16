from __future__ import annotations
# from environment_variables import env_grid_size_,tile_size_,see_through_walls_

# ===================================================
# >>> CENTRAL ENV CONFIG  <<<
# ===================================================

ENV_GRID_SIZE     = 10
TILE_SIZE         = 4
SEE_THROUGH_WALLS = True

# ===================================================

KEYCORRIDOR_ROOM_SIZE = 3         # For KeyCorridorEnv
KEYCORRIDOR_NUM_ROOMS = 1

AGENT_START_POS        = (1, 1)
AGENT_START_DIR        = 0

DEFAULT_MAX_STEPS     = None

# ===================================================


from typing import Literal, Tuple, Optional

import numpy as np

from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall
from minigrid.minigrid_env import MiniGridEnv
from minigrid.wrappers import (
    RGBImgObsWrapper,
    RGBImgPartialObsWrapper,
    FullyObsWrapper,
)

# Built-in MiniGrid envs
from minigrid.envs import DoorKeyEnv, KeyCorridorEnv

# ===================================
# Custom MiniGrid Environment
# ===================================

class SimpleEnv(MiniGridEnv):
    """
    A minimal custom environment:
      - Vertical wall splitting the room
      - A locked door & matching key
      - A goal square
    """

    def __init__(
        self,
        size: int = ENV_GRID_SIZE,
        agent_start_pos: Tuple[int, int] = AGENT_START_POS,
        agent_start_dir: int = AGENT_START_DIR,
        max_steps: Optional[int] = DEFAULT_MAX_STEPS,
        **kwargs,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 4 * size**2

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            see_through_walls=SEE_THROUGH_WALLS,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "reach the goal"
    
    # def _gen_grid(self, width: int, height: int):
    #     self.grid = Grid(width, height)

    #     # Surrounding walls
    #     self.grid.wall_rect(0, 0, width, height)

    #     # Create reachable maze paths from spawn area → goal
    #     # Left corridor (safe spawn zone)
    #     for y in range(1, height-2):  
    #         self.grid.set(1, y, None)  # Open path
        
    #     # Bottom corridor to goal  
    #     for x in range(1, width-2):  
    #         self.grid.set(x, height-2, None)
        
    #     # Strategic walls (don't block spawn!)
    #     for y in range(3, 7):
    #         self.grid.set(3, y, Wall())
    #     for y in range(2, 6):
    #         self.grid.set(6, y, Wall())
        
    #     # Scattered obstacles (avoid spawn area)
    #     obstacles = [(4,4), (5,5), (7,3)]
    #     for x, y in obstacles:
    #         if 2 < x < width-2 and 2 < y < height-2:  # Safe distance from spawn
    #             self.grid.set(x, y, Wall())

    #     # Goal (clear path)
    #     self.grid.set(width-2, height-2, None)
    #     self.grid.set(width-2, height-2, Goal())
    #     # self.goal_pos = (width-2, height-2)

    #     # DYNAMIC AGENT PLACEMENT - Let MiniGrid find safe spot
    #     self.place_agent()  # Auto-finds valid position (0,0) check passes
        
    #     self.mission = "reach the goal"

    def _gen_grid(self, width: int, height: int):
        """
        Generates a random valid grid. 
        Retries until a solvable path from Agent -> Goal exists.
        """
        max_retries = 100
        
        for _ in range(max_retries):
            # 1. Create empty grid with surrounding walls
            self.grid = Grid(width, height)
            self.grid.wall_rect(0, 0, width, height)

            # 2. Randomize Goal Placement
            # place_obj finds an empty spot automatically
            self.goal_pos = self.place_obj(Goal(), max_tries=100)

            # 3. Randomize Obstacles
            # Density: 15-20% of the grid is obstacles. 
            # 10x10 grid = 100 tiles. ~15 walls.
            num_obstacles = int((width * height) * 0.15) 
            
            for _ in range(num_obstacles):
                self.place_obj(Wall(), max_tries=100)

            # 4. Randomize Agent Placement
            self.place_agent()

            # 5. CRITICAL: Check solvability
            if self._is_reachable():
                self.mission = "reach the goal"
                return  # Success! Keep this grid.

        # If we failed 100 times (super rare), fallback to a simple open room
        # print("Warning: Map generation failed 100 times. Fallback to empty room.")
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)
        self.place_obj(Goal())
        self.place_agent()
        self.mission = "reach the goal"

    def _is_reachable(self):
        """
        Simple Breadth-First Search (BFS) to verify the goal is reachable.
        Returns: True if a path exists, False otherwise.
        """
        start_pos = self.agent_pos
        queue = [start_pos]
        visited = set()
        visited.add(start_pos)

        # Directions: Right, Down, Left, Up
        dirs = [(1, 0), (0, 1), (-1, 0), (0, -1)]

        while queue:
            current = queue.pop(0)

            # If we found the goal, the map is valid
            if current == self.goal_pos:
                return True

            cx, cy = current
            
            # Explore neighbors
            for dx, dy in dirs:
                nx, ny = cx + dx, cy + dy

                # Check bounds (inside grid)
                if 0 <= nx < self.grid.width and 0 <= ny < self.grid.height:
                    # Check if not visited
                    if (nx, ny) not in visited:
                        # Check if not a wall
                        cell = self.grid.get(nx, ny)
                        # MiniGrid logic: None is empty, Goal is walkable. Wall is not.
                        if cell is None or isinstance(cell, Goal):
                            visited.add((nx, ny))
                            queue.append((nx, ny))
        
        # Queue empty, goal never found
        return False




# ===================================================
# RL-Ready Wrapper (Unified interface for all envs)
# ===================================================

ObsMode = Literal["symbolic", "rgb"]
ObsScope = Literal["partial", "full"]
EnvKind = Literal["simple", "doorkey", "keycorridor"]


class RLReadyEnv:
    """
    Helper around MiniGrid envs (custom or built-in).

    Allows easy configuration of:

        env_kind : "simple" | "doorkey" | "keycorridor"
        obs_mode : "symbolic" | "rgb"
        obs_scope: "partial" | "full"
    """

    def __init__(
        self,
        env_kind: EnvKind = "simple",

        # Level params
        size: int = ENV_GRID_SIZE,

        # KeyCorridor difficulty params
        keycorridor_s: int = KEYCORRIDOR_ROOM_SIZE,
        keycorridor_r: int = KEYCORRIDOR_NUM_ROOMS,

        tile_size: int = TILE_SIZE,

        # Observation config
        obs_mode: ObsMode = "symbolic",
        obs_scope: ObsScope = "partial",

        # General options
        render_mode: Optional[str] = None,
        agent_start_pos: Tuple[int, int] = AGENT_START_POS,
        agent_start_dir: int = AGENT_START_DIR,
        max_steps: Optional[int] = DEFAULT_MAX_STEPS,
        seed: Optional[int] = None,
    ):

        # Construct base env
        env = self._build_base_env(
            env_kind=env_kind,
            size=size,
            keycorridor_s=keycorridor_s,
            keycorridor_r=keycorridor_r,
            render_mode=render_mode,
            agent_start_pos=agent_start_pos,
            agent_start_dir=agent_start_dir,
            max_steps=max_steps,
        )

        # -------------------------
        # Apply wrappers
        # -------------------------

        # Full observability
        if obs_scope == "full":
            env = FullyObsWrapper(env)

        # RGB output
        if obs_mode == "rgb":
            if obs_scope == "partial":
                env = RGBImgPartialObsWrapper(env, tile_size=tile_size)
            else:
                env = RGBImgObsWrapper(env, tile_size=tile_size)

        self.env = env
        self.env_kind = env_kind
        self.obs_mode = obs_mode
        self.obs_scope = obs_scope

        if seed is not None:
            self.seed(seed)

    # -------------------------------------------------

    def _build_base_env(
        self,
        env_kind: EnvKind,
        size: int,
        keycorridor_s: int,
        keycorridor_r: int,
        render_mode: Optional[str],
        agent_start_pos: Tuple[int, int],
        agent_start_dir: int,
        max_steps: Optional[int],
    ) -> MiniGridEnv:

        # ======================
        # Custom environment
        # ======================
        if env_kind == "simple":
            return SimpleEnv(
                size=size,
                agent_start_pos=agent_start_pos,
                agent_start_dir=agent_start_dir,
                max_steps=max_steps,
                render_mode=render_mode,
            )

        # ======================
        # DoorKey puzzle
        # ======================
        elif env_kind == "doorkey":
            return DoorKeyEnv(
                size=size,
                max_steps=max_steps,
                render_mode=render_mode,
            )

        # ======================
        # KeyCorridor puzzle
        # ======================
        elif env_kind == "keycorridor":
            return KeyCorridorEnv(
                room_size=keycorridor_s,
                num_rows=keycorridor_r,
                max_steps=max_steps,
                render_mode=render_mode,
            )

        else:
            raise ValueError(f"Unknown env_kind: {env_kind}")

    # -------------------------------------------------
    # Pass-through gym helpers
    # -------------------------------------------------

    def reset(self, **kwargs):
        # >>> FORCE RANDOM SEEDING <<<
        # If no seed is provided, generate a random one.
        # This ensures the agent spawns in a new location every episode
        # and the world model learns global features, not just one path.
        if "seed" not in kwargs:
            kwargs["seed"] = np.random.randint(0, 2**31 - 1)
            
        obs = self.env.reset(**kwargs)
        self.goal_pos = self._find_goal_pos()   
        return obs


    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        
        # Find positions
        agent_pos = self.env.unwrapped.agent_pos
        goal_pos = self.goal_pos
        
        if goal_pos:
            dist = abs(agent_pos[0] - goal_pos[0]) + abs(agent_pos[1] - goal_pos[1])
            shaped_reward = -dist / (ENV_GRID_SIZE * 2)
        else:
            shaped_reward = 0
        
        reached_goal = False
        # OVERRIDE: +1 success when AT goal (ignore 'done' action requirement)
        if goal_pos and agent_pos == goal_pos:
            reward = 10.0   # Big spike for success
            term = True
            reached_goal = True    
        # elif agent_pos != goal_pos:
        #     reward = -0.1   # Small nag to hurry up
        # total_reward = reward + shaped_reward
        total_reward = reward + shaped_reward
        # print(f"dist:{dist}, sparse:{reward}, shaped:{shaped_reward}, total:{total_reward}")
        
        return obs, total_reward, term, trunc, info, reached_goal


    def _find_goal_pos(self):
        """Scan grid for Goal object - works everywhere"""
        grid = self.env.unwrapped.grid
        for i in range(grid.width):
            for j in range(grid.height):
                if isinstance(grid.get(i, j), Goal):
                    return (i, j)
        return None

    def render(self):
        return self.env.render()

    def seed(self, seed: int):
        self.env.reset(seed=seed)

    def close(self):
        self.env.close()