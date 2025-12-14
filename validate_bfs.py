import time
import numpy as np
from gridworldenv import RLReadyEnv
from minigrid.core.world_object import Goal

# ==========================================
# 1. THE EXACT BFS SOLVER LOGIC
# ==========================================
def bfs_solve(env):
    """
    Returns a list of actions [int] to walk from agent_pos to goal_pos
    navigating around walls.
    Actions: 0=Left, 1=Right, 2=Forward
    """
    # Access the internal MiniGrid object
    grid = env.env.unwrapped.grid
    start_pos = env.env.unwrapped.agent_pos
    start_dir = env.env.unwrapped.agent_dir # 0=East, 1=South, 2=West, 3=North
    goal_pos = env.goal_pos
    
    # Queue state: (x, y, direction, path_of_actions_taken)
    queue = [(start_pos[0], start_pos[1], start_dir, [])]
    
    # Visited state: (x, y, direction)
    # We must include direction in visited because arriving at (5,5) facing North 
    # is different than arriving at (5,5) facing East (different moves available).
    visited = set()
    visited.add((start_pos[0], start_pos[1], start_dir))
    
    print(f"BFS Start: {start_pos} -> Goal: {goal_pos}")
    
    while queue:
        x, y, d, path = queue.pop(0)
        
        # Check if we are at goal (position match is enough)
        if (x, y) == goal_pos:
            return path 
        
        # Limit search depth to prevent freezing on impossible maps
        if len(path) > 50: 
            continue 
        
        # --- Try 3 possible moves: Left, Right, Forward ---
        
        # 1. Turn Left (Action 0)
        # New Dir = (Current - 1) % 4
        new_d = (d - 1) % 4
        if (x, y, new_d) not in visited:
            visited.add((x, y, new_d))
            queue.append((x, y, new_d, path + [0]))
            
        # 2. Turn Right (Action 1)
        # New Dir = (Current + 1) % 4
        new_d = (d + 1) % 4
        if (x, y, new_d) not in visited:
            visited.add((x, y, new_d))
            queue.append((x, y, new_d, path + [1]))
            
        # 3. Move Forward (Action 2)
        # Calculate new coordinates based on current direction
        fx, fy = x, y
        if d == 0: fx += 1   # East
        elif d == 1: fy += 1 # South
        elif d == 2: fx -= 1 # West
        elif d == 3: fy -= 1 # North
        
        # Check grid bounds
        if 0 <= fx < grid.width and 0 <= fy < grid.height:
            cell = grid.get(fx, fy)
            # Check if walkable (None = Empty, or explicit Goal object)
            if cell is None or isinstance(cell, Goal) or (goal_pos and (fx, fy) == goal_pos):
                if (fx, fy, d) not in visited:
                    visited.add((fx, fy, d))
                    queue.append((fx, fy, d, path + [2]))
                    
    return [] # No path found

# ==========================================
# 2. VISUALIZATION LOOP
# ==========================================
if __name__ == "__main__":
    
    # Initialize the environment EXACTLY as we do in training, but with Human Rendering
    env = RLReadyEnv(
        env_kind="simple",
        size=10,
        obs_mode="rgb",
        obs_scope="partial",
        render_mode="human", # <--- VISUALIZATION ON
        agent_start_pos=(1, 1),
        agent_start_dir=0,
        max_steps=None
    )

    print("Environment Initialized. Starting BFS Validation Loop...")
    
    try:
        for episode in range(10): # Run 10 test episodes
            print(f"\n--- Episode {episode + 1} ---")
            
            # Reset Environment (Generates new random map)
            obs = env.reset()
            env.render()
            
            # Solve using BFS
            print("Solving...")
            actions = bfs_solve(env)
            
            if not actions:
                print("No path found! (Map might be impossible or goal blocked)")
                time.sleep(1)
                continue
                
            print(f"Path found! Length: {len(actions)} steps")
            print(f"Actions: {actions} (0=Left, 1=Right, 2=Forward)")
            
            # Execute the actions to verify visually
            for i, action in enumerate(actions):
                # Step the env
                obs, reward, term, trunc, info, reached_goal = env.step(action)
                
                env.render()
                time.sleep(0.1) # Delay so you can watch it move
                
                if reached_goal:
                    print(f"SUCCESS! Reached goal at step {i+1}.")
                    break
                    
            if not reached_goal:
                print("Failed to reach goal after executing path. (Logic mismatch?)")
                
            time.sleep(1) # Pause between episodes

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        env.close()