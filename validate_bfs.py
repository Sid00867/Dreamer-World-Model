import time
import numpy as np
from gridworldenv import RLReadyEnv
from minigrid.core.world_object import Goal


def bfs_solve(env):

    grid = env.env.unwrapped.grid
    start_pos = env.env.unwrapped.agent_pos
    start_dir = env.env.unwrapped.agent_dir 
    goal_pos = env.goal_pos

    queue = [(start_pos[0], start_pos[1], start_dir, [])]
    

    visited = set()
    visited.add((start_pos[0], start_pos[1], start_dir))
    
    print(f"BFS Start: {start_pos} -> Goal: {goal_pos}")
    
    while queue:
        x, y, d, path = queue.pop(0)

        if (x, y) == goal_pos:
            return path 
        
        if len(path) > 50: 
            continue 

        new_d = (d - 1) % 4
        if (x, y, new_d) not in visited:
            visited.add((x, y, new_d))
            queue.append((x, y, new_d, path + [0]))
     
        new_d = (d + 1) % 4
        if (x, y, new_d) not in visited:
            visited.add((x, y, new_d))
            queue.append((x, y, new_d, path + [1]))

        fx, fy = x, y
        if d == 0: fx += 1   
        elif d == 1: fy += 1 
        elif d == 2: fx -= 1 
        elif d == 3: fy -= 1 
        
        # Check grid bounds
        if 0 <= fx < grid.width and 0 <= fy < grid.height:
            cell = grid.get(fx, fy)
            # Check if walkable (None = Empty, or explicit Goal object)
            if cell is None or isinstance(cell, Goal) or (goal_pos and (fx, fy) == goal_pos):
                if (fx, fy, d) not in visited:
                    visited.add((fx, fy, d))
                    queue.append((fx, fy, d, path + [2]))
                    
    return []


if __name__ == "__main__":
    
  
    env = RLReadyEnv(
        env_kind="simple",
        size=10,
        obs_mode="rgb",
        obs_scope="partial",
        render_mode="human",
        agent_start_pos=(1, 1),
        agent_start_dir=0,
        max_steps=None
    )

    print("Environment Initialized. Starting BFS Validation Loop...")
    
    try:
        for episode in range(10): 
            print(f"\n--- Episode {episode + 1} ---")
            
            obs = env.reset()
            env.render()
            
            print("Solving...")
            actions = bfs_solve(env)
            
            if not actions:
                print("No path found! (Map might be impossible or goal blocked)")
                time.sleep(1)
                continue
                
            print(f"Path found! Length: {len(actions)} steps")
            print(f"Actions: {actions} (0=Left, 1=Right, 2=Forward)")
            
            for i, action in enumerate(actions):
                obs, reward, term, trunc, info, reached_goal = env.step(action)
                
                env.render()
                time.sleep(0.1)
                
                if reached_goal:
                    print(f"SUCCESS! Reached goal at step {i+1}.")
                    break
                    
            if not reached_goal:
                print("Failed to reach goal after executing path. (Logic mismatch?)")
                
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        env.close()