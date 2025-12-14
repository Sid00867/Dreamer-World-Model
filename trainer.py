from environment_variables import *
import torch
import torch.nn.functional as F
from tqdm import tqdm
from fitter import train_sequence, rssmmodel, actor_net, critic_net
from explore_sample import run_data_collection, preprocess_obs
from replaybuffer import ReplayBuffer
from metrics import METRICS, plot_metrics
from planner import reset_planner
import numpy as np
from minigrid.core.world_object import Wall, Goal

# --- BFS SOLVER FOR SMART SEEDING ---
def bfs_solve(env):
    """
    Returns a list of actions [int] to walk from agent_pos to goal_pos
    navigating around walls.
    Actions: 0=Left, 1=Right, 2=Forward
    """
    grid = env.env.unwrapped.grid
    start_pos = env.env.unwrapped.agent_pos
    start_dir = env.env.unwrapped.agent_dir # 0=East, 1=South, 2=West, 3=North
    goal_pos = env.goal_pos
    
    # Queue: (x, y, dir, path_of_actions)
    queue = [(start_pos[0], start_pos[1], start_dir, [])]
    visited = set()
    visited.add((start_pos[0], start_pos[1], start_dir))
    
    while queue:
        x, y, d, path = queue.pop(0)
        
        if (x, y) == goal_pos:
            return path # Found it!
        
        if len(path) > 30: continue # Limit depth
        
        # Try 3 possible moves: Left, Right, Forward
        
        # 1. Turn Left (Action 0)
        new_d = (d - 1) % 4
        if (x, y, new_d) not in visited:
            visited.add((x, y, new_d))
            queue.append((x, y, new_d, path + [0]))
            
        # 2. Turn Right (Action 1)
        new_d = (d + 1) % 4
        if (x, y, new_d) not in visited:
            visited.add((x, y, new_d))
            queue.append((x, y, new_d, path + [1]))
            
        # 3. Move Forward (Action 2)
        # Get front cell
        fx, fy = x, y
        if d == 0: fx += 1
        elif d == 1: fy += 1
        elif d == 2: fx -= 1
        elif d == 3: fy -= 1
        
        # Check bounds and walls
        if 0 <= fx < grid.width and 0 <= fy < grid.height:
            cell = grid.get(fx, fy)
            if cell is None or isinstance(cell, Goal): # Walkable
                if (fx, fy, d) not in visited:
                    visited.add((fx, fy, d))
                    queue.append((fx, fy, d, path + [2]))
                    
    return [] # No path found

def seed_smart_wins(num_episodes=20):
    print(f"Generating {num_episodes} SMART WINNING episodes (BFS Solved)...")
    wins = 0
    
    while wins < num_episodes:
        reset_planner()
        obs_raw, _ = env.reset() 
        obs = preprocess_obs(obs_raw)
        
        # 1. Solve
        solution_actions = bfs_solve(env)
        
        # CRITICAL FIX: Skip if path is too short for our sequence length
        if len(solution_actions) < seq_len:
            continue 
            
        # 2. Execute
        done = False
        for action in solution_actions:
            a_onehot = F.one_hot(torch.tensor(action), action_dim).float().to(DEVICE)
            obs_next_raw, r, terminated, truncated, info, reached_goal = env.step(action)
            done = terminated or truncated or reached_goal
            
            buffer.add_step(obs.cpu(), a_onehot.cpu(), r, done)
            obs = preprocess_obs(obs_next_raw)
            if done: break
            
        if done and r > 0: 
            wins += 1
            print(f"  -> Seeded Win #{wins} (Length: {len(solution_actions)})")

buffer = ReplayBuffer(
    capacity_episodes=replay_buffer_capacity,
    max_episode_len=max_episode_len,
    obs_shape=obs_shape,
    action_dim=action_dim,
    device=DEVICE
)

def convergence_trainer():
    outer_iter = 0
    with tqdm(total=max_steps, desc="Convergence Loop") as pbar:

        while not METRICS.has_converged():
            outer_iter += 1

            # TRAIN with GOLDEN SAMPLING
            # 20% of every batch will be drawn from known wins

            if outer_iter % max(max_steps / total_env_steps / 20) == 0:
                seed_smart_wins(num_episodes=5)

            train_sequence(
                C=C,
                dataset=buffer, # Pass the buffer instance
                batch_size=batch_size,
                seq_len=seq_len,
            )

            run_data_collection(buffer, pbar)
            
            # ... (Rest of logging code remains the same) ...
            stats = METRICS.get_means()
            pbar.set_postfix({
                "Loss": f"{stats['loss_total']:.4f}",
                "Act": f"{stats['loss_actor']:.3f}",
                "Ret": f"{stats['return']:.1f}", 
                "Succ": f"{100*stats['success_rate']:.0f}%",
            })

            # Hardcoded save path as requested
            if outer_iter % weight_save_freq_for_outer_iters == 0:
                torch.save({
                    'rssm': rssmmodel.state_dict(),
                    'actor': actor_net.state_dict(),
                    'critic': critic_net.state_dict()
                }, "rssm_final.pth")

if __name__ == "__main__":
    # seed_replay_buffer() # <-- DELETE OLD
    seed_smart_wins(num_episodes=25) # <-- INSERT NEW SMART SEEDER
    convergence_trainer()
    
    torch.save({
        'rssm': rssmmodel.state_dict(),
        'actor': actor_net.state_dict(),
        'critic': critic_net.state_dict()
    }, "rssm_final.pth")

    print("Saving metrics to training_log.txt...")
    final_stats = METRICS.get_means()
    with open("training_log.txt", "w") as f:
        f.write("=== FINAL TRAINING METRICS ===\n")
        for key, value in final_stats.items():
            f.write(f"{key}: {value}\n")
    print("Done.")