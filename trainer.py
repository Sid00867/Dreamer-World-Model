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

# --- SIMPLE GEOMETRIC SOLVER (For MovingSquareEnv) ---
def solve_geometric(env):
    """
    Returns a list of actions to move the square to the goal.
    Actions: 0:Up, 1:Down, 2:Left, 3:Right
    """
    # Access positions directly from the simple env
    curr = env.agent_pos.copy()
    goal = env.goal_pos
    path = []
    
    # Limit max steps to prevent infinite loops (though unlikely here)
    for _ in range(50):
        if np.array_equal(curr, goal):
            break
            
        dy = goal[0] - curr[0]
        dx = goal[1] - curr[1]
        
        # Move along the larger distance axis
        if abs(dx) >= abs(dy):
            if dx > 0: 
                path.append(3) # Right
                curr[1] += 1
            else:      
                path.append(2) # Left
                curr[1] -= 1
        else:
            if dy > 0: 
                path.append(1) # Down
                curr[0] += 1
            else:      
                path.append(0) # Up
                curr[0] -= 1
                
    return path

def seed_smart_wins(num_episodes=20):
    print(f"Generating {num_episodes} SMART WINNING episodes (Geometric Solved)...")
    wins = 0
    
    while wins < num_episodes:
        reset_planner()
        obs_raw, _ = env.reset() 
        obs = preprocess_obs(obs_raw)
        
        # 1. Solve (Simple calculation)
        solution_actions = solve_geometric(env)
        
        # 2. Execute
        done = False
        for action in solution_actions:
            a_onehot = F.one_hot(torch.tensor(action), num_classes=action_dim).float().to(DEVICE)
            
            # Step the environment
            obs_next_raw, r, done, _, _, reached_goal = env.step(action)
            
            # Add to buffer
            buffer.add_step(obs.cpu(), a_onehot.cpu(), r, done)
            
            obs = preprocess_obs(obs_next_raw)
            if done: break
            
        # Verify it actually worked (it should always work in this env)
        if done and r > 0: 
            wins += 1
            # print(f"  -> Seeded Win #{wins} (Length: {len(solution_actions)})")

buffer = ReplayBuffer(
    capacity_episodes=replay_buffer_capacity,
    max_episode_len=max_episode_len,
    obs_shape=obs_shape,
    action_dim=action_dim,
    device=DEVICE
)

def convergence_trainer():
    outer_iter = 0
    # Increase outer loop speed by reducing total_env_steps in env_vars
    with tqdm(total=max_steps, desc="Sanity Check Loop") as pbar:

        while not METRICS.has_converged():
            outer_iter += 1

            # Seed winning demonstrations periodically to keep the buffer "happy"
            if outer_iter % 10 == 0:
                seed_smart_wins(num_episodes=2)

            train_sequence(
                C=C,
                dataset=buffer, 
                batch_size=batch_size,
                seq_len=seq_len,
            )

            run_data_collection(buffer, pbar)
            
            stats = METRICS.get_means()
            pbar.set_postfix({
                "L": f"{stats['loss_total']:.4f}",
                "Rec": f"{stats['recon_loss']:.4f}", # Watch this drop to 0.0000
                "Ret": f"{stats['return']:.1f}", 
            })

            if outer_iter % weight_save_freq_for_outer_iters == 0:
                torch.save({
                    'rssm': rssmmodel.state_dict(),
                    'actor': actor_net.state_dict(),
                    'critic': critic_net.state_dict()
                }, "rssm_sanity.pth")

if __name__ == "__main__":
    # 1. Fill buffer with some perfect games initially
    seed_smart_wins(num_episodes=10) 
    
    # 2. Start Training
    convergence_trainer()
    
    # 3. Final Save
    torch.save({
        'rssm': rssmmodel.state_dict(),
        'actor': actor_net.state_dict(),
        'critic': critic_net.state_dict()
    }, "rssm_sanity.pth")

    print("Sanity Check Complete.")