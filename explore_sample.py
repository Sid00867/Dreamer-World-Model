import torch
import torch.nn.functional as F

from environment_variables import *
from replaybuffer import ReplayBuffer
from fitter import rssmmodel
from planner import plan, reset_planner

from metrics_hooks import log_environment_step


def preprocess_obs(obs):
    if isinstance(obs, dict):
        obs = obs["image"]

    obs = torch.tensor(obs, dtype=torch.float32) / 255.0
    obs = obs.permute(2, 0, 1)
    return obs.to(DEVICE)


def run_data_collection(buffer, pbar):

    rssmmodel.eval()          

    with torch.no_grad(): 
        env_steps = 0

        obs_raw, _ = env.reset()
        obs = preprocess_obs(obs_raw)

        dummy_action = torch.zeros(1, action_dim, device=DEVICE)
        obs_embed = rssmmodel.obs_encoder(obs.unsqueeze(0))

        # Initial state
        # CORRECTED UNPACKING:
        _, _, _, _, h, s = rssmmodel.forward_train(
                    h_prev=torch.zeros(1, deterministic_dim, device=DEVICE),
                    s_prev=torch.zeros(1, latent_dim, device=DEVICE),
                    a_prev=dummy_action,
                    o_embed=obs_embed
                )
        
        done = False
        episode_len = 0
        cumulative_return = 0.0

        while env_steps < total_env_steps:

            # Plan
            a_onehot = plan(h, s)              

            # Exploration Noise
            if torch.rand(1) < exploration_noise:
                rnd = torch.randint(0, action_dim, (1,))
                a_onehot = F.one_hot(rnd, action_dim).float().to(DEVICE)

            action = a_onehot.argmax(-1).item()

            # --- ACTION STEP ---
            obs_next_raw, r, terminated, truncated, info, reached_goal = env.step(action)
            
            env_steps += 1
            pbar.update(1)
            episode_len += 1
            cumulative_return += r

            done = terminated or truncated

            # Stop if we hit the limit, but process the last frame first
            if env_steps >= total_env_steps:
                pass 
            
            obs_next = preprocess_obs(obs_next_raw)
            obs_input = obs_next.unsqueeze(0) 
            obs_embed = rssmmodel.obs_encoder(obs_input)

            # CORRECTED UNPACKING:
            # Replaced "(mu_post, _), ..." with "_, ..."
            _, _, _, _, h, s = rssmmodel.forward_train(
                h_prev=h, 
                s_prev=s, 
                a_prev=a_onehot, 
                o_embed=obs_embed
            )

            buffer.add_step(
                obs.cpu(),
                a_onehot.cpu(), 
                r,
                done
            )

            log_environment_step(
                reward=cumulative_return,
                episode_len=episode_len,
                done=done,
                action_repeat=1,
                success=reached_goal
            )

            obs = obs_next

            # Reset Logic
            if done:
                cumulative_return = 0.0
                reset_planner()
            
                obs_raw, _ = env.reset()
                obs = preprocess_obs(obs_raw)
                
                obs_input = obs.unsqueeze(0)
                obs_embed = rssmmodel.obs_encoder(obs_input)
                
                # CORRECTED UNPACKING:
                _, _, _, _, h, s = rssmmodel.forward_train(
                    h_prev=torch.zeros(1, deterministic_dim, device=DEVICE),
                    s_prev=torch.zeros(1, latent_dim, device=DEVICE),
                    a_prev=dummy_action,
                    o_embed=obs_embed
                )
                
                done = False
                episode_len = 0

        return buffer