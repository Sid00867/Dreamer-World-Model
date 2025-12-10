import torch
import torch.nn.functional as F

from environment_variables import *
from replaybuffer import ReplayBuffer
from fitter import rssmmodel
from planner import plan

from metrics_hooks import log_environment_step



def preprocess_obs(obs):
    if isinstance(obs, dict):
        obs = obs["image"]

    obs = torch.tensor(obs, dtype=torch.float32) / 255.0
    obs = obs.permute(2, 0, 1)
    return obs.to(DEVICE)


def run_data_collection(buffer):

    rssmmodel.eval()          

    with torch.no_grad(): 
        env_steps = 0

        obs_raw, _ = env.reset()
        obs = preprocess_obs(obs_raw)

        h = torch.zeros(1, deterministic_dim, device=DEVICE)
        s = torch.zeros(1, latent_dim, device=DEVICE)

        done = False
        episode_len = 0

        while env_steps < total_env_steps:

            #getting action, adding noise

            a_onehot = plan(h, s)              
            a_onehot = a_onehot.unsqueeze(0)  

            if torch.rand(1) < exploration_noise:
                rnd = torch.randint(0, action_dim, (1,))
                a_onehot = F.one_hot(rnd, action_dim).float().to(DEVICE)

            action = a_onehot.argmax(-1).item()

            # action repeat

            reward_sum = 0
            obs_next_raw = None

            for _ in range(action_repeat):

                obs_next_raw, r, terminated, truncated, info = env.step(action)

                reward_sum += r
                env_steps += 1
                episode_len += 1

                done = terminated or truncated
                if done or env_steps >= total_env_steps:
                    break


            #getting s and h for next step iteration        
            obs_next = preprocess_obs(obs_next_raw)
            (mu_post, _), _, _, _, h = rssmmodel.forward_train(
                h, a_onehot, obs_next.unsqueeze(0)
            )
            s = mu_post      # posterior mean

            buffer.add_step(
                obs.cpu(),
                a_onehot.squeeze(0).cpu(),
                reward_sum,
                done
            )


            log_environment_step(
                reward=reward_sum,
                episode_len=episode_len,
                done=done,
                action_repeat=action_repeat,
            )


            obs = obs_next

            # env.render()

            # Reset episode 
            if done:
                obs_raw, _ = env.reset()
                obs = preprocess_obs(obs_raw)
                h = torch.zeros(1, deterministic_dim, device=DEVICE)
                s = torch.zeros(1, latent_dim, device=DEVICE)
                done = False
                episode_len = 0

        return buffer
