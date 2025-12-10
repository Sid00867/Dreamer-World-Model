# Visualizes a trained RSSM + planner agent playing in the environment.

import time
import torch

from environment_variables import *
from rssm import rssm
from planner import plan
from explore_sample import preprocess_obs


MODEL_PATH = weights_path  
NUM_EPISODES = 5
STEP_DELAY = 0.05   # seconds between frames


rssmmodel = rssm().to(DEVICE)
rssmmodel.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
rssmmodel.eval()


def play():

    for ep in range(NUM_EPISODES):

        print(f"\nEpisode {ep + 1}/{NUM_EPISODES}")

        obs_raw, _ = playenv.reset()
        obs = preprocess_obs(obs_raw)

        h = torch.zeros(1, deterministic_dim, device=DEVICE)
        s = torch.zeros(1, latent_dim, device=DEVICE)

        done = False
        step = 0

        while not done:

            with torch.no_grad():

                a_onehot = plan(h, s).unsqueeze(0)

                action = a_onehot.argmax(-1).item()
                
                obs_next_raw, reward, terminated, truncated, info = playenv.step(action)
                done = terminated or truncated

                obs_next = preprocess_obs(obs_next_raw)

                (mu_post, _), _, _, _, h = rssmmodel.forward_train(
                    h, a_onehot, obs_next.unsqueeze(0)
                )

                s = mu_post

                playenv.render()
                time.sleep(STEP_DELAY)

                obs = obs_next
                step += 1

        print(f"Episode finished in {step} steps")


if __name__ == "__main__":
    play()
