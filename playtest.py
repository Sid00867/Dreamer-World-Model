# Visualizes a trained RSSM + planner agent playing in the environment.

import time
import torch

from environment_variables import *
from rssm import rssm
from planner import plan, reset_planner
from explore_sample import preprocess_obs
import matplotlib.pyplot as plt

playenv = make_play_env()

MODEL_PATH = weights_path  
NUM_EPISODES = 5
STEP_DELAY = 0.05   # seconds between frames

rssmmodel = rssm().to(DEVICE)
rssmmodel.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
rssmmodel.eval()

import numpy as np # Ensure numpy is imported

def showimage(image):
    # CASE 1: Raw Environment Observation (Numpy)
    # Shape is already (H, W, C) -> (28, 28, 3). Matplotlib loves this.
    if isinstance(image, np.ndarray):
        # If it's 3D and channels are last, just show it
        if image.ndim == 3 and image.shape[2] == 3:
            plt.imshow(image)
            plt.show()
            return
    
    # CASE 2: Model Output (PyTorch Tensor)
    # Shape is (Batch, C, H, W) or (C, H, W). Matplotlib hates this.
    if hasattr(image, "detach"):
        image = image.detach().cpu().numpy()

    # Remove batch dimension if present: (1, 3, 28, 28) -> (3, 28, 28)
    if image.ndim == 4:
        image = image[0]

    # Rearrange channels: (3, 28, 28) -> (28, 28, 3)
    if image.shape[0] == 3:
        image = image.transpose(1, 2, 0)

    # Normalize if the model output is not in 0-1 range (optional safety)
    if image.max() > 1.0:
        image = image / 255.0

    plt.imshow(image)
    plt.show()

def play(random_exp):

    for ep in range(NUM_EPISODES):

        print(f"\nEpisode {ep + 1}/{NUM_EPISODES}")

        reset_planner()
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

                if random_exp:
                    if torch.rand(1) < 0.30: # 10% chance to act randomly
                        action = torch.randint(0, action_dim, (1,)).item()
                    else:
                        action = a_onehot.argmax(-1).item()
                
                obs_next_raw, reward, terminated, truncated, info, _ = playenv.step(action)

                print(f"Step: {step}, Action: {action}, Reward: {reward:.3f}")

                done = terminated or truncated

                obs_next = preprocess_obs(obs_next_raw)

                obs_embed = rssmmodel.obs_encoder(obs_next.unsqueeze(0))

                (mu_post, _), _, o_recon, _, h, s = rssmmodel.forward_train(
                    h_prev=h,
                    s_prev=s,
                    a_prev=a_onehot, 
                    o_embed=obs_embed
                )

                # showimage(obs_next_raw['image'])
                # print(obs_next_raw['image'].shape)
                # showimage(o_recon)

                playenv.render()
                # time.sleep(STEP_DELAY)

                obs = obs_next
                step += 1

        print(f"Episode finished in {step} steps")


if __name__ == "__main__":
    play(False)