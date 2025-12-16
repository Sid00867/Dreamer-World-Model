import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import os

from environment_variables import *
from planner import plan, reset_planner
from explore_sample import preprocess_obs
from fitter import rssmmodel, actor_net 

# Only init env if running as main to avoid double-launch
if __name__ == "__main__":
    playenv = make_play_env()
else:
    playenv = None

MODEL_PATH = weights_path  
NUM_EPISODES = 5
STEP_DELAY = 0.05  

def showimage(image):
    if isinstance(image, np.ndarray):
        if image.ndim == 3 and image.shape[2] == 3:
            plt.imshow(image)
            plt.show()
            return

    if hasattr(image, "detach"):
        image = image.detach().cpu().numpy()
    if image.ndim == 4:
        image = image[0]
    if image.shape[0] == 3:
        image = image.transpose(1, 2, 0)
    if image.max() > 1.0:
        image = image / 255.0

    plt.imshow(image)
    plt.show()

def play(random_exp):
    if os.path.exists("rssm_final.pth"):
        checkpoint = torch.load("rssm_final.pth", map_location=DEVICE)
        rssmmodel.load_state_dict(checkpoint['rssm'])
        actor_net.load_state_dict(checkpoint['actor'])
        print("Loaded weights from rssm_final.pth")
    else:
        print("Warning: rssm_final.pth not found. Playing with random weights.")

    rssmmodel.eval()
    actor_net.eval()

    for ep in range(NUM_EPISODES):

        print(f"\nEpisode {ep + 1}/{NUM_EPISODES}")

        reset_planner()
        obs_raw, _ = playenv.reset()
        obs = preprocess_obs(obs_raw)

        dummy_action = torch.zeros(1, action_dim, device=DEVICE)
        obs_embed = rssmmodel.obs_encoder(obs.unsqueeze(0))
        
        # CORRECTED UNPACKING:
        _, _, _, _, h, s = rssmmodel.forward_train(
            h_prev=torch.zeros(1, deterministic_dim, device=DEVICE),
            s_prev=torch.zeros(1, latent_dim, device=DEVICE),
            a_prev=dummy_action,
            o_embed=obs_embed
        )

        done = False
        step = 0

        while not done:

            with torch.no_grad():
                state_features = torch.cat([s, h], dim=-1)
                logits = actor_net(state_features)
                probs = torch.nn.functional.softmax(logits, dim=-1)
                
                print(f"Action Probs: {probs[0].cpu().numpy().round(2)}, step = {step}")

                a_onehot = plan(h, s) 

                if a_onehot.dim() == 1:
                    a_onehot = a_onehot.unsqueeze(0)

                if random_exp:
                    if torch.rand(1) < 0.15: 
                        action = torch.randint(0, action_dim, (1,)).item()
                    else:
                        action = a_onehot.argmax(-1).item()
                else:
                    action = a_onehot.argmax(-1).item()

                obs_next_raw, reward, terminated, truncated, info, _ = playenv.step(action)
                done = terminated or truncated

                obs_next = preprocess_obs(obs_next_raw)
                obs_embed = rssmmodel.obs_encoder(obs_next.unsqueeze(0))

                # CORRECTED UNPACKING:
                _, _, o_recon, _, h, s = rssmmodel.forward_train(
                    h_prev=h,
                    s_prev=s,
                    a_prev=a_onehot, 
                    o_embed=obs_embed
                )

                playenv.render()
                
                # Uncomment to debug visual reconstruction
                # if step % 10 == 0:
                #    showimage(o_recon)

                obs = obs_next
                step += 1

        print(f"Episode finished in {step} steps")


if __name__ == "__main__":
    # Ensure this is FALSE to test the actual intelligence
    play(True)