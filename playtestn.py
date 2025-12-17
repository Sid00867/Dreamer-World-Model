import time
import torch
import numpy as np
import imageio
import matplotlib.pyplot as plt

from environment_variables import *
from gridworldenv import RLReadyEnv  # Import class directly to override config
from planner import plan, reset_planner
from explore_sample import preprocess_obs
from fitter import rssmmodel, actor_net 

# --- CONFIGURATION ---
MODEL_PATH = "rssm_final.pth"
GIF_FILENAME = "agent_gameplay_global.gif"
RECORD_EPISODES = 3
MAX_STEPS_PER_EP = 100

# 1. Setup Environment with GLOBAL RENDER
# We set render_mode="rgb_array" to get the full map image
playenv = RLReadyEnv(
    env_kind="simple",
    size=10,
    obs_mode="rgb",
    obs_scope="partial",     # Agent still sees partial (blind) view
    render_mode="rgb_array", # <-- We get the FULL view for the GIF
    seed=42
)

# 2. Load Weights
print(f"Loading weights from {MODEL_PATH}...")
if torch.cuda.is_available():
    checkpoint = torch.load(MODEL_PATH)
else:
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")

rssmmodel.load_state_dict(checkpoint['rssm'])
actor_net.load_state_dict(checkpoint['actor'])

rssmmodel.eval()
actor_net.eval()

def play_and_record():
    all_frames = []
    
    for ep in range(RECORD_EPISODES):
        print(f"Recording Episode {ep + 1}/{RECORD_EPISODES}...")
        
        reset_planner()
        obs_raw, _ = playenv.reset()
        
        # Capture Initial Frame (Global View)
        # Note: In MiniGrid, env.render() returns the full grid image
        global_view = playenv.render()
        all_frames.append(global_view)
        
        obs = preprocess_obs(obs_raw)

        # Initialize State
        dummy_action = torch.zeros(1, action_dim, device=DEVICE)
        obs_embed = rssmmodel.obs_encoder(obs.unsqueeze(0))
        
        _, _, _, _, h, s = rssmmodel.forward_train(
            h_prev=torch.zeros(1, deterministic_dim, device=DEVICE),
            s_prev=torch.zeros(1, latent_dim, device=DEVICE),
            a_prev=dummy_action,
            o_embed=obs_embed
        )

        done = False
        step = 0

        while not done and step < MAX_STEPS_PER_EP:
            with torch.no_grad():
                # --- AGENT LOGIC (PARTIAL VIEW) ---
                state_features = torch.cat([s, h], dim=-1)
                
                # Plan Action
                a_onehot = plan(h, s)
                if a_onehot.dim() == 1: a_onehot = a_onehot.unsqueeze(0)
                action = a_onehot.argmax(-1).item()

                # --- ENV STEP ---
                obs_next_raw, reward, terminated, truncated, info, _ = playenv.step(action)
                done = terminated or truncated

                # --- RECORD FRAME (GLOBAL VIEW) ---
                global_view = playenv.render()
                all_frames.append(global_view)

                # --- UPDATE RSSM ---
                obs_next = preprocess_obs(obs_next_raw)
                obs_embed = rssmmodel.obs_encoder(obs_next.unsqueeze(0))

                _, _, _, _, h, s = rssmmodel.forward_train(
                    h_prev=h, s_prev=s, a_prev=a_onehot, o_embed=obs_embed
                )

                obs = obs_next
                step += 1

        print(f"  Finished in {step} steps. Reward: {reward}")

    # Save GIF
    print(f"Saving {len(all_frames)} frames to {GIF_FILENAME}...")
    try:
        # Loop=0 means infinite loop
        imageio.mimsave(GIF_FILENAME, all_frames, fps=8, loop=0) 
        print(f"✅ Saved! Open {GIF_FILENAME} to see the full grid view.")
    except Exception as e:
        print(f"❌ Error saving GIF: {e}")

if __name__ == "__main__":
    play_and_record()