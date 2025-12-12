import torch
from gridworldenv import RLReadyEnv 

# ======================================================
# ARCHITECTURE (leave unchanged)
# ======================================================

latent_dim = 64
deterministic_dim = 200
action_dim = 3 

obs_shape = (3, 28, 28)
observation_dim = 28 * 28 * 3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

weights_path = "rssm_latest.pth"

# ======================================================
# TRAINING (CPU OPTIMIZED)
# ======================================================

learnrate = 1e-3            # Higher LR for faster (rougher) convergence
log_sigma_clamp = 5
beta = 1e-5
grad_clipping_value = 10.0

# ======================================================
# PLANNING (CPU LIGHTWEIGHT)
# ======================================================

planning_horizon = 20       # Reduced from 20: fast short-term planning
optimization_iters = 20      # Reduced from 15: "good enough" planning
candidates = 2000            # Reduced from 500: less compute per step
K = 100                      # Reduced from 50: fit to top 10%

# planning_horizon = 20       # Reduced from 20: fast short-term planning
# optimization_iters = 10      # Reduced from 15: "good enough" planning
# candidates = 500            # Reduced from 500: less compute per step
# K = 50                      # Reduced from 50: fit to top 10%

# ======================================================
# MODEL FITTING (CPU LIGHTWEIGHT)
# ======================================================

C = 150                       # Reduced from 20: Train less, act more
batch_size = 64              # Reduced from 16: Fits in CPU cache better
seq_len = 8                # Reduced from 25: Faster backprop

# ======================================================
# EXPLORATION
# ======================================================

total_env_steps = 200      # Short data collection cycles
exploration_noise = 0.15    # Higher noise to find goals quickly
action_repeat = 2

# ======================================================
# REPLAY BUFFER
# ======================================================

replay_buffer_capacity = 5000   
max_episode_len = 200        # Short episodes (if not solved in 50, fail)
seed_replay_buffer_episodes = 20 # Quick start

# ======================================================
# METRICS (EARLY STOPPING)
# ======================================================

metrics_storage_window = 1000
small_metric_window = 200

loss_eps = 1e-4
recon_eps = 1e-4
psnr_eps = 0.05
min_success = 0.20          # Stop if we hit 20% success (proof of learning)
min_steps = 250            # Minimum interactions
max_steps = 10000            # Hard stop after ~5-10 mins

# ======================================================
raw_freq = int(max_steps / total_env_steps / 10)
weight_save_freq_for_outer_iters = max(1, raw_freq)
# ======================================================
# TRAIN ENV
# ======================================================

env = RLReadyEnv(
    env_kind="simple",
    size=10,
    obs_mode="rgb",
    obs_scope="partial",
    render_mode=None,
    agent_start_pos=(1, 1),
    agent_start_dir=0,
    max_steps=None,
    seed=82,
)

# ======================================================
# PLAY ENV
# ======================================================

def make_play_env():
    return RLReadyEnv(
        env_kind="simple",
        size=10,
        obs_mode="rgb",
        obs_scope="partial",
        render_mode="human",
        agent_start_pos=(1,1),
        agent_start_dir=0,
        max_steps=None,
        seed=33,
    )