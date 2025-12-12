# import torch
# from gridworldenv import RLReadyEnv 

# #architectural parameters
# latent_dim = 32
# deterministic_dim = 100
# action_dim = 3  

# #observations
# observation_dim = 28*28*3 #assuming rgb images of size 28x28
# obs_shape=(3, 28, 28)

# #utility parameters
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# checkpoint_weights_save_iters = 5000
# weights_path = "rssm_latest.pth"

# #training parameters
# learnrate = 0.001
# log_sigma_clamp = 5 
# beta = 1e-4 #kl divergence weight
# grad_clipping_value = 10.0

# #planning parameters
# planning_horizon = 15
# optimization_iters = 10
# candidates = 1000
# K = 100

# #model fitting parameters
# C = 100
# batch_size = 32
# seq_len = 20

# #exploration parameters (data collection cycle)
# total_env_steps = 5000
# exploration_noise = 0.15
# action_repeat = 2

# #replay buffer parameters
# replay_buffer_capacity = 10000
# max_episode_len = 200
# seed_replay_buffer_episodes = 20

# #metrics config
# metrics_storage_window = 1000
# small_metric_window = 200

# loss_eps=1e-4
# recon_eps=1e-4
# psnr_eps=0.05
# min_success=0.85
# min_steps=10000
# max_steps=100_000 #overall training steps

# #environment variables (shifted to gridworldenv.py due to circular import issues)
# # env_grid_size_ = 10       
# # tile_size_ = 4        # agent view - 7x7, tiling - 4x4, 4x7 = 28 so 28x28x3 images
# # see_through_walls_ = False    

# # training environment setup

# # change seed for different spawn locations for the agent

# env = RLReadyEnv(
#         env_kind= "simple",
#         size = 10,

#         # obs config
#         obs_mode = "rgb",
#         obs_scope = "partial",  #agent sight - 7x7

#         render_mode=None,
#         agent_start_pos= (1, 1),
#         agent_start_dir=0,
#         max_steps= None,
#         seed= 82,
# )

# #playtest environment setup
# def make_play_env():
#     return RLReadyEnv(
#         env_kind="simple",
#         size=10,
#         obs_mode="rgb",
#         obs_scope="partial",
#         render_mode="human",
#         agent_start_pos=(1,1),
#         agent_start_dir=0,
#         max_steps=None,
#         seed=82,
#  )

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

planning_horizon = 32       # Reduced from 20: fast short-term planning
optimization_iters = 20      # Reduced from 15: "good enough" planning
candidates = 500            # Reduced from 500: less compute per step
K = 50                      # Reduced from 50: fit to top 10%

# ======================================================
# MODEL FITTING (CPU LIGHTWEIGHT)
# ======================================================

C = 20                       # Reduced from 20: Train less, act more
batch_size = 16              # Reduced from 16: Fits in CPU cache better
seq_len = 20                # Reduced from 25: Faster backprop

# ======================================================
# EXPLORATION
# ======================================================

total_env_steps = 2000      # Short data collection cycles
exploration_noise = 0.15    # Higher noise to find goals quickly
action_repeat = 2

# ======================================================
# REPLAY BUFFER
# ======================================================

replay_buffer_capacity = 2000   
max_episode_len = 100        # Short episodes (if not solved in 50, fail)
seed_replay_buffer_episodes = 20 # Quick start

# ======================================================
# METRICS (EARLY STOPPING)
# ======================================================

metrics_storage_window = 100
small_metric_window = 20

loss_eps = 1e-4
recon_eps = 1e-4
psnr_eps = 0.05
min_success = 0.20          # Stop if we hit 20% success (proof of learning)
min_steps = 250            # Minimum interactions
max_steps = 10000            # Hard stop after ~5-10 mins

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