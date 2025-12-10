import torch
from gridworldenv import RLReadyEnv 

#architectural parameters
latent_dim = 32
deterministic_dim = 100
action_dim = 7  #assuming 7 discrete actions in the environment

#observations
observation_dim = 28*28*3 #assuming rgb images of size 28x28
obs_shape=(3, 28, 28)

#utility parameters
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint_weights_save_iters = 5000
weights_path = "rssm_latest.pth"

#training parameters
learnrate = 0.001
log_sigma_clamp = 5 
beta = 1e-4 #kl divergence weight
grad_clipping_value = 10.0

#planning parameters
planning_horizon = 15
optimization_iters = 10
candidates = 1000
K = 100

#model fitting parameters
C = 100
batch_size = 32
seq_len = 20

#exploration parameters (data collection cycle)
total_env_steps = 5000
exploration_noise = 0.15
action_repeat = 2

#replay buffer parameters
replay_buffer_capacity = 10000
max_episode_len = 200
seed_replay_buffer_episodes = 20

#metrics config
metrics_storage_window = 1000
small_metric_window = 200

loss_eps=1e-4,
recon_eps=1e-4,
psnr_eps=0.05,
min_success=0.85,
min_steps=10000,
max_steps=100_000 #overall training steps

#environment variables (shifted to gridworldenv.py due to circular import issues)
# env_grid_size_ = 10       
# tile_size_ = 4        # agent view - 7x7, tiling - 4x4, 4x7 = 28 so 28x28x3 images
# see_through_walls_ = False    

# training environment setup

# change seed for different spawn locations for the agent

env = RLReadyEnv(
        env_kind= "simple",
        size = 10,

        # obs config
        obs_mode = "rgb",
        obs_scope = "partial",  #agent sight - 7x7

        render_mode=None,
        agent_start_pos= (1, 1),
        agent_start_dir=0,
        max_steps= None,
        seed= 82,
)

#playtest environment setup
playenv = RLReadyEnv(
        env_kind= "simple",
        size = 10,
        # obs config
        obs_mode = "rgb",
        obs_scope = "partial",  #agent sight - 7x7

        render_mode='human',
        agent_start_pos= (1, 1),
        agent_start_dir=0,
        max_steps= None,
        seed= 82,
)    