import torch
from gridworldenv import RLReadyEnv 

# ======================================================
# ARCHITECTURE (DREAMER V2 CONFIG)
# ======================================================

# V2 uses a matrix of Categorical variables (32 variables, 32 classes each)
stoch_size = 32 
class_size = 32 

# Flattened size for the Linear layers (32 * 32 = 1024)
latent_dim = stoch_size * class_size 
deterministic_dim = 200
action_dim = 3 

# 28x28 GridWorld
obs_shape = (3, 28, 28)
observation_dim = 28 * 28 * 3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
weights_path = "rssm_latest.pth"

# ======================================================
# TRAINING
# ======================================================

learnrate = 6e-4            
grad_clipping_value = 100.0 

# KL BALANCING (Crucial for V2)
# We prioritize training the Prior (prediction) over the Posterior (representation)
kl_balance = 0.9 
kl_scale = 1.0    # V2 usually uses a smaller scale like 0.1 or 1.0

# ======================================================
# PLANNING
# ======================================================

actor_lr = 8e-5         
value_lr = 8e-5
imagination_horizon = 20

gamma = 0.99
lambda_ = 0.95 

actor_entropy_scale = 1e-3

# ======================================================
# MODEL FITTING
# ======================================================

C = 75                      
batch_size = 50              
seq_len = 8                

# ======================================================
# EXPLORATION
# ======================================================

total_env_steps = 2000        
exploration_noise = 0.0     # Not needed, we rely on Actor Entropy

# ======================================================
# REPLAY BUFFER
# ======================================================

replay_buffer_capacity = 10000  
max_episode_len = 175       
seed_replay_buffer_episodes = 5 

# ======================================================
# METRICS & STOPPING
# ======================================================

metrics_storage_window = 2000
small_metric_window = 500

loss_eps = 1e-5             
recon_eps = 1e-5
psnr_eps = 0.01
min_success = 0.90          # Aiming higher now
min_steps = 5000            
max_steps = 1500000          

# ======================================================
# SAVE FREQUENCY
# ======================================================
raw_freq = int(max_steps / total_env_steps / 10)
weight_save_freq_for_outer_iters = max(1, raw_freq)

# ======================================================
# ENV CONFIG
# ======================================================
# (Keeping your SimpleEnv config)
env = RLReadyEnv(
    env_kind="simple",
    size=10,
    obs_mode="rgb",
    obs_scope="partial",
    render_mode=None,
    seed=None 
)

def make_play_env():
    return RLReadyEnv(
        env_kind="simple",
        size=10,
        obs_mode="rgb",
        obs_scope="partial",
        render_mode="human",
        seed=33
    )