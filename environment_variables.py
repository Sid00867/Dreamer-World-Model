import torch
from gridworldenv import RLReadyEnv 

# ======================================================
# ARCHITECTURE (leave unchanged)
# ======================================================

latent_dim = 64
deterministic_dim = 200
action_dim = 3 

# Ensure this matches your TILE_SIZE setting (4 -> 28, 8 -> 56)
obs_shape = (3, 28, 28)
observation_dim = 28 * 28 * 3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

weights_path = "rssm_latest.pth"

# ======================================================
# TRAINING (ROBUST GENERALIZATION)
# ======================================================

learnrate = 1e-3            
log_sigma_clamp = 5
beta = 1e-5                 # Low KL penalty to prioritize sharp reconstruction of new maps
grad_clipping_value = 100.0 # Allow strong gradients to learn physics quickly

# ======================================================
# PLANNING (DENSE SEARCH)
# ======================================================

actor_lr = 8e-5         
value_lr = 8e-5
imagination_horizon = 20

#lambda returns
gamma = 0.99
lambda_=0.95 #95 percent trust in reality over dreamt predictions in bellman optimality equatiob

actor_entropy_scale = 1e-3

# ======================================================
# MODEL FITTING (HEAVY DUTY)
# ======================================================

# Train MORE per cycle because every episode is "new" info
C = 75                      
batch_size = 64              
seq_len = 8                # Longer memory to handle navigation/backtracking

# ======================================================
# EXPLORATION (AGGRESSIVE)
# ======================================================

total_env_steps = 1000  #SHOULD NOT GO BELOW 400       
exploration_noise = 0.15    # High noise prevents getting stuck in random corners

# ======================================================
# REPLAY BUFFER (LONG TERM)
# ======================================================

replay_buffer_capacity = 12000  # Store more history of different maps 
max_episode_len = 150       
seed_replay_buffer_episodes = 20 

# ======================================================
# METRICS & STOPPING
# ======================================================

metrics_storage_window = 8000
small_metric_window = 500

loss_eps = 1e-5             # Tighter convergence needed
recon_eps = 1e-5
psnr_eps = 0.01
min_success = 0.85          # Expect slightly lower success on truly random hard maps
min_steps = 5000            
max_steps = 120000          # 10x longer training for generalization

# ======================================================
# SAVE FREQUENCY (SAFE)
# ======================================================
raw_freq = int(max_steps / total_env_steps / 20)
weight_save_freq_for_outer_iters = max(1, raw_freq)

# ======================================================
# TRAIN ENV (RANDOMIZED)
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
    # seed=82, <--- REMOVED FIXED SEED to allow random generation
)

# ======================================================
# PLAY ENV (RANDOMIZED TEST)
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
        seed=33, # Keep fixed seed for playtest CONSISTENCY only
    )