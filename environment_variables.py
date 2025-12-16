import torch
from gridworldenv import MovingSquareEnv # <--- IMPORT THE NEW FILE

# ======================================================
# ARCHITECTURE (Fixed for 28x28)
# ======================================================
stoch_size = 32 
class_size = 32 
latent_dim = stoch_size * class_size 
deterministic_dim = 200
action_dim = 4  # Up, Down, Left, Right
obs_shape = (3, 28, 28)
observation_dim = 28 * 28 * 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
weights_path = "rssm_sanity.pth"

# ======================================================
# TRAINING (High Stability Config)
# ======================================================
learnrate = 4e-4            
grad_clipping_value = 100.0 
kl_balance = 0.9  # High balance prevents hallucinations
kl_scale = 0.1    

# ======================================================
# REST OF CONFIG
# ======================================================
actor_lr = 8e-5         
value_lr = 8e-5
imagination_horizon = 10
gamma = 0.99
lambda_ = 0.95 
actor_entropy_scale = 1e-4

C = 50                      
batch_size = 50              
seq_len = 15                

total_env_steps = 1000        
exploration_noise = 0.3     

replay_buffer_capacity = 10000  
max_episode_len = 50       
seed_replay_buffer_episodes = 5 

metrics_storage_window = 2000
small_metric_window = 500

loss_eps = 1e-5             
recon_eps = 1e-5
psnr_eps = 0.01
min_success = 0.95          
min_steps = 1000            
max_steps = 200000          

raw_freq = 100
weight_save_freq_for_outer_iters = 10

# ======================================================
# ENV HOOK
# ======================================================
# Initialize the Dummy Env
env = MovingSquareEnv()

def make_play_env():
    return MovingSquareEnv()