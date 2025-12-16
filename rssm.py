import torch
import torch.nn as nn
import torch.nn.functional as F
from environment_variables import *

class rssm(nn.Module):
    def __init__(self):
        super().__init__()

        # --- ENCODER (28x28 -> 128x3x3) ---
        self.obs_encoder = nn.Sequential(
            # Input: 3 x 28 x 28
            nn.Conv2d(3, 32, 4, 2, 1),    # -> 32 x 14 x 14
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),   # -> 64 x 7 x 7
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),  # -> 128 x 3 x 3
            nn.ReLU(),
            nn.Flatten(),
        )

        # Calculate embed_dim dynamically to be safe
        with torch.no_grad():
            dummy = torch.zeros(1, *obs_shape)
            self.embed_dim = self.obs_encoder(dummy).shape[1] 

        # --- RSSM CORE (DREAMERV2 / CATEGORICAL) ---
        
        # 1. Deterministic State (GRU)
        self.gru = nn.GRUCell(latent_dim + action_dim, deterministic_dim)

        # 2. Posterior (The Eye) - Observed
        self.fc_post = nn.Linear(self.embed_dim + deterministic_dim, stoch_size * class_size)

        # 3. Prior (The Dream) - Imagined
        self.fc_prior = nn.Sequential(
            nn.Linear(deterministic_dim, 256),
            nn.ELU(), 
            nn.Linear(256, stoch_size * class_size)
        )

        # --- DECODER (128x3x3 -> 28x28) ---
        # We first map the latent vector back to the spatial volume 128x3x3
        self.fc_dec = nn.Sequential(
            nn.Linear(latent_dim + deterministic_dim, 128 * 3 * 3),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            # Input: 128 x 3 x 3
            
            # Layer 1: 3x3 -> 7x7 
            # (k=3, s=2, p=0) -> (3-1)*2 - 0 + 3 = 7
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            
            # Layer 2: 7x7 -> 14x14
            # (k=4, s=2, p=1) -> (7-1)*2 - 2 + 4 = 14
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),

            # Layer 3: 14x14 -> 28x28
            # (k=4, s=2, p=1) -> (14-1)*2 - 2 + 4 = 28
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            
            # Output is strictly 3 x 28 x 28
        )

        # --- REWARD HEAD ---
        self.reward_model = nn.Sequential(
            nn.Linear(latent_dim + deterministic_dim, 256),
            nn.ELU(), 
            nn.Linear(256, 1)
        )

    def get_stoch_state(self, logits):
        """
        Input: (Batch, 1024)
        Output: (Batch, 1024) One-Hot Sample via Gumbel Softmax
        """
        shape = logits.shape
        logits = logits.view(shape[0], stoch_size, class_size)
        stoch = F.gumbel_softmax(logits, tau=1.0, hard=True)
        return stoch.reshape(shape[0], -1)

    def forward_train(self, h_prev, s_prev, a_prev, o_embed):
        # 1. Update Deterministic State
        gru_input = torch.cat([s_prev, a_prev], dim=-1)   
        h_t = self.gru(gru_input, h_prev)                    

        # 2. Compute Prior (Dream)
        logits_prior = self.fc_prior(h_t)
        
        # 3. Compute Posterior (Reality)
        concat_post = torch.cat([o_embed, h_t], dim=-1)
        logits_post = self.fc_post(concat_post)
        
        # 4. Sample Discrete State
        s_t_post = self.get_stoch_state(logits_post)

        # 5. Decode Observation
        dec_input = torch.cat([s_t_post, h_t], dim=-1)       
        x = self.fc_dec(dec_input)                               
        
        # Reshape to spatial volume
        x = x.view(-1, 128, 3, 3)                         
        o_recon = self.decoder(x)                          

        # 6. Predict Reward
        reward_input = torch.cat([s_t_post, h_t], dim=-1)   
        reward_pred = self.reward_model(reward_input).squeeze(-1)

        return logits_post, logits_prior, o_recon, reward_pred, h_t, s_t_post

    def imagine_rollout(self, actor, start_h, start_s, horizon):
        h = start_h
        s = start_s
        h_seq, s_seq, action_seq = [], [], []
        
        for t in range(horizon):
            state_features = torch.cat([s, h], dim=-1)
            action_logits = actor(state_features.detach()) 

            if self.training:
                # Stochastic Actor for training
                action_prob = F.softmax(action_logits, dim=-1)
                action_dist = torch.distributions.OneHotCategorical(probs=action_prob)
                action = action_dist.sample() 
                # Straight-through gradient estimator
                action = action + (action_prob - action_prob.detach()) 
            else:
                # Deterministic Actor for evaluation
                action = F.one_hot(action_logits.argmax(-1), action_dim).float() 

            # Step World Model
            gru_input = torch.cat([s, action], dim=-1)
            h = self.gru(gru_input, h)
            
            # Dream the next state (Prior)
            logits_prior = self.fc_prior(h)
            s = self.get_stoch_state(logits_prior)
            
            h_seq.append(h)
            s_seq.append(s)
            action_seq.append(action)
            
        return torch.stack(h_seq), torch.stack(s_seq), torch.stack(action_seq)