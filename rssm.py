import torch
import torch.nn as nn
import torch.nn.functional as F
from environment_variables import *

class rssm(nn.Module):
    def __init__(self):
        super().__init__()

        # --- ENCODER (28x28 -> 1152) ---
        self.obs_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),    
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),   
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, 1), # Preserves size
            nn.ReLU(),
            nn.Flatten(),
        )

        # Dynamic check for embedding size (approx 1152 for 28x28)
        with torch.no_grad():
            dummy = torch.zeros(1, *obs_shape)
            self.embed_dim = self.obs_encoder(dummy).shape[1]

        # --- RSSM CORE (CATEGORICAL) ---
        
        # 1. Deterministic State (GRU)
        self.gru = nn.GRUCell(latent_dim + action_dim, deterministic_dim)

        # 2. Posterior (The Eye): Observes Image -> Guesses Discrete State
        # Output: 32 * 32 logits
        self.fc_post = nn.Linear(self.embed_dim + deterministic_dim, stoch_size * class_size)

        # 3. Prior (The Dream): Uses History -> Guesses Discrete State
        # Output: 32 * 32 logits
        self.fc_prior = nn.Sequential(
            nn.Linear(deterministic_dim, 256),
            nn.ELU(), 
            nn.Linear(256, stoch_size * class_size)
        )

        # --- DECODER ---
        self.fc_dec = nn.Sequential(
            nn.Linear(latent_dim + deterministic_dim, 128 * 3 * 3),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 3, 1, 1), nn.ReLU(),       
            nn.ConvTranspose2d(128, 64, 4, 2, 1, output_padding=1), nn.ReLU(), 
            nn.ConvTranspose2d(64, 32, 4, 2, 1),   nn.ReLU(),       
            nn.ConvTranspose2d(32, 3, 4, 2, 1),                     
            nn.Sigmoid()
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
        # Reshape to (Batch, 32, 32)
        logits = logits.view(shape[0], stoch_size, class_size)
        
        # Gumbel Softmax with Straight-Through Estimator (hard=True)
        # Forward pass: Discrete One-Hot
        # Backward pass: Soft Gradients
        stoch = F.gumbel_softmax(logits, tau=1.0, hard=True)
        
        # Flatten back to (Batch, 1024)
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
        
        # 4. Sample Discrete State from Posterior (Reality)
        s_t_post = self.get_stoch_state(logits_post)

        # 5. Decode
        dec_input = torch.cat([s_t_post, h_t], dim=-1)       
        x = self.fc_dec(dec_input)                               
        x = x.view(-1, 128, 3, 3)                         
        o_recon = self.decoder(x)                          

        # 6. Predict Reward
        reward_input = torch.cat([s_t_post, h_t], dim=-1)   
        reward_pred = self.reward_model(reward_input).squeeze(-1)

        # Return logits instead of mu/sigma
        return logits_post, logits_prior, o_recon, reward_pred, h_t, s_t_post

    def imagine_rollout(self, actor, start_h, start_s, horizon):
        h = start_h
        s = start_s
        h_seq, s_seq, action_seq = [], [], []
        
        for t in range(horizon):
            state_features = torch.cat([s, h], dim=-1)
            action_logits = actor(state_features.detach()) 

            if self.training:
                # Stochastic Actor
                action_prob = F.softmax(action_logits, dim=-1)
                action_dist = torch.distributions.OneHotCategorical(probs=action_prob)
                action = action_dist.sample() 
                action = action + (action_prob - action_prob.detach()) 
            else:
                # Deterministic Actor
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