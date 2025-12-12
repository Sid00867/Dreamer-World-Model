import numpy as mp
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.jit
from environment_variables import *

class rssm(nn.Module):
    def __init__(self):
        super().__init__()

        self.obs_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.AdaptiveAvgPool2d((1,1)), 
            nn.Flatten(),
        )

        self.embed_dim = 128 

        # Posterior heads
        concat_dim = self.embed_dim + deterministic_dim
        self.mu = nn.Linear(concat_dim, latent_dim)
        self.log_sigma = nn.Linear(concat_dim, latent_dim)

        # gru
        self.gru = nn.GRUCell(latent_dim + action_dim, deterministic_dim)

        # decoder
        self.fc = nn.Sequential(
            nn.Linear(latent_dim + deterministic_dim, 128 * 3 * 3),
            nn.Dropout(0.3)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128,128,3,1,1), nn.ReLU(),       
            nn.ConvTranspose2d(128,64,4,2,1, output_padding=1), nn.ReLU(), 
            nn.ConvTranspose2d(64,32,4,2,1),   nn.ReLU(),      
            nn.ConvTranspose2d(32,3,4,2,1),                     
            nn.Sigmoid()
        )

        # reward model
        self.reward_model = nn.Sequential(
            nn.Linear(latent_dim + deterministic_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )

        # prior model
        self.prior_fc = nn.Sequential(
            nn.Linear(deterministic_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
        )

        self.prior_mu = nn.Linear(128, latent_dim)
        self.prior_log_sigma = nn.Linear(128, latent_dim)


    def forward_train(self, h_prev, s_prev, a_prev, o_embed):
        # gru
        gru_input = torch.cat([s_prev, a_prev], dim=-1)   
        h_t = self.gru(gru_input, h_prev)                    

        # posterior
        concat_post = torch.cat([o_embed, h_t], dim=-1)  
        mu_post = self.mu(concat_post)                      
        log_sigma_post = self.log_sigma(concat_post)        
        sigma_post = torch.exp(log_sigma_post)
        s_t_post = mu_post + sigma_post * torch.randn_like(sigma_post)  

        # prior
        prior_hidden = self.prior_fc(h_t)                    
        mu_prior = self.prior_mu(prior_hidden)               
        log_sigma_prior = self.prior_log_sigma(prior_hidden) 

        # decoder
        dec_input = torch.cat([s_t_post, h_t], dim=-1)       
        x = self.fc(dec_input)                               
        x = x.view(-1, 128, 3, 3)                            
        o_recon = self.decoder(x)                          

        # reward
        reward_input = torch.cat([s_t_post, h_t], dim=-1)   
        reward_pred = self.reward_model(reward_input).squeeze(-1)

        return (mu_post, log_sigma_post), (mu_prior, log_sigma_prior), o_recon, reward_pred, h_t, s_t_post
    
    @torch.jit.export
    def rollout_horizon(self, h, s, actions):
        """
        Fast loop for planning
        """
        rewards = torch.zeros(actions.size(0), device=h.device)
        
        for t in range(actions.size(1)):
            act = actions[:, t]
            
            gru_input = torch.cat([s, act], dim=-1)
            h = self.gru(gru_input, h)
            
            prior_hidden = self.prior_fc(h)
            mu_prior = self.prior_mu(prior_hidden)
            s = mu_prior 

            reward_input = torch.cat([s, h], dim=-1)
            r = self.reward_model(reward_input).squeeze(-1)
            
            rewards += r
            
        return rewards
    
    
    def imagine_step(self, h, s, a, sample=False):
        gru_input = torch.cat([s, a], dim=-1)    
        h_next = self.gru(gru_input, h)          

        prior_hidden = self.prior_fc(h_next)
        mu_prior = self.prior_mu(prior_hidden)
        log_sigma_prior = self.prior_log_sigma(prior_hidden)

        if sample:
            sigma = torch.exp(log_sigma_prior)
            s_next = mu_prior + sigma * torch.randn_like(sigma)
        else:
            s_next = mu_prior

        return h_next, s_next
    
    
    def reward(self, s, h):
        reward_input = torch.cat([s, h], dim=-1)
        return self.reward_model(reward_input).squeeze(-1)