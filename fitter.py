import torch
import torch.nn.functional as F
import torch.optim as optim
from environment_variables import *
from metrics_hooks import log_training_step
from rssm import rssm
from actor_critic import ActionDecoder, ValueDecoder

rssmmodel = rssm().to(DEVICE)
actor_net = ActionDecoder().to(DEVICE)
critic_net = ValueDecoder().to(DEVICE)

model_optimizer = optim.Adam(rssmmodel.parameters(), lr=learnrate, eps=1e-5)
actor_optimizer = optim.Adam(actor_net.parameters(), lr=actor_lr, eps=1e-5)
critic_optimizer = optim.Adam(critic_net.parameters(), lr=value_lr, eps=1e-5)

def calculate_lambda_returns(rewards, values, gamma=gamma, lambda_=lambda_):
    """
    Computes the 'Lambda Target' - a mix of immediate reward and 
    long-term value estimates. This stabilizes training.
    """
    returns = torch.zeros_like(rewards)
    next_return = values[-1] # Bootstrap from last step
    
    # Iterate backwards through time
    for t in reversed(range(len(rewards) - 1)):
        r_t = rewards[t]
        v_next = values[t+1]
        
        # Bellman equation with lambda smoothing
        # Target = Reward + Gamma * ( (1-lambda)*Value_Next + lambda*Return_Next )
        returns[t] = r_t + gamma * ( (1 - lambda_) * v_next + lambda_ * next_return )
        next_return = returns[t]
        
    return returns


def compute_model_loss(o_t, o_embed, a_t, r_t, h_prev, s_prev):
    (mu_post, log_sigma_post), (mu_prior, log_sigma_prior), o_recon, reward_pred, h_t, s_t = rssmmodel.forward_train(
        h_prev, s_prev, a_t, o_embed
    )

    recon_loss  = F.mse_loss(o_recon, o_t, reduction='mean')
    reward_loss = F.mse_loss(reward_pred, r_t.squeeze(-1), reduction='mean')

    log_sigma_post  = torch.clamp(log_sigma_post,  -log_sigma_clamp, log_sigma_clamp)
    log_sigma_prior = torch.clamp(log_sigma_prior, -log_sigma_clamp, log_sigma_clamp)

    kl_loss = torch.mean(
        torch.distributions.kl_divergence(
            torch.distributions.Normal(mu_post,  torch.exp(log_sigma_post)),
            torch.distributions.Normal(mu_prior, torch.exp(log_sigma_prior))
        )
    )

    total_loss = recon_loss + reward_loss + (beta * kl_loss)

    return total_loss, recon_loss, kl_loss, reward_loss, h_t, s_t

def train_actor_critic(start_h, start_s):
    
    """
    1. Detaches state from real world (stops gradients leaking to RSSM)
    2. Imagines a future trajectory using the Actor
    3. Calculates value targets
    4. Updates Actor (maximize Value) and Critic (predict Value)
    """
    
    # Detach start states so we don't update RSSM here
    start_h = start_h.detach()
    start_s = start_s.detach()
    
    h_seq, s_seq, action_seq = rssmmodel.imagine_rollout(actor_net, start_h, start_s, horizon=imagination_horizon)
    
    # Flatten time/batch for simpler processing
    seq_len, batch, _ = h_seq.shape
    flat_h = h_seq.view(-1, deterministic_dim)
    flat_s = s_seq.view(-1, latent_dim)
    
    target_input = torch.cat([flat_s, flat_h], dim=-1)
    
    imagined_rewards = rssmmodel.reward_model(target_input).view(seq_len, batch)
    imagined_values  = critic_net(target_input).view(seq_len, batch)
    
    lambda_targets = calculate_lambda_returns(imagined_rewards, imagined_values)
    

    #CALCULATE ENTROPY
    current_logits = actor_net(target_input).view(seq_len, batch, -1)
    current_dist = torch.distributions.OneHotCategorical(logits=current_logits)
    entropy = current_dist.entropy() # Shape: (Seq, Batch)
    
    # We want the actor to pick actions that lead to high Lambda Returns.
    # Since optimizers minimize, we take the negative mean.
    # Actor Loss = Minimize Negative Value - Entropy Bonus
    actor_loss = -torch.mean(lambda_targets[:-1]) - (actor_entropy_scale * torch.mean(entropy[:-1]))

    # actor loss (Maximize Lambda Return)

    actor_loss = -torch.mean(lambda_targets[:-1])
    
    # critic loss (Predict Lambda Return) 
    # The Critic tries to predict the Lambda Return accurately.
    # Stop gradients on the targets so the critic doesn't try to move the target
    critic_loss = F.mse_loss(imagined_values[:-1], lambda_targets[:-1].detach())
    
    actor_optimizer.zero_grad()
    critic_optimizer.zero_grad()
    
    actor_loss.backward(retain_graph=True) 
    critic_loss.backward()
    
    # Clip grads
    torch.nn.utils.clip_grad_norm_(actor_net.parameters(), grad_clipping_value)
    torch.nn.utils.clip_grad_norm_(critic_net.parameters(), grad_clipping_value)
    
    # Update weights (Only now is it safe to modify parameters)
    actor_optimizer.step()
    critic_optimizer.step()
    
    return actor_loss.item(), critic_loss.item()


#main loop

def train_sequence(C, dataset, batch_size, seq_len):

    rssmmodel.train()
    actor_net.train()
    critic_net.train()

    for step in range(C):
        
        o_t, a_t, r_t, _ = dataset.sample(batch_size, seq_len, golden_ratio=0.25)
        B = o_t.size(0)

        # Encode Observations
        flat_obs = o_t.view(-1, *obs_shape)
        flat_embed = rssmmodel.obs_encoder(flat_obs)
        embed_t = flat_embed.view(B, seq_len, -1)
        
        # Shift Actions
        shifted_actions = torch.cat([
            torch.zeros(B, 1, action_dim, device=DEVICE),
            a_t[:, :-1, :]
        ], dim=1)

        #initialise
        h_t = torch.zeros(B, deterministic_dim, device=DEVICE)
        s_t = torch.zeros(B, latent_dim, device=DEVICE) 

        total_loss_accum = 0
        total_recon = 0
        total_kl = 0
        total_reward = 0
        
        # Store states for Dreamer Training 
        posterior_h_list = []
        posterior_s_list = []

        model_optimizer.zero_grad()

        for L in range(seq_len):
            
            (steploss, recon, kl, rew, h_t, s_t) = compute_model_loss(
                o_t=o_t[:, L],          
                o_embed=embed_t[:, L], 
                a_t=shifted_actions[:, L], 
                r_t=r_t[:, L],
                h_prev=h_t,
                s_prev=s_t
            )

            total_loss_accum += steploss
            total_recon      += recon
            total_kl         += kl
            total_reward     += rew
            
            # Save states for the actor to dream from
            posterior_h_list.append(h_t)
            posterior_s_list.append(s_t)

        total_loss_accum.backward()
        torch.nn.utils.clip_grad_norm_(rssmmodel.parameters(), grad_clipping_value)
        model_optimizer.step()

        # Train Actor & Critic
        # We take all the posterior states we just calculated (the "Real" states)
        # and use them as starting points for Dreams.
        
        # Stack: (Seq_Len * Batch, Dim)
        start_h = torch.cat(posterior_h_list, dim=0) 
        start_s = torch.cat(posterior_s_list, dim=0)
        
        act_loss, crit_loss = train_actor_critic(start_h, start_s)

        # Logging (Optional: You can add these to metrics hooks if you want)
        # print(f"Actor: {act_loss:.3f} | Critic: {crit_loss:.3f}")

        log_training_step(
                    total_loss    = (total_loss_accum / seq_len).item(),
                    recon_loss    = (total_recon   / seq_len).item(),
                    kl_loss       = (total_kl      / seq_len).item(),
                    reward_loss   = (total_reward / seq_len).item(),
                    actor_loss    = act_loss,  
                    critic_loss   = crit_loss, 
                    psnr          = 0 
        )