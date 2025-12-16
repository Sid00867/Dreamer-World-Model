import torch
import torch.nn.functional as F
import torch.optim as optim
from environment_variables import *
from metrics_hooks import log_training_step
from rssm import rssm
from actor_critic import ActionDecoder, ValueDecoder

# Initialize Models
rssmmodel = rssm().to(DEVICE)
actor_net = ActionDecoder().to(DEVICE)
critic_net = ValueDecoder().to(DEVICE)

model_optimizer = optim.Adam(rssmmodel.parameters(), lr=learnrate, eps=1e-5)
actor_optimizer = optim.Adam(actor_net.parameters(), lr=actor_lr, eps=1e-5)
critic_optimizer = optim.Adam(critic_net.parameters(), lr=value_lr, eps=1e-5)

def calculate_lambda_returns(rewards, values, gamma=gamma, lambda_=lambda_):
    returns = torch.zeros_like(rewards)
    # Bootstrap from last value
    next_return = values[-1] 
    for t in reversed(range(len(rewards) - 1)):
        r_t = rewards[t]
        v_next = values[t+1]
        returns[t] = r_t + gamma * ( (1 - lambda_) * v_next + lambda_ * next_return )
        next_return = returns[t]
    return returns

def compute_model_loss(o_t, o_embed, a_t, r_t, h_prev, s_prev):
    logits_post, logits_prior, o_recon, reward_pred, h_t, s_t = rssmmodel.forward_train(
        h_prev, s_prev, a_t, o_embed
    )

    # 1. Reconstruction Loss (Safety Check Added)
    if o_recon.shape != o_t.shape:
        raise ValueError(f"Shape Mismatch! Recon: {o_recon.shape}, Target: {o_t.shape}. Check rssm.py Decoder.")

    recon_loss = F.mse_loss(o_recon, o_t, reduction='mean')
    
    # 2. Reward Loss
    reward_loss = F.mse_loss(reward_pred, r_t.squeeze(-1), reduction='mean')

    # 3. KL Loss with Balancing
    # DreamerV2: KL(Post || Prior)
    post_dist = torch.distributions.OneHotCategorical(logits=logits_post.view(-1, stoch_size, class_size))
    prior_dist = torch.distributions.OneHotCategorical(logits=logits_prior.view(-1, stoch_size, class_size))

    # Minimize KL(sg(Post) || Prior) -> Train Prior to match Post
    loss_prior = torch.distributions.kl_divergence(
        torch.distributions.OneHotCategorical(logits=logits_post.detach().view(-1, stoch_size, class_size)), 
        prior_dist
    ).sum(dim=1).mean()

    # Minimize KL(Post || sg(Prior)) -> Train Post to be regularized towards Prior
    loss_post = torch.distributions.kl_divergence(
        post_dist, 
        torch.distributions.OneHotCategorical(logits=logits_prior.detach().view(-1, stoch_size, class_size))
    ).sum(dim=1).mean()
    
    kl_loss_balanced = (kl_balance * loss_prior) + ((1 - kl_balance) * loss_post)
    
    # Total Loss
    total_loss = recon_loss + reward_loss + (kl_scale * kl_loss_balanced)

    # Metric logging
    with torch.no_grad():
        raw_kl = torch.distributions.kl_divergence(post_dist, prior_dist).sum(dim=1).mean()

    return total_loss, recon_loss, raw_kl, reward_loss, h_t, s_t

def train_actor_critic(start_h, start_s):
    # Detach start states so we don't backprop into RSSM here
    start_h = start_h.detach()
    start_s = start_s.detach()
    
    # Imagine trajectories
    h_seq, s_seq, action_seq = rssmmodel.imagine_rollout(actor_net, start_h, start_s, horizon=imagination_horizon)
    
    seq_len_rollout, batch, _ = h_seq.shape
    flat_h = h_seq.view(-1, deterministic_dim)
    flat_s = s_seq.view(-1, latent_dim)
    
    target_input = torch.cat([flat_s, flat_h], dim=-1)
    
    imagined_rewards = rssmmodel.reward_model(target_input).view(seq_len_rollout, batch)
    imagined_values  = critic_net(target_input).view(seq_len_rollout, batch)
    
    lambda_targets = calculate_lambda_returns(imagined_rewards, imagined_values)

    # Actor Loss
    current_logits = actor_net(target_input).view(seq_len_rollout, batch, -1)
    current_dist = torch.distributions.OneHotCategorical(logits=current_logits)
    entropy = current_dist.entropy() 
    
    # Maximize Value + Entropy
    actor_loss = -torch.mean(lambda_targets[:-1]) - (actor_entropy_scale * torch.mean(entropy[:-1]))
    
    # Critic Loss (Value Prediction)
    critic_loss = F.mse_loss(imagined_values[:-1], lambda_targets[:-1].detach())
    
    actor_optimizer.zero_grad()
    critic_optimizer.zero_grad()
    
    actor_loss.backward(retain_graph=True) 
    critic_loss.backward()
    
    torch.nn.utils.clip_grad_norm_(actor_net.parameters(), grad_clipping_value)
    torch.nn.utils.clip_grad_norm_(critic_net.parameters(), grad_clipping_value)
    
    actor_optimizer.step()
    critic_optimizer.step()
    
    return actor_loss.item(), critic_loss.item()

def train_sequence(C, dataset, batch_size, seq_len):
    rssmmodel.train()
    actor_net.train()
    critic_net.train()

    for step in range(C):
        o_t, a_t, r_t, _ = dataset.sample(batch_size, seq_len, golden_ratio=0.25)
        B = o_t.size(0)

        # Encode all observations at once for speed
        flat_obs = o_t.view(-1, *obs_shape)
        flat_embed = rssmmodel.obs_encoder(flat_obs)
        embed_t = flat_embed.view(B, seq_len, -1)
        
        # Shift actions (a_t is the action TAKEN at t, we need it as input for t+1)
        shifted_actions = torch.cat([
            torch.zeros(B, 1, action_dim, device=DEVICE),
            a_t[:, :-1, :]
        ], dim=1)

        # Initialize state
        h_t = torch.zeros(B, deterministic_dim, device=DEVICE)
        s_t = torch.zeros(B, latent_dim, device=DEVICE) 

        total_loss_accum = 0
        total_recon = 0
        total_kl = 0
        total_reward = 0
        
        posterior_h_list = []
        posterior_s_list = []

        model_optimizer.zero_grad()

        # BPTT Loop
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
            
            # Store for Actor-Critic training
            posterior_h_list.append(h_t)
            posterior_s_list.append(s_t)

        # Backprop through time
        total_loss_accum.backward()
        torch.nn.utils.clip_grad_norm_(rssmmodel.parameters(), grad_clipping_value)
        model_optimizer.step()

        # Train Behavior (Actor/Critic) on the posterior states we just calculated
        # Flatten time and batch dimensions
        start_h = torch.cat(posterior_h_list, dim=0).detach() 
        start_s = torch.cat(posterior_s_list, dim=0).detach()
        
        act_loss, crit_loss = train_actor_critic(start_h, start_s)

        log_training_step(
            total_loss    = (total_loss_accum / seq_len).item(),
            recon_loss    = (total_recon   / seq_len).item(),
            kl_loss       = (total_kl      / seq_len).item(),
            reward_loss   = (total_reward / seq_len).item(),
            actor_loss    = act_loss,  
            critic_loss   = crit_loss, 
            psnr          = 0 
        )