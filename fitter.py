import torch
import torch.nn.functional as F
from rssm import rssm
from environment_variables import *
from metrics_hooks import log_training_step

rssmmodel = rssm().to(DEVICE)
optimizer = torch.optim.Adam(rssmmodel.parameters(), lr=learnrate)

def compute_psnr(x, x_hat):
    mse = F.mse_loss(x, x_hat, reduction="mean")
    return 10.0 * torch.log10(1.0 / mse)

def compute_loss(o_t, a_t, r_t, h_prev):
    (mu_post, log_sigma_post), (mu_prior, log_sigma_prior), o_recon, reward_pred, h_t = rssmmodel.forward_train(
        h_prev, a_t, o_t
    )

    recon_loss  = F.mse_loss(o_recon, o_t, reduction='mean')
    reward_loss = F.mse_loss(reward_pred, r_t, reduction='mean')

    log_sigma_post  = torch.clamp(log_sigma_post,  -log_sigma_clamp, log_sigma_clamp)
    log_sigma_prior = torch.clamp(log_sigma_prior, -log_sigma_clamp, log_sigma_clamp)

    kl_loss = torch.mean(
        torch.distributions.kl_divergence(
            torch.distributions.Normal(mu_post,  torch.exp(log_sigma_post)),
            torch.distributions.Normal(mu_prior, torch.exp(log_sigma_prior))
        )
    )

    total_loss = recon_loss + reward_loss + beta * kl_loss

    psnr = compute_psnr(o_t, o_recon)

    return total_loss, recon_loss, kl_loss, reward_loss, psnr, h_t



def train_sequence(C, dataset, batch_size, seq_len):

    rssmmodel.train()

    for step in range(C):

        o_t, a_t, r_t, _ = dataset.sample(batch_size, seq_len)

        B = o_t.size(0)
        h_t = torch.zeros(B, deterministic_dim, device=DEVICE)

        total_loss = 0
        total_recon = 0
        total_kl = 0
        total_reward = 0
        total_psnr = 0

        for L in range(seq_len):

            (
                steploss,
                recon_loss,
                kl_loss,
                reward_loss,
                psnr,
                h_t
            ) = compute_loss(
                o_t[:, L],
                a_t[:, L],
                r_t[:, L],
                h_t
            )

            total_loss    += steploss
            total_recon   += recon_loss
            total_kl      += kl_loss
            total_reward += reward_loss
            total_psnr   += psnr

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(rssmmodel.parameters(), grad_clipping_value)
        optimizer.step()

        log_training_step(
            total_loss    = total_loss.item(),
            recon_loss    = (total_recon   / seq_len).item(),
            kl_loss       = (total_kl      / seq_len).item(),
            reward_loss   = (total_reward / seq_len).item(),
            psnr          = (total_psnr   / seq_len).item(),
        )

        # if step % 10 == 0:
        #     print(f"Step {step}, loss: {total_loss.item():.3f}")
