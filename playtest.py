import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

from environment_variables import *
from planner import plan, reset_planner
from explore_sample import preprocess_obs
from fitter import rssmmodel, actor_net 


NUM_EPISODES = 5
MANUAL_STEPS_DURATION = 15  

def make_interactive_env():
    return RLReadyEnv(
        env_kind="simple",
        size=10,
        obs_mode="rgb",
        obs_scope="partial",
        render_mode="rgb_array", 
        seed=42
    )

class DreamController:
    def __init__(self):
        self.mode = "AUTO" 
        self.manual_steps_left = 0
        self.next_action = None
        self.waiting_for_input = False
        
    def on_key(self, event):
        if event.key == 'm':
            self.mode = "MANUAL"
            self.manual_steps_left = MANUAL_STEPS_DURATION
            return

        if self.mode == "MANUAL" and self.waiting_for_input:
            if event.key == 'left': self.next_action = 0
            elif event.key == 'right': self.next_action = 1
            elif event.key == 'up': self.next_action = 2
            
            if self.next_action is not None:
                self.waiting_for_input = False 

def play():
    playenv = make_interactive_env()
    
    # Load weights
    if torch.cuda.is_available():
        checkpoint = torch.load("rssm_final.pth")
    else:
        checkpoint = torch.load("rssm_final.pth", map_location="cpu")
        
    rssmmodel.load_state_dict(checkpoint['rssm'])
    actor_net.load_state_dict(checkpoint['actor'])
    rssmmodel.eval()
    actor_net.eval()

    # Setup Plot
    controller = DreamController()
    plt.ion() 
    fig, (ax_real, ax_dream) = plt.subplots(1, 2, figsize=(10, 5))
    
    ax_real.set_title("Reality (Ground Truth)")
    ax_dream.set_title("Dream (Blind Prediction)")
    ax_real.axis('off'); ax_dream.axis('off')

    img_real = ax_real.imshow(np.zeros((28, 28, 3)))
    img_dream = ax_dream.imshow(np.zeros((28, 28, 3)))
    status_text = fig.suptitle("Mode: AUTO", fontsize=14, color='green')

    fig.canvas.mpl_connect('key_press_event', controller.on_key)
    print("Controls: 'm' = Manual Mode (Blind Dream) | Arrows = Move")

    for ep in range(NUM_EPISODES):
        reset_planner()
        obs_raw, _ = playenv.reset()
        obs = preprocess_obs(obs_raw)

        # Initialize State
        dummy_action = torch.zeros(1, action_dim, device=DEVICE)
        obs_embed = rssmmodel.obs_encoder(obs.unsqueeze(0))
        
        # Start with a grounded state
        _, _, _, _, h, s = rssmmodel.forward_train(
            h_prev=torch.zeros(1, deterministic_dim, device=DEVICE),
            s_prev=torch.zeros(1, latent_dim, device=DEVICE),
            a_prev=dummy_action,
            o_embed=obs_embed
        )

        done = False
        step = 0

        while not done:
            fig.canvas.flush_events()

            action_idx = 0
            if controller.mode == "MANUAL":
                status_text.set_text(f"MANUAL (BLIND DREAM) | Steps: {controller.manual_steps_left}")
                status_text.set_color('red')
                
                controller.waiting_for_input = True
                while controller.waiting_for_input:
                    plt.pause(0.1) 
                    if not plt.fignum_exists(fig.number): return
                
                action_idx = controller.next_action
                controller.next_action = None 
                
                controller.manual_steps_left -= 1
                if controller.manual_steps_left <= 0:
                    controller.mode = "AUTO"
            else:
                status_text.set_text("AUTO (Posterior Corrected)")
                status_text.set_color('green')
                with torch.no_grad():
                    a_onehot_plan = plan(h, s)
                    action_idx = a_onehot_plan.argmax(-1).item()
                plt.pause(0.05)


            a_curr = torch.zeros(1, action_dim, device=DEVICE)
            a_curr[0, action_idx] = 1.0


            obs_next_raw, reward, terminated, truncated, info, _ = playenv.step(action_idx)
            done = terminated or truncated
            

            with torch.no_grad():
                if controller.mode == "MANUAL":

                    gru_input = torch.cat([s, a_curr], dim=-1)
                    h = rssmmodel.gru(gru_input, h)
                    

                    prior_hidden = rssmmodel.prior_fc(h)
                    mu_prior = rssmmodel.prior_mu(prior_hidden)
                    log_sigma_prior = rssmmodel.prior_log_sigma(prior_hidden)
                    sigma_prior = torch.exp(log_sigma_prior)

                    noise = torch.randn_like(sigma_prior)
                    s = mu_prior + sigma_prior * noise

                    dec_input = torch.cat([s, h], dim=-1)

                    x_dec = rssmmodel.fc(dec_input)
                    x_dec = x_dec.view(-1, 128, 3, 3) 
                    o_recon = rssmmodel.decoder(x_dec)

                    
                else:

                    obs_next = preprocess_obs(obs_next_raw)
                    obs_embed = rssmmodel.obs_encoder(obs_next.unsqueeze(0))

                    _, _, o_recon, _, h, s = rssmmodel.forward_train(
                        h_prev=h,
                        s_prev=s,
                        a_prev=a_curr, 
                        o_embed=obs_embed
                    )

            img_real.set_data(obs_next_raw['image'])

            dream_np = o_recon[0].permute(1, 2, 0).cpu().numpy()
            img_dream.set_data(np.clip(dream_np, 0, 1))
            
            fig.canvas.draw()
            step += 1

        print(f"Episode finished in {step} steps")

if __name__ == "__main__":
    play()
