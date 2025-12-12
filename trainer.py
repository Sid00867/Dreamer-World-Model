from environment_variables import *

import torch
from tqdm import tqdm 
import torch.nn.functional as F
from fitter import train_sequence
from fitter import rssmmodel
from explore_sample import run_data_collection
from replaybuffer import ReplayBuffer
from metrics import METRICS, plot_metrics
from explore_sample import preprocess_obs
from planner import reset_planner

buffer = ReplayBuffer(
    capacity_episodes=replay_buffer_capacity,
    max_episode_len=max_episode_len,
    obs_shape=obs_shape,
    action_dim=action_dim,
    device=DEVICE
)

def seed_replay_buffer(num_episodes= seed_replay_buffer_episodes):
    for ep in range(num_episodes):
        reset_planner()
        obs_raw, _ = env.reset()
        obs = preprocess_obs(obs_raw)
        done = False
        episode_len = 0

        while not done and episode_len < max_episode_len:

            rnd = torch.randint(0, action_dim, (1,))
            a_onehot = F.one_hot(rnd, action_dim).float().to(DEVICE)

            action = rnd.item()

            obs_next_raw, reward, terminated, truncated, info, _ = env.step(action)
            done = terminated or truncated

            obs_next = preprocess_obs(obs_next_raw)

            buffer.add_step(
                obs.cpu(),
                a_onehot.cpu(),
                reward,
                done
            )

            obs = obs_next
            episode_len += 1
    print(f"Seeded replay buffer with {num_episodes} episodes.")        


def convergence_trainer():

    outer_iter = 0

    with tqdm(total=max_steps, desc="Convergence Loop") as pbar:

        while not METRICS.has_converged():
            outer_iter += 1

            train_sequence(
                C=C,
                dataset=buffer,
                batch_size=batch_size,
                seq_len=seq_len,
            )

            run_data_collection(buffer, pbar)

            stats = METRICS.get_means()

            pbar.set_postfix({
                "Loss": f"{stats['loss_total']:.4f}",
                "PSNR": f"{stats['psnr']:.2f}",
                "Return": f"{stats['return']:.1f}",
                "Success": f"{100*stats['success_rate']:.1f}%",
                "EnvSteps": stats["env_steps"],
            })

            if outer_iter % weight_save_freq_for_outer_iters == 0:
                torch.save(rssmmodel.state_dict(), weights_path)

    print("\n==== TRAINING CONVERGED ====")
    print("Final metrics:")

    for k, v in METRICS.get_means().items():
        print(f"{k:16s}: {v:.4f}")

    plot_metrics(METRICS)


if __name__ == "__main__":
    seed_replay_buffer()
    convergence_trainer()
    torch.save(rssmmodel.state_dict(), weights_path)