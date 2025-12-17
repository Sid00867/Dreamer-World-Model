import torch
import random
import numpy as np

class ReplayBuffer:
    def __init__(
        self,
        capacity_episodes: int,
        max_episode_len: int,
        obs_shape,
        action_dim,
        device= "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.capacity = capacity_episodes
        self.max_episode_len = max_episode_len
        self.obs_shape = obs_shape
        self.action_dim = action_dim

        self.ptr = 0
        self.size = 0

        # Storage
        self.obs = torch.zeros((capacity_episodes, max_episode_len, *obs_shape), dtype=torch.float32)
        self.actions = torch.zeros((capacity_episodes, max_episode_len, action_dim), dtype=torch.float32)
        self.rewards = torch.zeros((capacity_episodes, max_episode_len, 1), dtype=torch.float32)
        self.dones = torch.zeros((capacity_episodes, max_episode_len, 1), dtype=torch.bool)
        self.lengths = torch.zeros(capacity_episodes, dtype=torch.int32)

        # Episode Construction
        self._cur_ep_len = 0
        self._cur_ep_reward = 0.0 
        self.winning_episodes = set() 

    def add_step(self, obs, action, reward, done):
        ep = self.ptr
        t = self._cur_ep_len

        if t >= self.max_episode_len:
            self._finalize_episode()
            ep = self.ptr
            t = 0

        self.obs[ep, t] = torch.as_tensor(obs)
        self.actions[ep, t] = torch.as_tensor(action)
        self.rewards[ep, t] = float(reward)
        self.dones[ep, t] = bool(done)

        self._cur_ep_len += 1
        self._cur_ep_reward += float(reward)

        if done:
            self._finalize_episode()

    def _finalize_episode(self):

        if self.ptr in self.winning_episodes:
            self.winning_episodes.remove(self.ptr)

        if self._cur_ep_reward > 0.0: 
            self.winning_episodes.add(self.ptr)

        self.lengths[self.ptr] = self._cur_ep_len
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        

        self._cur_ep_len = 0
        self._cur_ep_reward = 0.0

    def sample(self, batch_size: int, seq_len: int, golden_ratio: float = 0.2):
        """
        golden_ratio: % of batch that MUST be winning episodes
        """
        assert self.size > 0
        
        # Calculate split
        num_golden = int(batch_size * golden_ratio)
        num_random = batch_size - num_golden

        if len(self.winning_episodes) == 0:
            num_golden = 0
            num_random = batch_size

        indices = []
        

        if num_golden > 0:
            golden_list = list(self.winning_episodes)
            for _ in range(num_golden):
                indices.append(random.choice(golden_list))
                

        for _ in range(num_random):
            indices.append(random.randint(0, self.size - 1))

        batch_obs, batch_act, batch_rew, batch_done = [], [], [], []
        
        for ep in indices:
            ep_len = self.lengths[ep].item()
            

            if ep_len < seq_len:
                while True:
                    ep = random.randint(0, self.size - 1)
                    if self.lengths[ep].item() >= seq_len: break
                    
            # Random start point in episode
            start = random.randint(0, self.lengths[ep].item() - seq_len)
            
            batch_obs.append(self.obs[ep, start:start+seq_len])
            batch_act.append(self.actions[ep, start:start+seq_len])
            batch_rew.append(self.rewards[ep, start:start+seq_len])
            batch_done.append(self.dones[ep, start:start+seq_len])

        return (
            torch.stack(batch_obs).to(self.device),
            torch.stack(batch_act).to(self.device),
            torch.stack(batch_rew).to(self.device),
            torch.stack(batch_done).to(self.device)
        )

    def __len__(self):
        return self.size

    def is_full(self):
        return self.size == self.capacity

    def clear(self):
        self.ptr = 0
        self.size = 0
        self._cur_ep_len = 0
        self.lengths.zero_()
