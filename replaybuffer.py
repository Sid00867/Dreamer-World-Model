import torch
import random

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

        # Storage tensors
        self.obs = torch.zeros(
            (capacity_episodes, max_episode_len, *obs_shape),
            dtype=torch.float32
        )

        self.actions = torch.zeros(
            (capacity_episodes, max_episode_len, action_dim),
            dtype=torch.float32
        )

        self.rewards = torch.zeros(
            (capacity_episodes, max_episode_len, 1),
            dtype=torch.float32
        )

        self.dones = torch.zeros(
            (capacity_episodes, max_episode_len, 1),
            dtype=torch.bool
        )

        # Store actual episode lengths
        self.lengths = torch.zeros(
            capacity_episodes,
            dtype=torch.int32
        )

        # Active episode builder
        self._cur_ep_len = 0


    def add_step(self, obs, action, reward, done):
        """
        Add single environment step to current episode.
        When 'done' is True, episode is finalized automatically.
        """
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

        if done:
            self._finalize_episode()

    # --------------------------------------------------------

    def _finalize_episode(self):
        """
        Saves current episode length, advances buffer pointer
        """
        self.lengths[self.ptr] = self._cur_ep_len
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self._cur_ep_len = 0

    # --------------------------------------------------------
    # SAMPLING
    # --------------------------------------------------------

    def sample(
        self,
        batch_size: int,
        seq_len: int
    ):
        """
        Sample batch of contiguous sequences.

        Returns:
            obs      (B,S,C,H,W)
            actions  (B,S,A)
            rewards  (B,S,1)
            dones    (B,S,1)
        """

        assert self.size > 0
        assert seq_len <= self.max_episode_len

        batch_obs = []
        batch_act = []
        batch_rew = []
        batch_done = []

        for _ in range(batch_size):
            valid = False

            # Resample until a valid subsequence is found
            while not valid:

                ep = random.randint(0, self.size - 1)

                ep_len = self.lengths[ep].item()

                if ep_len < seq_len:
                    continue

                start = random.randint(0, ep_len - seq_len)

                # Don't sample sequences crossing termination
                done_flags = self.dones[ep, start:start + seq_len]
                if done_flags[:-1].any():
                    continue

                valid = True

            batch_obs.append(
                self.obs[ep, start:start + seq_len]
            )
            batch_act.append(
                self.actions[ep, start:start + seq_len]
            )
            batch_rew.append(
                self.rewards[ep, start:start + seq_len]
            )
            batch_done.append(
                self.dones[ep, start:start + seq_len]
            )

        obs = torch.stack(batch_obs).to(self.device)
        acts = torch.stack(batch_act).to(self.device)
        rews = torch.stack(batch_rew).to(self.device)
        dones = torch.stack(batch_done).to(self.device)

        return obs, acts, rews, dones

    # --------------------------------------------------------
    # UTILITIES
    # --------------------------------------------------------

    def __len__(self):
        return self.size

    def is_full(self):
        return self.size == self.capacity

    def clear(self):
        self.ptr = 0
        self.size = 0
        self._cur_ep_len = 0
        self.lengths.zero_()
