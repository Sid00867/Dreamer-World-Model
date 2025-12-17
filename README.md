# DreamerV1 — A World Model–Based Reinforcement Learning Implementation

This repository contains my implementation of **DreamerV1**, a world-model–based reinforcement learning algorithm that learns latent dynamics using a Recurrent State Space Model (RSSM) and trains policies via imagination rollouts.

The implementation was built to better understand how latent imagination works in practice and to experimentally evaluate its behavior in environments outside the original DeepMind Control Suite, particularly **Gridworld-style tasks**.

A detailed write-up of the motivation, architecture, experiments, and failure analysis can be found here:

👉 **Medium article:**  
[Planning by Dreaming: Why DreamerV1 Breaks in Simple Worlds](https://medium.com/@siddhartharduino/planning-by-dreaming-why-dreamerv1-breaks-in-simple-worlds-64ef14acd82a)

---

## Overview

DreamerV1 learns:
- a **world model** (encoder, decoder, RSSM, reward model)
- an **actor-critic policy**
- entirely from latent imagination rollouts rather than real environment interaction

Key ideas implemented:
- Recurrent State Space Model (deterministic + stochastic latent state)
- Latent imagination rollouts
- Actor-critic training inside the learned latent space
- Offline planning without environment interaction during policy updates

---

## Architecture

The implementation follows the original DreamerV1 structure:

- **Encoder / Decoder** – compress observations into a latent representation and reconstruct them
- **RSSM**
  - Deterministic state (`h`) for memory
  - Stochastic state (`s`) for uncertainty
- **Transition Model** – predicts future latent states
- **Reward Model** – predicts rewards from latent states
- **Actor & Critic** – trained purely on imagined trajectories

World model components are trained on real trajectories sampled from a replay buffer, while the policy and value function are trained using imagination rollouts.

---

## Experiments

The primary experimental focus of this repository is evaluating DreamerV1 on **Gridworld-like environments** with:
- partial observability
- discrete state transitions
- long-horizon dependencies
- randomized layouts

Despite strong reconstruction quality, the agent fails to learn meaningful behavior due to breakdowns in latent imagination over long horizons. Detailed results, reconstructions, and analysis are discussed in the linked article.

---

## References

- Hafner et al., *Dream to Control: Learning Behaviors by Latent Imagination*, 2019  
