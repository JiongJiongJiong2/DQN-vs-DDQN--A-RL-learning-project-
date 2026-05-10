# DQN & Double DQN: Training Report on CartPole-v1 and Acrobot-v1
2230026047

## 1. Introduction

This report presents the experimental results of Deep Q-Network (DQN) and Double Deep Q-Network (DDQN) on two classic control environments: **CartPole-v1** and **Acrobot-v1**. We evaluate both algorithms across four metrics: Training Rewards, Evaluation Rewards, Max Q-value, and Loss. The goal is to verify the correctness of our implementations and to compare the performance of DQN versus DDQN.

> **Note on Environment Version**: The assignment specifies CartPole-v0; however, since our codebase uses the Gymnasium library (the official successor to OpenAI Gym), which has deprecated CartPole-v0, we use CartPole-v1 instead. The two versions differ only in the maximum episode length (200 vs. 500 steps), and CartPole-v1 is the standard in modern Gymnasium.

## 2. Methodology

### 2.1 DQN

DQN (Mnih et al., 2015) combines Q-learning with deep neural networks and introduces two key mechanisms:

- **Experience Replay**: Transitions $(s, a, r, s')$ are stored in a replay buffer and sampled randomly during training to break temporal correlations.
- **Target Network**: A separate target network $Q(s, a; \theta^-)$ is used to compute the TD target, which is periodically updated by copying the online network weights.

**DQN Loss**:

$$\mathcal{L}(\theta) = \mathbb{E}\left[\left(r + \gamma (1 - d) \max_{a'} Q_{\text{target}}(s', a'; \theta^-) - Q_{\text{online}}(s, a; \theta)\right)^2\right]$$

where $d$ is the done flag, $\gamma = 0.99$ is the discount factor.

### 2.2 Double DQN

Double DQN (van Hasselt et al., 2016) addresses the overestimation bias of standard DQN by decoupling action selection from value estimation:

- **Online network** selects the best action: $a^* = \arg\max_{a'} Q_{\text{online}}(s', a'; \theta)$
- **Target network** evaluates the action: $Q_{\text{target}}(s', a^*; \theta^-)$

**Double DQN Target**:

$$y = r + \gamma (1 - d) \cdot Q_{\text{target}}\left(s', \arg\max_{a'} Q_{\text{online}}(s', a'; \theta); \theta^-\right)$$

### 2.3 Target Network Update

We use **hard update** (periodic weight copying):

$$\theta^- \leftarrow \theta \quad \text{every } \texttt{target\_update\_freq} \text{ steps}$$

## 3. Experimental Setup

### 3.1 Network Architecture

Both DQN and DDQN use the same Q-network architecture — a 3-layer MLP:

```
Linear(state_dim, 256) → ReLU → Linear(256, 256) → ReLU → Linear(256, action_dim)
```

- CartPole-v1: `state_dim = 4`, `action_dim = 2`
- Acrobot-v1: `state_dim = 6`, `action_dim = 3`

### 3.2 Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `batch_size` | 32 | Mini-batch size for training |
| `gamma` | 0.99 | Discount factor |
| `replay_buffer_size` | 100,000 | Capacity of experience replay buffer |
| `learning_start` | 100 | Steps before training begins |
| `learning_freq` | 1 | Training frequency (every step) |
| `target_update_freq` | 50 | Steps between target network updates |
| `lr_begin` / `lr_end` | 0.001 | Learning rate (constant) |
| `eps_begin` | 1.0 | Initial exploration rate (ε-greedy) |
| `eps_end` | 0.1 | Final exploration rate |
| `eps_nsteps` | 1,000 | Steps for ε linear decay |
| `num_timesteps` | 30,000 | Total training steps |
| `clip_val` | 5.0 | Gradient clipping threshold |
| `num_episodes_eval` | 10 | Episodes for evaluation |
| `eval_freq` | 100 | Steps between evaluations |
| Optimizer | RMSprop (α=0.95, ε=0.01) | — |

### 3.3 Random Seed

All experiments use `seed_all(seed=0)` to ensure reproducibility. The seed function sets seeds for `random`, `numpy`, `torch`, and the Gymnasium environment.

### 3.4 Environment Details

| Environment | State Dim | Action Dim | Reward Range | Goal |
|-------------|-----------|------------|--------------|------|
| CartPole-v1 | 4 | 2 | [0, +∞) | Balance the pole as long as possible (max 500 steps) |
| Acrobot-v1 | 6 | 3 | (−∞, 0] | Swing up the link in minimum steps (max 500 steps) |

> **Note**: CartPole rewards are positive (higher is better), while Acrobot rewards are negative (closer to 0 is better, with −1 per timestep).

## 4. Results: DQN

### 4.1 DQN on CartPole-v1

#### Training Rewards
![DQN CartPole Training Rewards](results/260510-2-C-DQN-test/finaltrainrewards.jpg)

#### Evaluation Rewards
![DQN CartPole Eval Rewards](results/260510-2-C-DQN-test/evalrewards.jpg)

#### Max Q
![DQN CartPole Max Q](results/260510-2-C-DQN-test/maxQ.jpg)

#### Loss
![DQN CartPole Loss](results/260510-2-C-DQN-test/loss.jpg)

**Summary**: DQN on CartPole-v1 shows a clear learning progression. Training rewards increase from ~37 to ~380, and evaluation rewards reach the maximum of 500 by the end of training. The Max Q values exhibit significant variance (ranging from −4 to 138), indicating some overestimation. Loss values are generally small but with occasional spikes.

---

### 4.2 DQN on Acrobot-v1

#### Training Rewards
![DQN Acrobot Training Rewards](results/260510-2-A-DQN-test/finaltrainrewards.jpg)

#### Evaluation Rewards
![DQN Acrobot Eval Rewards](results/260510-2-A-DQN-test/evalrewards.jpg)

#### Max Q
![DQN Acrobot Max Q](results/260510-2-A-DQN-test/maxQ.jpg)

#### Loss
![DQN Acrobot Loss](results/260510-2-A-DQN-test/loss.jpg)

**Summary**: DQN on Acrobot-v1 shows gradual improvement. Training rewards improve from −500 to approximately −100, and evaluation rewards reach around −80 to −90 at best, though with high variance (frequent −500 evaluations). The Max Q values remain negative (around −3 to −15), consistent with the negative reward structure. The agent learns to solve the task faster than random but does not consistently achieve the optimal policy within 30,000 steps.

---

## 5. Results: Double DQN (DDQN)

### 5.1 DDQN on CartPole-v1

#### Training Rewards
![DDQN CartPole Training Rewards](results/260510-2-C-DDQN-test/finaltrainrewards.jpg)

#### Evaluation Rewards
![DDQN CartPole Eval Rewards](results/260510-2-C-DDQN-test/evalrewards.jpg)

#### Max Q
![DDQN CartPole Max Q](results/260510-2-C-DDQN-test/maxQ.jpg)

#### Loss
![DDQN CartPole Loss](results/260510-2-C-DDQN-test/loss.jpg)

**Summary**: DDQN on CartPole-v1 also demonstrates successful learning. Training rewards increase from ~34 to ~429, and evaluation rewards consistently reach 500. The Max Q values show similar variance to DQN (ranging up to ~128), and loss values remain generally low with occasional spikes.

---

### 5.2 DDQN on Acrobot-v1

#### Training Rewards
![DDQN Acrobot Training Rewards](results/260510-2-A-DDQN-test/finaltrainrewards.jpg)

#### Evaluation Rewards
![DDQN Acrobot Eval Rewards](results/260510-2-A-DDQN-test/evalrewards.jpg)

#### Max Q
![DDQN Acrobot Max Q](results/260510-2-A-DDQN-test/maxQ.jpg)

#### Loss
![DDQN Acrobot Loss](results/260510-2-A-DDQN-test/loss.jpg)

**Summary**: DDQN on Acrobot-v1 shows similar performance to DQN. Training rewards improve from −500 to approximately −117, and evaluation rewards reach around −74 to −128. The Max Q values remain negative (around −13 to −30), which is more negative than DQN's Max Q values (around −3 to −15). This may reflect DDQN's more conservative value estimation. Loss values are moderate and show occasional spikes.

---

## 6. Comparison: DQN vs. DDQN

### 6.1 Quantitative Comparison

| Metric | DQN CartPole | DDQN CartPole | DQN Acrobot | DDQN Acrobot |
|--------|-------------|---------------|-------------|--------------|
| Final Training Rewards | ~380 | ~429 | ~−123 | ~−117 |
| Final Eval Rewards | 500 | 476–500 | ~−136 | ~−128 |
| Max Q (final) | ~108 | ~124 | ~−3.8 | ~−13.5 |
| Loss (final) | ~4.3 | ~2.0 | ~1.4 | ~1.6 |

### 6.2 Analysis

#### CartPole-v1

Both DQN and DDQN successfully solve CartPole-v1, achieving the maximum evaluation reward of 500. The key observations are:

1. **Learning Speed**: DDQN appears to learn slightly faster, reaching high rewards earlier in training. By timestep ~16,000, DDQN already achieves training rewards of ~316, while DQN reaches similar levels around timestep ~18,000.

2. **Max Q Overestimation**: Both algorithms show significant variance in Max Q values, with occasional spikes above 100. This suggests that both DQN and DDQN suffer from some degree of Q-value overestimation on this environment, though it does not prevent successful learning. Interestingly, DDQN's Max Q values are not consistently lower than DQN's, which is somewhat unexpected given DDQN's theoretical advantage in reducing overestimation.

3. **Stability**: Both algorithms show occasional loss spikes, but training remains stable overall. The evaluation rewards for both converge to 500 by the end of training.

#### Acrobot-v1

On Acrobot-v1, both algorithms show more modest performance:

1. **Partial Learning**: Both DQN and DDQN improve from the initial reward of −500 (random policy) to approximately −80 to −120, indicating that the agent learns to solve the task faster than random but does not consistently find the optimal policy within 30,000 steps.

2. **High Variance**: Evaluation rewards show high variance, frequently dropping to −500 (failure to solve), interspersed with successful episodes around −80 to −90. This suggests the learned policy is not robust.

3. **Max Q Difference**: DQN's Max Q values are closer to 0 (around −3 to −5) compared to DDQN's (around −13 to −30). This is consistent with DDQN's more conservative estimation — DDQN's decoupled action selection tends to produce lower (less overestimated) Q-values.

4. **Similar Convergence**: Both algorithms converge to similar final performance, with DDQN showing slightly better training rewards (−117 vs. −123) but comparable evaluation performance.

### 6.3 Does DDQN Make a Difference?

On these two simple environments with the current hyperparameters, the difference between DQN and DDQN is **minimal**:

- On **CartPole-v1**, both algorithms solve the environment successfully. The overestimation bias that DDQN is designed to address does not significantly impact performance on this simple task.
- On **Acrobot-v1**, both algorithms show partial learning with high variance. DDQN's more conservative Q-value estimation is visible in the Max Q plots but does not translate to significantly better policy performance.

This is consistent with the literature: DDQN's advantages are more pronounced in environments with larger action spaces and more complex dynamics, where overestimation bias has a greater impact on policy quality (van Hasselt et al., 2016).

---

## 7. Implementation Details

### 7.1 Key Components

| Component | File | Description |
|-----------|------|-------------|
| Q-Network | `model.py` | 3-layer MLP (256 hidden units, ReLU) |
| DQN Loss | `learn.py` | `compute_DQN_loss()` — standard MSE loss with target network |
| DDQN Loss | `learn.py` | `compute_DoubleDQN_loss()` — online selects action, target evaluates |
| Target Update | `learn.py` | `update_target()` — hard update every 50 steps |
| Exploration | `schedule.py` | `ExplorationSchedule` — ε-greedy with linear decay |
| Learning Rate | `schedule.py` | `LinearSchedule` — constant at 0.001 |

### 7.2 Bug Fix: Target Network Update Condition

During development, we identified and fixed a critical bug in the target network update condition. The original code used floating-point division in the modulo check, which caused the target network to almost never update, leading to Q-value explosion:

```python
# Bug: floating-point division causes condition to almost never trigger
if t / self.config.learning_freq % self.config.target_update_freq == 0:

# Fix: use integer modulo on global timestep
if t % self.config.target_update_freq == 0:
```

This fix was essential for stable training. See `bugfix_log.md` for details.

---

## 8. Conclusion

We successfully implemented and evaluated DQN and Double DQN on CartPole-v1 and Acrobot-v1. Key findings:

1. **Both algorithms solve CartPole-v1**, achieving the maximum reward of 500 within 30,000 timesteps.
2. **Both algorithms show partial learning on Acrobot-v1**, improving from −500 to approximately −80 to −120, but with high variance and inconsistent performance.
3. **DDQN does not show a significant advantage over DQN** on these simple environments, which is expected given the small action spaces and relatively simple dynamics.
4. **Max Q values** show that DDQN produces more conservative estimates on Acrobot (more negative Q-values), consistent with its theoretical property of reducing overestimation bias.
5. The **target network update bug fix** was critical for training stability.

Future work could include: tuning hyperparameters for Acrobot (longer training, different exploration schedule), testing with multiple random seeds for statistical significance, and evaluating on more complex environments where DDQN's advantages are more pronounced.

---

## References

1. Mnih, V., et al. (2015). "Human-level control through deep reinforcement learning." *Nature*, 518(7540), 529–533.
2. van Hasselt, H., Guez, A., & Silver, D. (2016). "Deep reinforcement learning with double Q-learning." *Proceedings of the 30th AAAI Conference on Artificial Intelligence*, 2094–2100.
