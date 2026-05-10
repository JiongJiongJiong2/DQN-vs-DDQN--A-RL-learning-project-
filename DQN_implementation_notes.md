# DQN & Double DQN 实现笔记

## 项目概述

本项目实现了 DQN（Deep Q-Network）和 Double DQN 两种深度强化学习算法，用于在 `CartPole-v1` 和 `Acrobot-v1` 环境中训练智能体。

---

## 一、DQN 核心知识

### 1.1 Q-Learning 基础

Q-Learning 是一种无模型（model-free）的离策略（off-policy）强化学习算法，通过学习状态-动作价值函数（Q 函数）来找到最优策略。

**Bellman 方程**：

```
Q*(s, a) = E[ r + γ * max_a' Q*(s', a') ]
```

其中：
- `s`：当前状态
- `a`：当前动作
- `r`：即时奖励
- `γ`：折扣因子（0~1）
- `s'`：下一状态
- `a'`：下一状态的动作

### 1.2 DQN 的核心创新

DQN（Mnih et al., 2015）将深度神经网络与 Q-Learning 结合，解决了传统 Q-Learning 无法处理高维状态空间的问题。其两大核心创新为：

#### (1) Experience Replay（经验回放）

将智能体与环境交互得到的转移 `(s, a, r, s')` 存储在回放缓冲区中，训练时从中随机采样小批量数据。

**优点**：
- 打破数据之间的时间相关性
- 提高数据利用效率
- 使训练更稳定（类似监督学习）

#### (2) Target Network（目标网络）

使用两个 Q 网络：
- **Online Network** `Q(s, a; θ)`：每步更新，用于选择动作和计算当前 Q 值
- **Target Network** `Q(s, a; θ^-)`：定期从 Online Network 复制权重，用于计算 target

**为什么需要 Target Network？**

如果不使用 Target Network，target 值 `r + γ * max_a' Q(s', a'; θ)` 中的 `θ` 和当前 Q 值的 `θ` 是同一个网络，导致"追逐移动目标"的问题——网络在更新 Q 值的同时也在更新 target，使得训练不稳定。

**Hard Update（本项目使用）**：
```python
# 每隔 target_update_freq 步执行一次
θ^- ← θ
```

### 1.3 DQN Loss 函数

```
L(θ) = E[ ( r + γ * (1 - done) * max_a' Q(s', a'; θ^-) - Q(s, a; θ) )^2 ]
```

其中：
- `D`：经验回放缓冲区
- `done`：episode 是否结束的标志
- `θ`：Online Network 参数
- `θ^-`：Target Network 参数

**关键**：target 的计算不需要梯度，使用 `torch.no_grad()` 阻止梯度回传。

---

## 二、Double DQN 核心知识

### 2.1 DQN 的过估计问题

标准 DQN 在计算 target 时使用 `max` 操作：
```
target = r + γ * max_a' Q_target(s', a')
```

这会导致**过估计（Overestimation）**问题：由于 `max` 操作总是选择最大的 Q 值，即使 Q 值估计有噪声，也会倾向于选择被高估的值，导致 target 偏高。

### 2.2 Double DQN 的解决方案

Double DQN（van Hasselt et al., 2016）将动作选择和价值估计解耦：

- **Online Network 选择动作**：`a* = argmax_a' Q_online(s', a'; θ)`
- **Target Network 估价值**：`Q_target(s', a*; θ^-)`

```
target = r + γ * (1 - done) * Q_target(s', argmax_a' Q_online(s', a'; θ); θ^-)
```

**直觉理解**：
- Online Network 负责"决定哪个动作最好"（可能有一定偏差）
- Target Network 负责"评估这个动作到底值多少"（更客观）
- 即使 Online Network 选错了动作，Target Network 会给出一个更保守的估计，避免过估计

### 2.3 DQN vs Double DQN 对比

| 方面 | DQN | Double DQN |
|------|-----|------------|
| 动作选择 | Target Network | **Online Network** |
| 价值估计 | Target Network | Target Network |
| Target 公式 | `r + γ * max Q_target(s', ·)` | `r + γ * Q_target(s', argmax Q_online(s', ·))` |
| 过估计 | 严重 | 缓解 |
| 额外计算量 | 无 | 几乎无（只需一次额外前向传播） |

---

## 三、代码实现详解

### 3.1 model.py — QModel 类

**功能**：定义 Q 函数的神经网络结构

```python
import torch.nn as nn

class QModel(nn.Module):
    def __init__(self, in_features=128, num_actions=18):
        super(QModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
        )

    def forward(self, x):
        return self.model(x)
```

**设计说明**：
- **输入**：状态向量（`in_features` 维），例如 CartPole 的 4 维状态
- **输出**：每个动作的 Q 值（`num_actions` 维），例如 CartPole 的 2 个动作
- **网络结构**：2 个隐藏层，每层 256 个神经元，ReLU 激活
- **为什么用 MLP**：CartPole 和 Acrobot 的状态是低维向量，不需要 CNN

**网络结构图**：
```
State (dim=4) → Linear(4, 256) → ReLU → Linear(256, 256) → ReLU → Linear(256, 2) → Q values
```

---

### 3.2 schedule.py — LinearSchedule.update

**功能**：实现参数的线性衰减（用于学习率和探索率）

```python
def update(self, t):
    if t >= self.nsteps:
        self.curr_val = self.val_end
    else:
        self.curr_val = self.val_begin + (self.val_end - self.val_begin) * (t / self.nsteps)
```

**数学公式**：
- 当 `t < nsteps` 时：`curr_val = val_begin + (val_end - val_begin) * t / nsteps`
- 当 `t >= nsteps` 时：`curr_val = val_end`（保持不变）

**应用场景**：
- **探索率**：从 1.0 线性衰减到 0.1（前期多探索，后期多利用）
- **学习率**：从 0.001 保持到 0.001（本项目学习率不变）

---

### 3.3 schedule.py — ExplorationSchedule.get_action

**功能**：实现 ε-greedy 探索策略

```python
def get_action(self, q_vals):
    if np.random.random() < self.curr_val:
        return self.env.action_space.sample()
    else:
        return np.argmax(q_vals)
```

**ε-greedy 策略**：
- 以概率 ε（`curr_val`）随机选择动作 → **探索**
- 以概率 1-ε 选择 Q 值最大的动作 → **利用**

**为什么需要探索？**
- 如果始终选择当前最优动作，可能陷入局部最优
- 探索可以发现更好的策略
- 训练初期 ε 大（多探索），后期 ε 小（多利用）

---

### 3.4 learn.py — update_target

**功能**：Hard Update，将 Online Network 的权重复制到 Target Network

```python
def update_target(self):
    self.target_network.load_state_dict(self.q_network.state_dict())
```

**工作原理**：
- `self.q_network.state_dict()`：获取 Online Network 的所有参数
- `self.target_network.load_state_dict(...)`：将这些参数加载到 Target Network
- 这是完全复制，不是插值（区别于 Soft Update）

**Hard Update vs Soft Update**：
- **Hard Update**（本项目）：每隔固定步数，直接复制全部权重 `θ^- ← θ`
- **Soft Update**：每步缓慢更新 `θ^- ← τθ + (1-τ)θ^-`，其中 τ 很小（如 0.005）

**调用时机**：在 `training_step` 中，每隔 `target_update_freq` 步调用一次

---

### 3.5 learn.py — compute_DQN_loss（核心！）

**功能**：计算标准 DQN 的 MSE Loss

```python
def compute_DQN_loss(self, state_batch, action_batch,
                     reward_batch, next_state_batch, done_batch):
    # 1. 获取当前状态采取动作的 Q 值
    q_values = self.q_network(state_batch).gather(1, action_batch).squeeze()

    # 2. 计算 target（使用 Target Network，不需要梯度）
    with torch.no_grad():
        target_q_values = self.target_network(next_state_batch).max(dim=1)[0]
        target = reward_batch.squeeze() + self.config.gamma * (1 - done_batch.float()) * target_q_values

    # 3. 计算 MSE Loss
    loss = nn.MSELoss()(q_values, target)

    return loss
```

**代码详解**：

**步骤 1：提取已采取动作的 Q 值**
```python
q_values = self.q_network(state_batch).gather(1, action_batch).squeeze()
```
- `self.q_network(state_batch)` 输出 shape `[batch_size, num_actions]`，包含每个状态下所有动作的 Q 值
- `.gather(1, action_batch)` 根据 `action_batch` 中的索引提取对应动作的 Q 值
- `.squeeze()` 将 shape `[batch_size, 1]` 变为 `[batch_size]`

**步骤 2：计算 target**
```python
with torch.no_grad():
    target_q_values = self.target_network(next_state_batch).max(dim=1)[0]
    target = reward_batch.squeeze() + self.config.gamma * (1 - done_batch.float()) * target_q_values
```
- `torch.no_grad()`：阻止梯度回传到 Target Network（关键！）
- `.max(dim=1)[0]`：获取每个下一状态的最大 Q 值
- `(1 - done_batch.float())`：done mask，episode 结束时 future reward 为 0
- `self.config.gamma`：折扣因子（本项目为 0.99）

**步骤 3：计算 MSE Loss**
```python
loss = nn.MSELoss()(q_values, target)
```
- 计算 Q 值与 target 之间的均方误差

**公式总结**：
```
target = r + γ * (1 - done) * max_a' Q_target(s', a')
loss = mean( (Q_online(s, a) - target)^2 )
```

---

### 3.6 learn.py — compute_DoubleDQN_loss（核心！）

**功能**：计算 Double DQN 的 MSE Loss

```python
def compute_DoubleDQN_loss(self, state_batch, action_batch,
                           reward_batch, next_state_batch, done_batch):
    # 1. 获取当前状态采取动作的 Q 值
    q_values = self.q_network(state_batch).gather(1, action_batch).squeeze()

    # 2. 计算 target（Online Network 选动作，Target Network 估价值）
    with torch.no_grad():
        # Online Network 选择最优动作
        next_actions = self.q_network(next_state_batch).argmax(dim=1, keepdim=True)
        # Target Network 评估该动作的价值
        target_q_values = self.target_network(next_state_batch).gather(1, next_actions).squeeze()
        # 计算 target
        target = reward_batch.squeeze() + self.config.gamma * (1 - done_batch.float()) * target_q_values

    # 3. 计算 MSE Loss
    loss = nn.MSELoss()(q_values, target)

    return loss
```

**代码详解**：

**与 DQN 的关键区别**：
```python
# DQN：直接用 Target Network 的 max
target_q_values = self.target_network(next_state_batch).max(dim=1)[0]

# Double DQN：Online Network 选动作，Target Network 估价值
next_actions = self.q_network(next_state_batch).argmax(dim=1, keepdim=True)
target_q_values = self.target_network(next_state_batch).gather(1, next_actions).squeeze()
```

**步骤分解**：
1. `self.q_network(next_state_batch)` — Online Network 对下一状态的 Q 值估计
2. `.argmax(dim=1, keepdim=True)` — 选择 Q 值最大的动作索引，shape `[batch_size, 1]`
3. `self.target_network(next_state_batch)` — Target Network 对下一状态的 Q 值估计
4. `.gather(1, next_actions)` — 根据 Online Network 选的动作，提取 Target Network 的 Q 值

**公式总结**：
```
next_action = argmax_a' Q_online(s', a')
target = r + γ * (1 - done) * Q_target(s', next_action)
loss = mean( (Q_online(s, a) - target)^2 )
```

---

## 四、训练流程总结

### 4.1 训练循环（learn 函数）

```
1. 初始化 replay_buffer、logger、网络
2. while t < num_timesteps:
   3. 获取当前状态的 Q 值，用 ε-greedy 选择动作
   4. 执行动作，存储 (s, a, s', r, done) 到 replay_buffer
   5. 如果 t > learning_start:
      6. 从 replay_buffer 采样 batch
      7. 计算 loss（DQN 或 Double DQN）
      8. 反向传播更新 Online Network
      9. 每隔 target_update_freq 步更新 Target Network
   10. 记录日志、评估性能
```

### 4.2 关键超参数（main.py config）

| 参数 | 值 | 说明 |
|------|---|------|
| `batch_size` | 32 | 每次训练的样本数 |
| `gamma` | 0.99 | 折扣因子 |
| `learning_start` | 100 | 开始学习前的步数 |
| `target_update_freq` | 20 | Target Network 更新频率 |
| `eps_begin` | 1.0 | 初始探索率 |
| `eps_end` | 0.1 | 最终探索率 |
| `eps_nsteps` | 1000 | 探索率衰减步数 |
| `num_timesteps` | 30000 | 总训练步数 |

---

## 五、常见问题与解决方案

### 5.1 Loss 不下降 / Rewards 不上升

**可能原因**：
- 网络结构过于简单或过于复杂
- 学习率过大或过小
- 探索率衰减太快或太慢

**解决方案**：
- 先用 CartPole 验证代码正确性
- 检查 `learning_start` 是否足够大（让 replay buffer 先积累数据）
- 观察初期 Loss 是否异常大（可能需要调整网络初始化）

### 5.2 Q 值过大（过估计）

**现象**：Max Q 持续上升，远超实际奖励

**解决方案**：
- 使用 Double DQN 替代 DQN
- 降低 `gamma`（减少对未来奖励的关注）

### 5.3 训练不稳定

**可能原因**：
- Target Network 更新频率太低
- 梯度爆炸

**解决方案**：
- 减小 `target_update_freq`（更频繁更新 Target Network）
- 检查 `clip_val`（梯度裁剪阈值，本项目为 5.0）

---

## 六、参考文献

1. Mnih, V., et al. (2015). "Human-level control through deep reinforcement learning." Nature.
2. van Hasselt, H., Guez, A., & Silver, D. (2016). "Deep reinforcement learning with double Q-learning." AAAI.
