# AI423 Deep Reinforcement Learning - Project 1 Rules
**DQN & Double DQN Implementation (2026 Spring)**

## 项目目标
- 实现一个能正常运行的简化版 DQN 和 Double DQN。
- 必须在 `CartPole-v1` 和 `Acrobot-v1` 上训练并成功生成训练曲线（rewards, loss 等 plots）。
- 优先保证**代码能跑通、符合题目要求**，不需要追求最优性能。
- 每一次修改添加代码等操作要给我解释做了什么，怎么做的，为什么。
- 每一次指令，尤其涉及到环境配置问题，必须询问是否执行，不能擅自修改。

## 核心公式（严格按照以下方式实现）

### 1. DQN Loss
```python
# Target 计算
target = r + γ * (1 - done) * max(Q_target(s', ·))
# Loss
loss = mean( (Q_online(s, a) - target) ** 2 )

```

### 2. Double DQN (DDQN) Target 
```python
# 使用 online network 选择动作，使用 target network 计算价值
next_action = argmax(Q_online(s', ·))
target = r + γ * (1 - done) * Q_target(s', next_action)

```

### 3. Target Network 更新（Hard Update）
```Python
# 每隔 target_update_freq 步执行一次
target.load_state_dict(online.state_dict())
```

## 2. 文件实现要求
### 1. model.py → QModel 类（15 points）

使用简单 MLP（多层感知机）。
推荐网络结构（对两个环境都适用）：
```Python
Pythonself.net = nn.Sequential(
    nn.Linear(state_dim, 256),
    nn.ReLU(),
    nn.Linear(256, 256),
    nn.ReLU(),
    nn.Linear(256, action_dim)
)
```
forward() 返回 Q 值（shape: [batch_size, action_dim]）。


### 2. learn.py
#### (1) compute_DQN_loss（20 points）
- 实现标准的 DQN loss。
- 注意处理 done（episode 结束时 future reward 为 0）。
- 使用 torch.no_grad() 计算 target。

#### (2) update_target（20 points）
- 实现 hard update（直接复制权重）。

#### (3) compute_DoubleDQN_loss（15 points）
- 实现 Double DQN target 公式（online 选动作，target 估价值）。
- 可通过 double=True 参数切换测试。

### 3. schedule.py
LinearSchedule 和 ExplorationSchedule 按题目要求实现（通常已提供框架，按需补全）。

## 实现原则（重要！）

1. **最优先**：代码能跑通，成功生成 8 个 plots（Training rewards、Evaluation rewards、max Q、Loss 等）。
2. **严格遵守** 题目要求的函数签名和接口，不要随意修改已有框架。
3. **先实现 DQN**（设置 `double=False`），跑通并出图后再实现 Double DQN。
4. **网络保持简单**：使用 2-3 层 MLP（256 隐藏单元即可），不要加复杂结构。
5. **使用已有的 Replay Buffer、config 参数和工具函数**。
6. **随机种子**：保持 `seed_all()` 的设置，确保结果可复现。

## 调试建议
- 先使用 `CartPole-v1` 快速验证代码是否正确（该环境更容易学到好策略）。
- 观察 loss 是否整体下降，training/evaluation rewards 是否上升。
- 如果初期不收敛，可适当调整 `learning_start` 或网络大小（但不要大改 config）。
- 可以使用 `print` 语句输出 loss、Q 值、rewards 等信息帮助调试。

## 禁止事项
- 不要大幅修改 config 中的超参数（除非实在无法运行）。
- 不要使用 CNN（这两个环境是低维向量状态）。
- 不要加入 Prioritized Replay、Dueling DQN、Noisy Nets 等高级技巧。
- 不要修改 main.py 中训练和绘图的主体逻辑。

---

**使用提示**：  
当你需要我帮你写具体函数时，请直接告诉我文件名和函数名（例如：“写 model.py 中的 QModel” 或 “写 compute_DQN_loss”），我会给你**可直接复制粘贴**的代码 + 必要说明。