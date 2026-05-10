# Bug & 环境修复记录

## 2026-05-10：Gym → Gymnasium 迁移修复

### 问题描述

项目原始代码基于 `gym` 库编写，但 `gym` 自 2022 年后已停止维护，且与 NumPy 2.0+ 不兼容，导致运行时出现 `_ARRAY_API not found`、`AttributeError` 等错误。需要迁移到 `gymnasium`（Gym 的官方继任者）。

---

### 修改清单

#### 1. main.py — `seed_all` 函数

**位置**：`seed_all` 函数定义

**修改前**：
```python
def seed_all(seed, env):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    env.seed(seed)
    env.action_space.seed(seed)
```

**修改后**：
```python
def seed_all(seed, env=None):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    if env is not None:
        # Gymnasium 方式：通过 reset(seed=seed) 设置环境随机种子
        env.reset(seed=seed)
```

**原因**：Gymnasium 移除了 `env.seed()` 方法，改用 `env.reset(seed=seed)` 设置随机种子。同时增加 `env=None` 默认参数和 GPU 种子支持。

---

#### 2. main.py — 环境名称

**位置**：`if __name__=='__main__'` 块

**修改前**：
```python
env = gym.make('CartPole-v0')
```

**修改后**：
```python
env = gym.make('CartPole-v1')
```

**原因**：Gymnasium 中 `CartPole-v0` 已被移除，统一使用 `CartPole-v1`。

---

#### 3. learn.py — `env.reset()` 返回值（learn 函数）

**位置**：`DQNTrainer.learn()` 方法中

**修改前**：
```python
state = self.env.reset()
```

**修改后**：
```python
state, _ = self.env.reset()
```

**原因**：Gym 的 `env.reset()` 只返回 `observation`；Gymnasium 的 `env.reset()` 返回 `(observation, info)` 元组。

---

#### 4. learn.py — `env.step()` 返回值（learn 函数）

**位置**：`DQNTrainer.learn()` 方法中

**修改前**：
```python
next_state, reward, done, info = self.env.step(action)
```

**修改后**：
```python
next_state, reward, terminated, truncated, info = self.env.step(action)
done = terminated or truncated
```

**原因**：Gym 的 `env.step()` 返回 4 个值 `(obs, reward, done, info)`；Gymnasium 返回 5 个值 `(obs, reward, terminated, truncated, info)`，其中 `terminated` 表示是否达到终止状态，`truncated` 表示是否被截断，`done = terminated or truncated`。

---

#### 5. learn.py — `env.reset()` 返回值（evaluate 函数）

**位置**：`DQNTrainer.evaluate()` 方法中

**修改前**：
```python
state = self.env.reset()
```

**修改后**：
```python
state, _ = self.env.reset()
```

**原因**：同修改 #3。

---

#### 6. learn.py — `env.step()` 返回值（evaluate 函数）

**位置**：`DQNTrainer.evaluate()` 方法中

**修改前**：
```python
next_state, reward, done, info = self.env.step(action)
```

**修改后**：
```python
next_state, reward, terminated, truncated, info = self.env.step(action)
done = terminated or truncated
```

**原因**：同修改 #4。

---

### Gym vs Gymnasium API 对照表

| API | Gym (旧) | Gymnasium (新) |
|-----|----------|----------------|
| 设置种子 | `env.seed(seed)` | `env.reset(seed=seed)` |
| 重置环境 | `obs = env.reset()` | `obs, info = env.reset()` |
| 执行动作 | `obs, r, done, info = env.step(a)` | `obs, r, terminated, truncated, info = env.step(a)` |
| CartPole | `CartPole-v0` | `CartPole-v1` |
| Acrobot | `Acrobot-v1` | `Acrobot-v1`（不变） |

---

## 2026-05-10：Q 值爆炸 — Target Network 更新条件 Bug 修复

### 问题描述

在训练过程中观察到 **Q 值爆炸**（Max Q 持续异常增长，远超正常范围），导致训练完全不稳定、loss 无法收敛。

**根本原因**：`learn.py` 的 `training_step` 函数中，Target Network 的更新条件写法有误：

```python
if t / self.config.learning_freq % self.config.target_update_freq == 0:
```

**Bug 分析**：

这个表达式 `t / self.config.learning_freq % self.config.target_update_freq` 存在严重的逻辑错误：

1. **运算符优先级问题**：`/` 和 `%` 优先级相同，从左到右结合，所以实际计算的是 `(t / learning_freq) % target_update_freq`，而不是预期的 `t / (learning_freq % target_update_freq)`。
2. **整数除法问题**：Python 3 中 `/` 是浮点除法，`t / learning_freq` 的结果为浮点数，再用 `%` 取模，由于浮点精度问题，`== 0` 的判断几乎永远不会成立。
3. **即使使用 `//`（整除）**：`t // learning_freq % target_update_freq` 的语义也不对——它表示"每经过 `target_update_freq` 个 learning 步"，但 `t` 本身已经在 `learn()` 的训练循环中通过 `t % learning_freq == 0` 控制了调用 `training_step` 的频率，所以在 `training_step` 内部不应该再除以 `learning_freq`。

**后果**：Target Network 几乎从不更新（或更新时机完全错乱），导致 Online Network 持续追逐一个过时的 target，Q 值不断自我放大，最终爆炸。

---

### 修改清单

#### 1. learn.py — `training_step` 中 Target Network 更新条件

**位置**：`DQNTrainer.training_step()` 方法末尾

**修改前**：
```python

if t / self.config.learning_freq % self.config.target_update_freq == 0:
    self.update_target()
```

**修改后**：
```python
# 直接对全局步数 t 取模
if t % self.config.target_update_freq == 0:
    self.update_target()
```

**原因**：
- `t` 是全局时间步计数器，在 `learn()` 的主循环中每步 +1。
- `training_step` 已经只在 `t % learning_freq == 0` 时被调用，不需要在内部再除以 `learning_freq`。
- 正确语义是：每隔 `target_update_freq` 个全局步更新一次 Target Network。
- 使用 `%` 取模 + `== 0` 判断是标准且清晰的做法。

---

#### 2. main.py — `target_update_freq` 调整

**位置**：`config` 类

**修改前**：
```python
target_update_freq = 20
```

**修改后**：
```python
target_update_freq = 50
```

**原因**：
- 原先 `target_update_freq` 值过小（如 20），在修复更新条件后会导致 Target Network 更新过于频繁，失去"稳定 target"的作用。
- 设为 50 是一个合理的折中：既不会太频繁（导致 target 不稳定），也不会太稀疏（导致 Q 值过时太久）。
- 对于 CartPole-v1 和 Acrobot-v1 这类简单环境，50 步更新一次效果良好。

---

### 影响总结

| 项目 | 修复前 | 修复后 |
|------|--------|--------|
| Target Network 更新 | 几乎不更新 / 时机错乱 | 每隔 50 步稳定更新 |
| Q 值行为 | 持续爆炸增长 | 正常收敛 |
| 训练稳定性 | Loss 不收敛，Rewards 不上升 | Loss 下降，Rewards 逐步上升 |
