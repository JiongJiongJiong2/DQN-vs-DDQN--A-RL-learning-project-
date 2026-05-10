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