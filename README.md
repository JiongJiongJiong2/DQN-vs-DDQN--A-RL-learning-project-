<div align="center">



```text
--  /$$$$$$$   /$$$$$$  /$$   /$$                                    
-- | $$__  $$ /$$__  $$| $$$ | $$                                    
-- | $$  \ $$| $$  \ $$| $$$$| $$                                    
-- | $$  | $$| $$  | $$| $$ $$ $$                                    
-- | $$  | $$| $$  | $$| $$  $$$$                                    
-- | $$  | $$| $$/$$ $$| $$\  $$$                                    
-- | $$$$$$$/|  $$$$$$/| $$ \  $$                                    
-- |_______/  \____ $$$|__/  \__/                                    
--  /$$            \__/         /$$                                                /$$ 
-- |__/                        | $$                                               | $$ 
--  /$$ /$$$$$$/$$$$   /$$$$$$ | $$  /$$$$$$  /$$$$$$/$$$$   /$$$$$$   /$$$$$$$  /$$$$$$
-- | $$| $$_  $$_  $$ /$$__  $$| $$ /$$__  $$| $$_  $$_  $$ /$$__  $$ | $$__  $$|_  $$_/         
-- | $$| $$ \ $$ \ $$| $$  \ $$| $$| $$$$$$$$| $$ \ $$ \ $$| $$$$$$$$ | $$  \ $$  | $$  
-- | $$| $$ | $$ | $$| $$  | $$| $$| $$_____/| $$ | $$ | $$| $$_____/ | $$  | $$  | $$ /$$    
-- | $$| $$ | $$ | $$| $$$$$$$/| $$|  $$$$$$$| $$ | $$ | $$|  $$$$$$$ | $$  | $$  |  $$$$/        
-- |__/|__/ |__/ |__/| $$____/ |__/ \_______/|__/ |__/ |__/ \_______/ |__/  |__/   \___/  
--                   | $$                                            
--                   | $$                                            
--                   |__/                                            
                                                            
```
**DQN & Double DQN Implementation + differences between DQN & DDQN**
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
</div>

# DQN & Double DQN Implementation

A PyTorch implementation of DQN and Double DQN deep reinforcement learning algorithms, trained on `CartPole-v1` and `Acrobot-v1` environments.

## Project Structure

| File | Description |
|------|-------------|
| `main.py` | Training entry point with hyperparameter config |
| `model.py` | QModel class (MLP network) |
| `learn.py` | DQN / Double DQN loss computation and training logic |
| `schedule.py` | Linear decay schedule and ε-greedy exploration |
| `utils/` | Utilities (Replay Buffer, logging, plotting, etc.) |
| `DQN_implementation_notes.md` | 📝 Implementation notes (algorithm details, code walkthrough, bug fixes) |
| `bugfix_log.md` | Bug fix log |
| `report.md` | Experiment report |

## Quick Start

### 1. Install Dependencies

```bash
conda env create -f introRL-torch.yml
conda activate introRL-torch
```

### 2. Run Training

Edit the bottom of `main.py` to select environment and algorithm:

```python
# Select environment
env = gym.make('CartPole-v1')   # CartPole
# env = gym.make('Acrobot-v1')  # Acrobot

# Select algorithm
double = False  # DQN
double = True   # Double DQN
```

Then run:

```bash
python main.py
```

Training results (model weights, training curves, CSV logs) will be saved in the `results/` directory.

## Notes

For detailed algorithm explanations, line-by-line code analysis, and debugging experience, see **`DQN_implementation_notes.md`**.

## References

- Mnih et al. (2015). *Human-level control through deep reinforcement learning.* Nature.
- van Hasselt et al. (2016). *Deep reinforcement learning with double Q-learning.* AAAI.