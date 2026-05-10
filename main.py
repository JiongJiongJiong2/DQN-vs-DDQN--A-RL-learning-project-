import gymnasium as gym
from model import QModel
from schedule import LinearSchedule, ExplorationSchedule
from learn import DQNTrainer
from utils.general import csv_plot
import torch
import numpy as np
import random
from utils.test_env import EnvTest
import os
import datetime

class config():
    output_path = 'results/'+datetime.datetime.now().strftime('%y%m%d-%H%M%S')+'/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    plot_dir = output_path + 'rewards.pdf'
    train_plot_dir = output_path + 'during_training_rewards.pdf'
    model_dir = output_path + 'model.weights'
    csv_dir = output_path + 'log.csv'

    # hyperparams
    batch_size = 32
    gamma = 0.99
    replay_buffer_size = 100000
    learning_start = 100
    learning_freq = 1
    target_update_freq = 50
    lr_begin = 0.001#0.01
    lr_end = 0.001#0.01
    lr_nsteps = 10000
    eps_begin = 1.0
    eps_end = 0.1
    eps_nsteps = 1000
    num_timesteps = 30000
    tau = 0.03
    beta = 0.9

    clip_val = 5.
    log_freq = 100
    num_episodes_eval = 10
    high = 1.
    saving_freq = 1000
    eval_freq = 100

    double = None

def main(env, double):
    config.double = double
    exploration_schedule = ExplorationSchedule(env, config.eps_begin,
            config.eps_end, config.eps_nsteps)
    lr_schedule = LinearSchedule(config.lr_begin, config.lr_end,
            config.lr_nsteps)
    q_function = QModel
    trainer = DQNTrainer(env,
             exploration_schedule,
             lr_schedule,
             config,
             q_function)
    trainer.learn()


def seed_all(seed, env=None):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    if env is not None:
        # Gymnasium 方式：通过 reset(seed=seed) 设置环境随机种子
        env.reset(seed=seed)


if __name__=='__main__':
    #env = gym.make('CartPole-v1')
    env = gym.make('Acrobot-v1')
    # env = EnvTest((100,))
    seed_all(0, env)

    # set double to True for Double DQN, False for DQN
    double = True
    main(env, double)
    csv_plot(config.csv_dir, config.output_path)

