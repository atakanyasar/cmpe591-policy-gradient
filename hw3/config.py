import torch

class Config:
    num_envs = 4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_episodes = 100000
    save_interval = 128
    action_dim = 2
    state_dim = 6
    learning_rate = 1e-3
    gamma = 0.99
    batch_size = 4