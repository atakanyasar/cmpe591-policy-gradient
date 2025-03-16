import torch
from torch import optim
import torch.nn.functional as F

from config import Config

class Agent():
    def __init__(self, model: torch.nn.Module):
        self.model = model.to(Config.device)  # Ensure model is on the right device
        self.optimizer = optim.Adam(self.model.parameters(), lr=Config.learning_rate)
        self.gamma = Config.gamma

    def decide_action(self, state):
        state = torch.as_tensor(state, dtype=torch.float32, device=Config.device).unsqueeze(0)
        action_mean, act_std = self.model(state).chunk(2, dim=-1)
        action_std = F.softplus(act_std) + 5e-2
        action = torch.normal(action_mean, action_std).clamp(-1, 1)
        return action.squeeze(0).detach().cpu().numpy()

    def action_greedy(self, state):
        state = torch.as_tensor(state, dtype=torch.float32, device=Config.device).unsqueeze(0)
        action_mean, _ = self.model(state).chunk(2, dim=-1)
        return action_mean.squeeze(0).detach().cpu().numpy()

    def compute_discounted_returns(self, rewards):
        G = torch.zeros(len(rewards), dtype=torch.float32, device=Config.device)
        cumulative = 0.0
        for t in reversed(range(len(rewards))):
            cumulative = rewards[t] + self.gamma * cumulative
            G[t] = cumulative
        return G

    def update_model(self, trajectory_buffer):
        all_states, all_actions, all_rewards = [], [], []

        for trajectory in trajectory_buffer:
            states, actions, rewards, _, _ = zip(*trajectory)
            all_states.extend(states)
            all_actions.extend(actions)
            all_rewards.extend(rewards)

        trajectory_buffer.clear()

        # Convert lists to PyTorch tensors
        all_states = torch.stack([torch.as_tensor(s, dtype=torch.float32, device=Config.device) for s in all_states])
        all_actions = torch.stack([torch.as_tensor(a, dtype=torch.float32, device=Config.device) for a in all_actions])
        all_rewards = torch.as_tensor(all_rewards, dtype=torch.float32, device=Config.device)

        # Compute returns
        all_returns = self.compute_discounted_returns(all_rewards)

        # Normalize returns
        all_returns = (all_returns - all_returns.mean()) / (all_returns.std() + 1e-5)

        # Compute policy loss
        action_mean, act_std = self.model(all_states).chunk(2, dim=-1)
        action_std = F.softplus(act_std) + 5e-2
        action_dist = torch.distributions.Normal(action_mean, action_std)
        log_prob = action_dist.log_prob(all_actions).sum(dim=-1)

        loss = - (log_prob * all_returns).mean()

        # Perform gradient update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
