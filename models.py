import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import numpy as np


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class PolicyNN(nn.Module):
    def __init__(self, obs_dim, logit_dim) -> None:
        super(PolicyNN, self).__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, logit_dim), std=0.01),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logit = self.actor(x)
        value = self.critic(x)
        return logit, value

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic(x)

    def init_weights(self) -> None:
        pass

    @staticmethod
    def sample_action(logits: torch.Tensor, deterministic=False) -> torch.Tensor:
        """
        Input shape: (N, logit_dim), N - batch_size
        """
        prob = torch.softmax(logits, dim=-1)
        result = -1
        if deterministic:
            result = torch.argmax(prob, dim=1)
        else:
            dist = torch.distributions.Categorical(probs=prob)
            result = dist.sample()

        return result.detach().cpu().numpy()
