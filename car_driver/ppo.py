from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from car_driver.config import NetworkConfig


ACTION_NAMES = ["gas", "brake", "left", "right"]


def get_activation(name: str) -> nn.Module:
    mapping = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "elu": nn.ELU,
        "leaky_relu": nn.LeakyReLU,
        "selu": nn.SELU,
        "identity": nn.Identity,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported activation: {name}")
    return mapping[name]()


def build_mlp(
    input_dim: int,
    hidden_layers: Iterable[tuple[int, str]],
    output_dim: int,
    output_activation: str,
) -> nn.Sequential:
    modules: list[nn.Module] = []
    last_dim = input_dim
    for units, activation in hidden_layers:
        modules.append(nn.Linear(last_dim, units))
        modules.append(get_activation(activation))
        last_dim = units
    modules.append(nn.Linear(last_dim, output_dim))
    modules.append(get_activation(output_activation))
    return nn.Sequential(*modules)


class ActorCritic(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        network_config: NetworkConfig,
        init_std: float = 0.45,
    ) -> None:
        super().__init__()
        hidden_layers = [
            (layer.units, layer.activation)
            for layer in network_config.hidden_layers
        ]
        self.actor = build_mlp(
            observation_dim,
            hidden_layers,
            action_dim,
            network_config.output_activation,
        )
        self.critic = build_mlp(
            observation_dim,
            hidden_layers,
            1,
            "identity",
        )
        init_log_std = np.log(max(init_std, 1e-4))
        self.log_std = nn.Parameter(torch.full((action_dim,), init_log_std))
        self.network_config = network_config
        self.observation_dim = observation_dim
        self.action_dim = action_dim

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.actor(obs), self.critic(obs).squeeze(-1)

    def distribution(self, obs: torch.Tensor) -> tuple[Normal, torch.Tensor]:
        mean = self.actor(obs)
        std = torch.exp(self.log_std).expand_as(mean)
        return Normal(mean, std), mean

    def act(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dist, mean = self.distribution(obs)
        action = mean if deterministic else dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        value = self.critic(obs).squeeze(-1)
        return action, log_prob, value, mean

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist, _ = self.distribution(obs)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic(obs).squeeze(-1)
        return log_prob, entropy, value

    def act_numpy(
        self,
        obs: np.ndarray,
        device: torch.device,
        deterministic: bool = True,
    ) -> np.ndarray:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action, _, _, mean = self.act(obs_tensor, deterministic=deterministic)
        chosen = mean if deterministic else action
        return chosen.squeeze(0).cpu().numpy()

    def export_visualization_data(
        self,
        input_names: list[str],
        output_names: list[str],
    ) -> dict[str, Any]:
        layer_sizes = [self.observation_dim]
        weight_matrices: list[np.ndarray] = []
        for module in self.actor:
            if isinstance(module, nn.Linear):
                weight_matrices.append(
                    module.weight.detach().cpu().numpy().copy()
                )
                layer_sizes.append(module.out_features)

        activations = [
            layer.activation for layer in self.network_config.hidden_layers
        ] + [self.network_config.output_activation]

        return {
            "layer_sizes": layer_sizes,
            "weights": weight_matrices,
            "activations": activations,
            "input_names": input_names,
            "output_names": output_names,
            "log_std": self.log_std.detach().cpu().numpy().copy(),
        }
