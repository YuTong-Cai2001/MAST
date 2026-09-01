"""Multi-agent semantic alignment with explicit source/target adapters."""

import torch
import torch.nn as nn
import torch.nn.functional as F


VALID_DOMAINS = {"source", "target"}


class SemanticAgent(nn.Module):
    """Map either domain into one shared semantic space."""

    def __init__(
        self,
        target_dim,
        source_dim,
        hidden_dims=(512, 256),
        dropout_rate=0.4,
        agent_id=0,
    ):
        super().__init__()
        self.agent_id = agent_id
        self.hidden_dims = list(hidden_dims)
        self.target_dim = target_dim
        self.source_dim = source_dim
        common_dim = hidden_dims[0]

        # Both adapters are learned. Neither domain is an identity reference.
        self.source_adapter = nn.Sequential(
            nn.Linear(source_dim, common_dim),
            nn.LayerNorm(common_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        self.target_adapter = nn.Sequential(
            nn.Linear(target_dim, common_dim),
            nn.LayerNorm(common_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        self.semantic_focus = nn.Parameter(torch.randn(common_dim) * 0.01)
        self.transform = nn.Sequential(
            nn.Linear(common_dim, hidden_dims[1]),
            nn.LayerNorm(hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        self.policy_net = nn.Sequential(
            nn.Linear(hidden_dims[1], 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def _adapt(self, x, domain):
        if domain not in VALID_DOMAINS:
            raise ValueError(f"domain must be one of {sorted(VALID_DOMAINS)}, got {domain!r}")
        expected_dim = self.source_dim if domain == "source" else self.target_dim
        if x.ndim != 2 or x.size(1) != expected_dim:
            raise ValueError(
                f"{domain} semantic input must have shape [batch, {expected_dim}], "
                f"got {tuple(x.shape)}"
            )
        adapter = self.source_adapter if domain == "source" else self.target_adapter
        return adapter(x)

    def forward(self, x, *, domain):
        common = self._adapt(x, domain)
        focused = common * torch.sigmoid(self.semantic_focus)
        transformed = self.transform(focused)
        return transformed, self.policy_net(transformed)

class MultiAgentSemanticTransform(nn.Module):
    def __init__(
        self,
        target_dim,
        source_dim,
        num_agents=3,
        hidden_dims=(512, 256),
        dropout_rate=0.4,
    ):
        super().__init__()
        if num_agents != 3:
            raise ValueError(f"the paper protocol requires exactly 3 agents, got {num_agents}")

        self.num_agents = num_agents
        self.target_dim = target_dim
        self.source_dim = source_dim
        self.hidden_dims = list(hidden_dims)
        self.dropout_rate = dropout_rate
        self.agents = nn.ModuleList(
            SemanticAgent(
                target_dim,
                source_dim,
                self.hidden_dims,
                dropout_rate,
                agent_id,
            )
            for agent_id in range(num_agents)
        )
        self.collaboration_gate = nn.Linear(
            self.hidden_dims[1] * num_agents, num_agents
        )
        self.dim_transform = nn.Linear(self.hidden_dims[1], source_dim)
        self.projector = nn.Sequential(
            nn.Linear(self.hidden_dims[1], 128),
            nn.LayerNorm(128),
        )
        self.temperature = nn.Parameter(torch.ones(1))
        self._last_gates = None
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _combine(self, x, domain):
        agent_outputs = []
        policy_values = []
        for agent in self.agents:
            transformed, policy_value = agent(x, domain=domain)
            agent_outputs.append(transformed)
            policy_values.append(policy_value)

        combined_features = torch.cat(agent_outputs, dim=1)
        temperature = self.temperature.abs().clamp_min(1e-4)
        weights = F.softmax(self.collaboration_gate(combined_features) / temperature, dim=1)
        # [batch_size, num_agents, agent_output_dim]
        stacked_agent_outputs = torch.stack(agent_outputs, dim=1)
        weighted_sum = (
            stacked_agent_outputs * weights.unsqueeze(-1)
        ).sum(dim=1)
        self._last_gates = weights
        return weighted_sum, weights, policy_values, stacked_agent_outputs

    def get_projection(self, x, *, domain):
        weighted_sum, _, _, _ = self._combine(x, domain)
        return self.projector(weighted_sum)

    def get_last_attention_weights(self):
        attention_weights = [
            torch.sigmoid(agent.semantic_focus) for agent in self.agents
        ]
        return torch.stack(attention_weights, dim=0).mean(dim=0)

    def get_last_gates(self):
        if self._last_gates is None:
            raise RuntimeError("no routing gates are available before the first forward pass")
        return self._last_gates

    def forward(self, x, *, domain):
        (
            weighted_sum,
            weights,
            policy_values,
            stacked_agent_outputs,
        ) = self._combine(x, domain)
        return (
            self.dim_transform(weighted_sum),
            weights,
            policy_values,
            stacked_agent_outputs,
        )


DynamicSemanticTransformNet = MultiAgentSemanticTransform
