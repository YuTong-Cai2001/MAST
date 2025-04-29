import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SemanticAgent(nn.Module):
    """
    单个语义转换代理，负责将源域语义空间转换到目标域语义空间
    """
    def __init__(self, source_dim, target_dim, hidden_dims=[512, 256], dropout_rate=0.3):
        super(SemanticAgent, self).__init__()
        
        # 创建源域到目标域的转换网络
        layers = []
        input_dim = source_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            input_dim = hidden_dim
        
        # 最后一层输出目标域维度
        layers.append(nn.Linear(input_dim, target_dim))
        
        self.source_to_target = nn.Sequential(*layers)
        
        # 创建策略网络，用于决定该代理的权重
        policy_layers = []
        policy_input_dim = source_dim
        
        for hidden_dim in hidden_dims:
            policy_layers.append(nn.Linear(policy_input_dim, hidden_dim // 2))
            policy_layers.append(nn.ReLU())
            policy_layers.append(nn.Dropout(dropout_rate))
            policy_input_dim = hidden_dim // 2
        
        # 策略网络输出一个标量值
        policy_layers.append(nn.Linear(policy_input_dim, 1))
        policy_layers.append(nn.Sigmoid())  # 输出0-1之间的值
        
        self.policy_net = nn.Sequential(*policy_layers)
        
    def forward(self, x):
        # 转换语义特征
        transformed = self.source_to_target(x)
        
        # 计算策略值（该代理的权重）
        policy_value = self.policy_net(x)
        
        return transformed, policy_value

    def compute_semantic_score(self, x):
        """
        计算语义得分，用于语义一致性损失
        """
        # 简单地使用策略网络的输出作为语义得分
        return self.policy_net(x)

class MultiAgentSemanticTransform(nn.Module):
    """
    多代理语义转换系统，使用多个代理共同完成语义空间转换
    """
    def __init__(self, source_dim, target_dim, num_agents=3, hidden_dims=[512, 256], dropout_rate=0.3):
        super(MultiAgentSemanticTransform, self).__init__()
        
        self.source_dim = source_dim
        self.target_dim = target_dim
        self.num_agents = num_agents
        
        # 创建多个语义转换代理
        self.agents = nn.ModuleList([
            SemanticAgent(source_dim, target_dim, hidden_dims, dropout_rate)
            for _ in range(num_agents)
        ])
        
        # 添加投影头，用于对比学习
        projection_dim = 128  # 投影维度
        self.projection_head = nn.Sequential(
            nn.Linear(target_dim, 256),
            nn.ReLU(),
            nn.Linear(256, projection_dim)
        )
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # 收集所有代理的转换结果和策略值
        transformed_features = []
        policy_values = []
        
        for agent in self.agents:
            transformed, policy_value = agent(x)
            transformed_features.append(transformed)
            policy_values.append(policy_value)
        
        # 将所有转换结果堆叠起来
        transformed_stack = torch.stack(transformed_features, dim=1)  # [batch_size, num_agents, target_dim]
        policy_stack = torch.cat(policy_values, dim=1)  # [batch_size, num_agents]
        
        # 使用softmax归一化策略值，得到每个代理的权重
        agent_weights = F.softmax(policy_stack, dim=1)  # [batch_size, num_agents]
        
        # 使用权重对转换结果进行加权平均
        agent_weights_expanded = agent_weights.unsqueeze(2)  # [batch_size, num_agents, 1]
        weighted_sum = (transformed_stack * agent_weights_expanded).sum(dim=1)  # [batch_size, target_dim]
        
        return weighted_sum, agent_weights, policy_values
    
    def get_projection(self, x):
        """
        获取特征的投影表示，用于对比学习
        """
        # 首先获取转换后的特征
        transformed, _, _ = self.forward(x)
        
        # 然后通过投影头获取投影表示
        projection = self.projection_head(transformed)
        
        return projection

DynamicSemanticTransformNet = MultiAgentSemanticTransform  # 为了向后兼容