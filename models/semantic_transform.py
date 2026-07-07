import torch
import torch.nn as nn
import torch.nn.functional as F

class SemanticAgent(nn.Module):
    def __init__(self, target_dim, source_dim, hidden_dims=[512, 256], dropout_rate=0.3, agent_id=0):
        super().__init__()
        self.agent_id = agent_id
        self.hidden_dims = hidden_dims
        self.target_dim = target_dim
        self.source_dim = source_dim
        
        # 使用目标域维度初始化语义焦点
        self.semantic_focus = nn.Parameter(torch.randn(target_dim) * 0.01)
        
        # 添加双向维度转换层
        self.source_to_target = None
        self.target_to_target = None
        if source_dim != target_dim:
            self.source_to_target = nn.Linear(source_dim, target_dim)  # 用于源域输入
            self.target_to_target = nn.Identity()  # 用于目标域输入
        
        # Agent特定的编码器 - 使用目标域维度
        self.dropout = nn.Dropout(dropout_rate * 1.5)
        self.target_encoder = nn.Sequential(
            nn.Linear(target_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.ReLU(),
            self.dropout,
            nn.BatchNorm1d(hidden_dims[0])
        )
        
        # Agent特定的转换器
        self.transform = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.LayerNorm(hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Agent特定的投影头
        self.projector = nn.Sequential(
            nn.Linear(hidden_dims[1], 128),
            nn.LayerNorm(128)
        )
        
        # 策略网络 - 用于强化学习
        self.policy_net = nn.Sequential(
            nn.Linear(hidden_dims[1], 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # 添加语义正则化器
        self.semantic_regularizer = nn.Sequential(
            nn.Linear(target_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], 1),
            nn.Sigmoid()  # 确保输出在0-1之间
        )
        
    def forward(self, x):
        # 根据输入维度选择合适的转换
        if x.size(1) == self.source_dim:  # 源域输入
            x = self.source_to_target(x)
        else:  # 目标域输入
            x = self.target_to_target(x)
            
        # 基于语义焦点的注意力
        attention = torch.sigmoid(self.semantic_focus)
        x_focused = x * attention
        
        # 特征转换
        features = self.target_encoder(x_focused)
        transformed = self.transform(features)
        
        # 计算策略值
        policy_value = self.policy_net(transformed)
        
        return transformed, policy_value

    def compute_semantic_score(self, x):
        """计算语义得分，包含必要的维度转换"""
        # 首先进行维度转换
        if x.size(1) == self.source_dim:  # 源域输入
            x = self.source_to_target(x)
        else:  # 目标域输入
            x = self.target_to_target(x)
            
        # 计算语义得分
        return self.semantic_regularizer(x)

class MultiAgentSemanticTransform(nn.Module):
    def __init__(self, target_dim, source_dim, num_agents=2, hidden_dims=[512, 256], dropout_rate=0.3):
        super().__init__()
        self.num_agents = num_agents
        self.target_dim = target_dim
        self.source_dim = source_dim
        self.hidden_dims = hidden_dims  # 保存hidden_dims参数
        
        # 创建多个语义转换agent
        self.agents = nn.ModuleList([
            SemanticAgent(target_dim, source_dim, hidden_dims, dropout_rate, i)
            for i in range(num_agents)
        ])
        
        # 修改协作机制的输入维度
        self.collaboration_gate = nn.Sequential(
            nn.Linear(hidden_dims[1] * num_agents, num_agents),
            nn.Softmax(dim=1)
        )
        
        # 添加最终的维度转换层 - 从hidden_dim转换到source_dim
        self.dim_transform = nn.Linear(hidden_dims[1], source_dim)
        
        # 添加投影头
        self.projector = nn.Sequential(
            nn.Linear(hidden_dims[1], 128),
            nn.LayerNorm(128)
        )
        
        self.temperature = nn.Parameter(torch.ones(1))
        
        # 添加权重初始化
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        
    def get_projection(self, x, is_target=True):
        """获取特征的投影表示"""
        agent_outputs = []
        policy_values = []
        
        # 收集所有agent的输出
        for agent in self.agents:
            transformed, policy_value = agent(x)
            agent_outputs.append(transformed)
            policy_values.append(policy_value)
            
        # 计算协作权重
        combined_features = torch.cat(agent_outputs, dim=1)
        weights = self.collaboration_gate(combined_features)
        
        # 加权融合
        weighted_sum = torch.zeros_like(agent_outputs[0])
        for i, output in enumerate(agent_outputs):
            weighted_sum += output * weights[:, i].unsqueeze(1)
            
        # 在维度转换之前进行投影
        projection = self.projector(weighted_sum)
        
        return projection

    def get_last_attention_weights(self):
        """获取最后一次前向传播的注意力权重"""
        attention_weights = []
        for agent in self.agents:
            attention_weights.append(torch.sigmoid(agent.semantic_focus))
        return torch.stack(attention_weights, dim=0).mean(dim=0)

    def get_last_gates(self):
        """获取最后一次前向传播的门控信号"""
        return [self.collaboration_gate[0].weight.data]  # 返回协作门控的权重
        
    def forward(self, x):
        agent_outputs = []
        policy_values = []
        
        # 收集所有agent的输出
        for agent in self.agents:
            transformed, policy_value = agent(x)
            agent_outputs.append(transformed)
            policy_values.append(policy_value)
            
        # 计算协作权重
        combined_features = torch.cat(agent_outputs, dim=1)
        weights = F.softmax(
            self.collaboration_gate(combined_features) / self.temperature, 
            dim=1
        )
        
        # 加权融合
        weighted_sum = torch.zeros_like(agent_outputs[0])
        for i, output in enumerate(agent_outputs):
            weighted_sum += output * weights[:, i].unsqueeze(1)
            
        # 最终转换到源域维度
        final_output = self.dim_transform(weighted_sum)
        
        return final_output, weights, policy_values

DynamicSemanticTransformNet = MultiAgentSemanticTransform  # 为了向后兼容