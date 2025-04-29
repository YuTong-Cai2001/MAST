import os
import time
import pprint
from pathlib import Path
from tqdm import tqdm
import numpy as np
import yaml
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_distances

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

from models.gdan import CVAE, Discriminator, Regressor
from utils.config_gdan import parser
from utils.data_factory import DataManager
from utils.utils import load_data, update_values
from utils.logger import Logger
from models.semantic_transform import MultiAgentSemanticTransform


args = parser.parse_args()

# if yaml config exists, load and override default ones
if args.config is not None:
    with open(args.config, 'r',encoding="utf-8") as fin:
        options_yaml = yaml.load(fin, Loader=yaml.SafeLoader)
    update_values(options_yaml, vars(args))

# 添加源域和目标域的配置加载
source_data_dir = Path(args.source_data_root)
target_data_dir = Path(args.target_data_root)

source_dir = source_data_dir / Path(args.source_data_name)
target_dir = target_data_dir / Path(args.target_data_name)

source_att_path = source_dir / Path('att_splits.mat')
source_res_path = source_dir / Path('res101.mat')
target_att_path = target_dir / Path('att_splits.mat')
target_res_path = target_dir / Path('res101.mat')

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

pprint.pprint(vars(args))

save_dir = Path(args.vae_dir)
if not save_dir.is_dir():
    save_dir.mkdir(parents=True)

log_file = save_dir / Path('log_cvae.txt')
logMaster = Logger(str(log_file))


def main():
    # 创建必要的目录
    os.makedirs(args.vae_dir, exist_ok=True)
    
    # 加载源域数据
    source_x, source_y, source_att = load_data(args.source_data_name, args.source_data_root, 'train')
    source_test_x, source_test_y, _ = load_data(args.source_data_name, args.source_data_root, 'test_seen')
    
    # 打印调试信息
    print(f"源域训练标签唯一值: {np.unique(source_y)}")
    print(f"源域语义属性形状: {source_att.shape}")
    
    # 确保语义属性矩阵的大小与类别数量匹配
    unique_classes = np.unique(source_y)
    if len(unique_classes) != source_att.shape[0]:
        print("警告: 语义属性矩阵大小不匹配，进行调整...")
        
        # 创建新的语义属性矩阵
        new_att = np.zeros((len(unique_classes), source_att.shape[1]))
        
        # 复制原始属性
        for i, cls in enumerate(unique_classes):
            if i < source_att.shape[0]:
                new_att[i] = source_att[i]
        
        # 使用新的属性矩阵
        source_att = new_att
        print(f"调整后的语义属性形状: {source_att.shape}")
    
    # 加载目标域数据
    target_x, target_y, target_att = load_data(args.target_data_name, args.target_data_root, 'train')
    target_test_x, target_test_y, _ = load_data(args.target_data_name, args.target_data_root, 'test_seen')
    
    # 确保特征维度正确
    args.x_dim = source_x.shape[1]
    
    # 创建训练数据列表
    source_train_data = list(zip(source_x, source_y))
    target_train_data = list(zip(target_x, target_y))
    
    logger = logMaster.get_logger('main')
    logger.info('building model...')
    
    # 首先初始化CVAE
    cvae = CVAE(x_dim=args.x_dim, 
                s_dim=args.source_s_dim,
                z_dim=args.z_dim, 
                enc_layers=args.enc, 
                dec_layers=args.dec)
    cvae.cuda()
    
    # 初始化semantic_transform
    semantic_transform = MultiAgentSemanticTransform(
        target_dim=args.target_s_dim,
        source_dim=args.source_s_dim,
        num_agents=args.num_agents,
        hidden_dims=[512, 256],
        dropout_rate=0.3
    ).cuda()
    
    # 然后初始化优化器
    cvae_opt = optim.Adam(cvae.parameters(), lr=args.learning_rate * 0.1, weight_decay=0.01)
    semantic_transform_opt = optim.Adam(semantic_transform.parameters(), 
                                      lr=args.learning_rate * 0.1, 
                                      weight_decay=0.01)

    # 使用源域数据进行训练
    train_manager = DataManager(source_train_data, args.epoch, args.batch)

    logger.info('start training on source domain...')
    for epoch in range(500):
        running_loss = 0
        t1 = time.time()
        cvae.train()
        semantic_transform.train()
        
        for batch in tqdm(range(train_manager.num_batch), leave=False, ncols=70, unit='b'):
            cvae_opt.zero_grad()
            semantic_transform_opt.zero_grad()
            
            data = train_manager.get_batch()
            X = Variable(torch.from_numpy(np.asarray([item[0] for item in data]))).float().cuda()
            Y = [item[1] for item in data]
            S = Variable(torch.from_numpy(source_att[Y])).float().cuda()

            # 使用多Agent语义转换
            transformed, agent_weights, policy_values = semantic_transform(S)
            
            # VAE重建损失 - 添加归一化
            Xp, mu, log_sigma = cvae.forward(X, transformed)
            loss_vae = cvae.vae_loss(X, Xp, mu, log_sigma)
            loss_vae = loss_vae / X.size(0)  # 按批次大小归一化

            # Agent多样性损失 - 调整权重
            diversity_loss = -(agent_weights * torch.log(agent_weights + 1e-6)).sum(1).mean()
            
            # Agent策略损失 - 修改目标值和计算方式
            agent_loss = 0
            target_value = torch.ones(policy_values[0].shape).cuda() * 0.5
            for policy_value in policy_values:
                agent_loss += F.mse_loss(policy_value, target_value, reduction='mean')
            agent_loss = agent_loss / len(policy_values)  # 平均每个agent的损失

            # 总损失 - 调整权重
            total_loss = loss_vae + \
                        args.agent_weight * 0.01 * agent_loss + \
                        args.entropy_weight * 0.01 * diversity_loss

            total_loss.backward()
            
            # 添加梯度裁剪
            torch.nn.utils.clip_grad_norm_(cvae.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(semantic_transform.parameters(), 1.0)
            
            cvae_opt.step()
            semantic_transform_opt.step()

            running_loss += total_loss.item()  # 使用item()而不是numpy()
            
        # 计算平均损失
        avg_loss = running_loss / train_manager.num_batch
        
        if (epoch+1) % 10 == 0:
            logger.info(f'epoch: {epoch+1:4}, avg_loss: {avg_loss:.5f}, ' + 
                       f'vae_loss: {loss_vae.item():.5f}, ' +
                       f'agent_loss: {agent_loss.item():.5f}, ' +
                       f'diversity_loss: {diversity_loss.item():.5f}')
            
            # 保存检查点
            filename = f'cvae_cross_{args.source_data_name}_to_{args.target_data_name}_{epoch+1}.pkl'
            save_path = save_dir / Path(filename)
            states = {
                'model': cvae.state_dict(),
                'semantic_transform': semantic_transform.state_dict(),
                'z_dim': args.z_dim,
                'x_dim': args.x_dim,
                's_dim': args.source_s_dim,
                'optim': cvae_opt.state_dict(),
                'semantic_transform_optim': semantic_transform_opt.state_dict(),
                'num_agents': args.num_agents,
                'agent_weights': agent_weights.cpu().data,
                'policy_values': [p.cpu().data for p in policy_values],
                'epoch_loss': {
                    'avg_loss': avg_loss,
                    'vae_loss': loss_vae.item(),
                    'agent_loss': agent_loss.item(),
                    'diversity_loss': diversity_loss.item()
                }
            }
            torch.save(states, str(save_path))

    # 在训练结束时保存最终模型
    final_filename = f'cvae_cross_{args.source_data_name}_to_{args.target_data_name}_final.pkl'
    final_save_path = save_dir / Path(final_filename)
    final_states = {
        'model': cvae.state_dict(),
        'semantic_transform': semantic_transform.state_dict(),
        'z_dim': args.z_dim,
        'x_dim': args.x_dim,
        's_dim': args.source_s_dim,
        'optim': cvae_opt.state_dict(),
        'semantic_transform_optim': semantic_transform_opt.state_dict(),
        'num_agents': args.num_agents,
        'agent_weights': agent_weights.cpu().data,
        'policy_values': [p.cpu().data for p in policy_values],
        'final_loss': {
            'avg_loss': avg_loss,
            'vae_loss': loss_vae.item(),
            'agent_loss': agent_loss.item(),
            'diversity_loss': diversity_loss.item()
        }
    }
    torch.save(final_states, str(final_save_path))
    logger.info(f'Final model saved to {final_save_path}')


if __name__ == '__main__':
    main()
