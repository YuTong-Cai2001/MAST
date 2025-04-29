import os
import time
import yaml
import pprint
from pathlib import Path
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_distances

import torch
import torch.nn as nn
import torch.optim as optim
from torch import autograd
from torch.autograd import Variable
import torch.nn.functional as F

from models.gdan import CVAE, Discriminator, Regressor
from utils.config_gdan import parser
from utils.data_factory import DataManager
from utils.utils import load_data, update_values, get_negative_samples
from utils.logger import Logger, log_args
from models.semantic_transform import MultiAgentSemanticTransform


args = parser.parse_args()

# if yaml config exists, load and override default ones
if args.config is not None:
    with open(args.config, 'r',encoding="utf-8") as fin:
        options_yaml = yaml.load(fin,Loader=yaml.SafeLoader)
    update_values(options_yaml, vars(args))
9
# 修改数据加载部分
source_data_dir = Path(args.source_data_root)
target_data_dir = Path(args.target_data_root)

source_dir = source_data_dir / Path(args.source_data_name)
target_dir = target_data_dir / Path(args.target_data_name)

source_att_path = source_dir / Path('att_splits.mat')
source_res_path = source_dir / Path('res101.mat')
target_att_path = target_dir / Path('att_splits.mat')
target_res_path = target_dir / Path('res101.mat')

save_dir = Path(args.ckpt_dir)
if not save_dir.is_dir():
    save_dir.mkdir(parents=True)

result_dir = Path(args.result)
if not result_dir.is_dir():
    result_dir.mkdir(parents=True)


result_path = save_dir / Path('gdan_loss.txt')

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

pprint.pprint(vars(args))

log_path = save_dir / Path('gdan_log.txt')
print('log file:', log_path)
logMaster = Logger(str(log_path))
log_args(log_path, args)


def main():
    logger = logMaster.get_logger('main')
    logger.info('loading source and target domain data...')
    
    # 创建必要的目录
    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.result, exist_ok=True)
    
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
    target_unseen_x, target_unseen_y, _ = load_data(args.target_data_name, args.target_data_root, 'test_unseen')
    
    # 确保特征维度正确
    args.x_dim = source_x.shape[1]
    
    # 创建类别映射
    source_classes = np.unique(source_y)
    target_classes = np.unique(target_y)
    
    print(f"源域类别: {source_classes}")
    print(f"目标域类别: {target_classes}")
    
    # 创建训练数据列表，并确保类别索引在有效范围内
    source_train_data = []
    for x, y in zip(source_x, source_y):
        if y < len(source_att):  # 确保类别索引有效
            source_train_data.append((x, y))
    
    target_train_data = []
    for x, y in zip(target_x, target_y):
        if y < len(target_att):  # 确保类别索引有效
            target_train_data.append((x, y))
    
    print(f"源域训练样本数: {len(source_train_data)}")
    print(f"目标域训练样本数: {len(target_train_data)}")
    
    print(f"source_att.shape: {source_att.shape}")
    print(f"target_att.shape: {target_att.shape}")
    print(f"source_y范围: {np.min(source_y)} - {np.max(source_y)}")
    print(f"target_y范围: {np.min(target_y)} - {np.max(target_y)}")
    
    semantic_transform_hidden_dims = [512, 256]
    
    logger.info('building model...')

    # 恢复使用预训练模型
    cvae = CVAE(x_dim=args.x_dim, s_dim=args.source_s_dim, z_dim=args.z_dim,
                enc_layers=args.enc, dec_layers=args.dec)
    
    # 恢复加载预训练模型
    states = torch.load(args.vae_ckpt)
    cvae.load_state_dict(states['model'])
    cvae.cuda()
    cvae.eval()  # 设置为评估模式，因为我们使用预训练的模型

    # 正确传递维度参数
    discriminator = Discriminator(x_dim=args.x_dim, s_dim=args.target_s_dim, layers=args.dis)
    regressor = Regressor(x_dim=args.x_dim, s_dim=args.target_s_dim, layers=args.reg)

    # 注意这里参数顺序应该是source_dim, target_dim
    semantic_transform = MultiAgentSemanticTransform(
        source_dim=args.source_s_dim,
        target_dim=args.target_s_dim,
        num_agents=args.num_agents,
        hidden_dims=[512, 256],
        dropout_rate=0.3
    ).cuda()

    discriminator.cuda()
    regressor.cuda()

    mse_loss = nn.MSELoss()
    l1_loss = nn.L1Loss()

    adam_betas = (0.8, 0.999)
    cvae_opt = optim.Adam(cvae.parameters(), lr=args.learning_rate, weight_decay=0.01, betas=adam_betas)
    dis_opt = optim.Adam(discriminator.parameters(), lr=args.learning_rate, weight_decay=0.01, betas=adam_betas)
    reg_opt = optim.Adam(regressor.parameters(), lr=args.learning_rate, weight_decay=0.01, betas=adam_betas)
    semantic_transform_opt = optim.Adam(semantic_transform.parameters(), lr=args.learning_rate, weight_decay=0.01, betas=adam_betas)

    # 添加学习率调度器
    try:
        min_lr = float(args.min_lr)  # 尝试转换为浮点数
    except (ValueError, TypeError):
        min_lr = 1e-6  # 如果转换失败，使用默认值

    semantic_transform_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        semantic_transform_opt, 
        mode='min', 
        factor=float(args.lr_scheduler_factor),  # 同样确保factor是浮点数
        patience=int(args.lr_scheduler_patience),  # 确保patience是整数
        min_lr=min_lr
    )

    train_manager = DataManager(target_train_data, args.epoch, args.batch)

    ones = Variable(torch.ones([args.batch, 1]), requires_grad=False).float().cuda()
    zeros = Variable(torch.zeros([args.batch, 1]), requires_grad=False).float().cuda()

    loss_history = []
    logger.info('start training...')
    # 初始化早停相关变量
    early_stopping_patience = 10  # 减小早停耐心值
    best_val_loss = float('inf')
    no_improve_count = 0

    # 添加权重记录列表
    agent_weights_history = []
    agent_performance_history = []
    actual_epochs = []

    # 添加验证集评估
    def validate(semantic_transform, val_data):
        semantic_transform.eval()
        with torch.no_grad():
            val_loss = 0
            for batch in val_data:
                # 计算验证损失
                ...
            return val_loss / len(val_data)

    for epoch in range(args.epoch):
        # 设置训练模式
        cvae.train()
        discriminator.train()
        regressor.train()
        semantic_transform.train()

        running_loss = 0
        t1 = time.time()
        d_total_loss = 0.0
        g_total_loss = 0.0
        cyc_total_loss = 0.0
        r_total_loss = 0.0
        rd_total_loss = 0.0
        vae_total_loss = 0.0
        g_scores = 0.0
        epoch_total_loss = 0.0  # 用于记录整个epoch的损失

        # 在每个epoch开始时
        batch_weights = []
        batch_performances = []

        # 添加梯度裁剪
        def clip_gradient(optimizer, grad_clip):
            for group in optimizer.param_groups:
                for param in group['params']:
                    if param.grad is not None:
                        param.grad.data.clamp_(-grad_clip, grad_clip)

        if args.steps == -1:
            steps = train_manager.num_batch
        else:
            steps = args.steps

        # 添加对比学习损失函数
        def contrastive_loss(features1, features2):
            # 计算正样本对的相似度
            batch_size = features1.size(0)
            features1_norm = F.normalize(features1, p=2, dim=1)
            features2_norm = F.normalize(features2, p=2, dim=1)
            
            # 计算余弦相似度矩阵
            similarity_matrix = torch.matmul(features1_norm, features2_norm.transpose(0, 1))
            
            # 对角线上是正样本对
            positives = torch.diag(similarity_matrix)
            
            # 负样本是同一批次中的其他样本
            mask = torch.eye(batch_size, device=features1.device) == 0
            negatives = similarity_matrix[mask].view(batch_size, -1)
            
            # 计算InfoNCE损失
            logits = torch.cat([positives.unsqueeze(1), negatives], dim=1)
            labels = torch.zeros(batch_size, device=features1.device).long()
            
            return F.cross_entropy(logits / 0.07, labels)  # 0.07是温度参数

        # 记录每个batch的Agent权重和性能
        batch_weights = []
        batch_performances = []

        for batch in tqdm(range(steps), leave=False, ncols=70, unit='b'):
            # 训练判别器
            for _ in range(args.d_iter):
                dis_opt.zero_grad()

                data = train_manager.get_batch()
                X = Variable(torch.from_numpy(np.asarray([item[0] for item in data]))).float().cuda()
                Y = [item[1] for item in data]
                
                # 确保所有类别索引都在有效范围内
                valid_indices = [i for i, y in enumerate(Y) if y < len(source_att)]
                if len(valid_indices) == 0:
                    continue  # 如果没有有效索引，跳过这个批次
                
                X = X[valid_indices]
                Y = [Y[i] for i in valid_indices]
                
                # 使用source_att而不是source_att_feats
                S = Variable(torch.from_numpy(source_att[Y])).float().cuda()
                S_target = Variable(torch.from_numpy(target_att[Y])).float().cuda()

                # 使用动态语义转换
                S_transformed, agent_weights, policy_values = semantic_transform(S)
                
                # 确保所有批次使用相同的批次大小
                fixed_batch_size = args.batch
                if agent_weights.shape[0] == fixed_batch_size:
                    # 记录Agent权重和性能
                    batch_weights.append(agent_weights.detach().cpu().numpy())
                    batch_performances.append(torch.cat([p.detach() for p in policy_values], dim=1).cpu().numpy())

                # 生成样本
                Xp, mu, log_sigma = cvae.forward(X, S_transformed)

                # 判别器前向传播
                real_scores = discriminator.forward(X, S_target)
                fake_scores = discriminator.forward(Xp.detach(), S_target)

                # 计算判别器损失
                d_loss = -torch.mean(torch.log(real_scores + 1e-10) + torch.log(1 - fake_scores + 1e-10))
                d_loss.backward()
                dis_opt.step()

                d_total_loss += d_loss.cpu().data.numpy()

            # 训练生成器
            for _ in range(args.g_iter):
                cvae_opt.zero_grad()
                reg_opt.zero_grad()
                semantic_transform_opt.zero_grad()

                data = train_manager.get_batch()
                X = Variable(torch.from_numpy(np.asarray([item[0] for item in data]))).float().cuda()
                Y = [item[1] for item in data]
                
                # 确保所有类别索引都在有效范围内
                valid_indices = [i for i, y in enumerate(Y) if y < len(source_att)]
                if len(valid_indices) == 0:
                    continue  # 如果没有有效索引，跳过这个批次
                
                X = X[valid_indices]
                Y = [Y[i] for i in valid_indices]
                
                # 使用source_att而不是source_att_feats
                S = Variable(torch.from_numpy(source_att[Y])).float().cuda()
                S_target = Variable(torch.from_numpy(target_att[Y])).float().cuda()

                # 使用动态语义转换
                S_transformed, agent_weights, policy_values = semantic_transform(S)
                
                # 确保所有批次使用相同的批次大小
                fixed_batch_size = args.batch
                if agent_weights.shape[0] == fixed_batch_size:
                    # 记录Agent权重和性能
                    batch_weights.append(agent_weights.detach().cpu().numpy())
                    batch_performances.append(torch.cat([p.detach() for p in policy_values], dim=1).cpu().numpy())

                # 生成样本
                Xp, mu, log_sigma = cvae.forward(X, S_transformed)

                # 计算VAE损失
                vae_loss = cvae.vae_loss(X, Xp, mu, log_sigma)

                # 判别器前向传播
                fake_scores = discriminator.forward(Xp, S_target)

                # 计算生成器损失
                g_loss = -torch.mean(torch.log(fake_scores + 1e-10))

                # 计算循环一致性损失
                Xpp, _, _ = cvae.forward(Xp, S_transformed)
                cyc_loss = mse_loss(Xpp, X)

                # 计算回归损失
                S_pred = regressor.forward(Xp)
                r_loss = mse_loss(S_pred, S_target)

                # 计算回归判别损失
                S_real = regressor.forward(X)
                rd_loss = mse_loss(S_real, S_target)

                # 计算KL散度损失 - 对齐源域和目标域的分布
                kl_loss = F.kl_div(
                    F.log_softmax(S_transformed, dim=1),
                    F.softmax(S_target, dim=1),
                    reduction='batchmean'
                )

                # 计算对比学习损失
                # 获取投影表示
                proj_source = semantic_transform.get_projection(S)
                proj_target = semantic_transform.get_projection(S_target)
                
                # 计算对比损失
                contr_loss = contrastive_loss(proj_source, proj_target)

                # 计算语义一致性损失
                semantic_consistency_loss = 0
                semantic_scores = []
                for agent in semantic_transform.agents:
                    # 使用新的compute_semantic_score方法
                    source_score = agent.compute_semantic_score(S)
                    target_score = agent.compute_semantic_score(S_target)
                    semantic_consistency_loss += F.mse_loss(source_score, target_score)
                    semantic_scores.append(source_score)

                # 初始化gates_list
                gates_list = []

                # 收集所有代理的门控值
                for agent in semantic_transform.agents:
                    # 使用策略网络的输出作为门控值
                    gates = agent.policy_net(S)
                    gates_list.append(gates)

                # 恢复路由多样性损失计算
                routing_diversity_loss = 0
                for gates in gates_list:
                    routing_diversity_loss += -torch.mean(torch.sum(gates * torch.log(gates + 1e-6), dim=1))
                routing_diversity_loss /= len(gates_list)

                # 设置默认权重
                semantic_consistency_weight = getattr(args, 'semantic_consistency_weight', 0.1)
                routing_diversity_weight = getattr(args, 'routing_diversity_weight', 0.05)
                contrastive_weight = getattr(args, 'contrastive_weight', 0.2)
                kl_weight = getattr(args, 'kl_weight', 0.1)

                # 计算总损失
                total_loss = vae_loss + args.theta1 * g_loss + args.theta2 * cyc_loss + \
                             args.theta3 * r_loss + args.theta4 * rd_loss + \
                             kl_weight * kl_loss + \
                             contrastive_weight * contr_loss + \
                             semantic_consistency_weight * semantic_consistency_loss + \
                             routing_diversity_weight * routing_diversity_loss

                # 恢复协作熵计算
                agent_loss = 0
                # 定义目标值 - 可以根据任务调整
                target_value = torch.ones_like(policy_values[0]) * 0.5  # 设置为0.5作为中间目标
                for policy_value in policy_values:
                    agent_loss += F.mse_loss(policy_value, target_value)
                
                collaboration_entropy = -(agent_weights * torch.log(agent_weights + 1e-6)).sum(1).mean()
                
                # 在计算总损失之前添加
                total_loss += args.agent_weight * agent_loss + args.entropy_weight * collaboration_entropy

                total_loss.backward()

                # 在优化器步骤之前添加梯度裁剪
                clip_gradient(semantic_transform_opt, args.gradient_clip)
                semantic_transform_opt.step()

                cvae_opt.step()
                reg_opt.step()

                epoch_total_loss += total_loss.item()

                vae_total_loss += vae_loss.cpu().data.numpy()
                g_total_loss += g_loss.cpu().data.numpy()
                cyc_total_loss += cyc_loss.cpu().data.numpy()
                r_total_loss += r_loss.cpu().data.numpy()
                rd_total_loss += rd_loss.cpu().data.numpy()
                g_scores += np.mean(fake_scores.cpu().data.numpy())

            # 在每个batch结束时计算平均值
            if batch_weights:  # 确保列表不为空
                try:
                    # 尝试转换为numpy数组
                    batch_weights = np.array(batch_weights)
                    batch_performances = np.array(batch_performances)
                    
                    # 计算平均值
                    batch_avg_weights = np.mean(batch_weights, axis=0)
                    batch_avg_performances = np.mean(batch_performances, axis=0)
                    
                    # 添加到epoch记录
                    batch_weights = []
                    batch_performances = []
                except ValueError as e:
                    # 如果形状不一致，打印警告并跳过
                    print(f"警告: 批次权重形状不一致，跳过此批次的统计。错误: {e}")
                    
                    # 清空批次记录，准备下一个epoch
                    batch_weights = []
                    batch_performances = []

        # 在epoch结束时计算整个epoch的平均值
        if batch_weights:  # 确保列表不为空
            epoch_weights = np.array(batch_weights)  # 转换为numpy数组
            epoch_performance = np.array(batch_performances)
            avg_weights = np.mean(epoch_weights, axis=0)
            avg_performance = np.mean(epoch_performance, axis=0)
            agent_weights_history.append(avg_weights)
            agent_performance_history.append(avg_performance)
            actual_epochs.append(epoch)

        # 在epoch结束时进行早停检查
        avg_epoch_loss = epoch_total_loss / steps
        semantic_transform_scheduler.step(avg_epoch_loss)
        
        if avg_epoch_loss < best_val_loss:
            best_val_loss = avg_epoch_loss
            no_improve_count = 0
        else:
            no_improve_count += 1

        if no_improve_count >= early_stopping_patience:
            logger.info(f'Early stopping at epoch {epoch}')
            break

        g_total_steps = steps * args.g_iter
        d_total_steps = steps * args.d_iter
        vae_avg_loss = vae_total_loss / g_total_steps
        g_avg_loss = g_total_loss / g_total_steps
        cyc_avg_loss = cyc_total_loss / g_total_steps
        r_avg_loss = r_total_loss / g_total_steps
        rd_avg_loss = rd_total_loss / g_total_steps
        d_avg_loss = d_total_loss / d_total_steps
        g_avg_score = g_scores / g_total_steps
        loss_history.append(f'{g_avg_loss:.4}\t{d_avg_loss:.4}\t{cyc_avg_loss:.4}\t{r_avg_loss:.4}\t'
                            f'{rd_avg_loss:.4}\t{g_avg_score:.4}\t{vae_avg_loss:.4}\n')
        elapsed = (time.time() - t1)/60.0

        if (epoch+1) % 10 == 0 or epoch == 0:
            filename = 'gdan_' + str(epoch + 1) + '.pkl'
            save_path = save_dir / Path(filename)
            states = dict()
            states['cvae'] = cvae.state_dict()
            states['discriminator'] = discriminator.state_dict()
            states['regressor'] = regressor.state_dict()
            states['semantic_transform'] = semantic_transform.state_dict()
            
            # 使用传入semantic_transform的hidden_dims参数，而不是尝试访问对象属性
            states['semantic_transform_hidden_dims'] = semantic_transform_hidden_dims
            
            torch.save(states, str(save_path))
            logger.info('model saved to {}'.format(str(save_path)))

    # 保存分析数据
    analysis_data = {
        'agent_weights_history': agent_weights_history,
        'agent_performance_history': agent_performance_history,
        'epochs': actual_epochs
    }
    torch.save(analysis_data, str(save_dir / 'agent_analysis.pt'))

    with result_path.open('w') as fout:
        for s in loss_history:
            fout.write(s)

    logger.info('program finished')


def augment_features(X, noise_level=0.01):
    """添加高斯噪声进行特征增强"""
    noise = torch.randn_like(X) * noise_level
    return X + noise


if __name__ == '__main__':
    main()
