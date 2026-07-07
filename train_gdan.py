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
    

    semantic_transform_hidden_dims = [512, 256]
    
    source_att_feats, source_train_data, source_val_data, source_test_data, source_test_s_data, source_classes = \
        load_data(att_path=source_att_path, res_path=source_res_path)
    
    target_att_feats, target_train_data, target_val_data, target_test_data, target_test_s_data, target_classes = \
        load_data(att_path=target_att_path, res_path=target_res_path)

    logger.info('building model...')

    # 加载预训练的VAE模型（在源域上训练的）
    cvae = CVAE(x_dim=args.x_dim, s_dim=args.source_s_dim, z_dim=args.z_dim,
                enc_layers=args.enc, dec_layers=args.dec)
    
    states = torch.load(args.vae_ckpt)
    cvae.load_state_dict(states['model'])
    cvae.cuda()
    cvae.eval()  # 设置为评估模式

    # 正确传递维度参数
    discriminator = Discriminator(x_dim=args.x_dim, s_dim=args.target_s_dim, layers=args.dis)
    regressor = Regressor(x_dim=args.x_dim, s_dim=args.target_s_dim, layers=args.reg)


    semantic_transform = MultiAgentSemanticTransform(
        target_dim=args.target_s_dim,
        source_dim=args.source_s_dim,
        num_agents=3,  # 确保使用3个Agent
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

        # 在训练循环外初始化列表
        epoch_weights = []
        epoch_performance = []

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
            pos_sim = F.cosine_similarity(features1, features2, dim=1)
            
            # 计算负样本对的相似度
            neg_sim = []
            for i in range(features1.size(0)):
                neg = features2[torch.arange(features2.size(0)) != i]
                sim = F.cosine_similarity(features1[i:i+1], neg, dim=1)
                neg_sim.append(sim)
            neg_sim = torch.stack(neg_sim)
            
            # 计算对比损失
            loss = -torch.log(
                torch.exp(pos_sim) / 
                (torch.exp(pos_sim) + torch.sum(torch.exp(neg_sim), dim=1))
            ).mean()
            
            return loss

        # 添加学习率调度器
        semantic_transform_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            semantic_transform_opt, 
            mode='min', 
            factor=0.5,
            patience=10,
            min_lr=1e-6
        )

        for batch in tqdm(range(steps), leave=False, ncols=70, unit='b'):
            batch_weights = []
            batch_performances = []
            
            for i in range(args.d_iter):
                discriminator.zero_grad()

                # get true data
                data = train_manager.get_batch()
                batch_size = len(data)  # 获取实际的批次大小
                
                # 根据实际批次大小创建ones和zeros
                ones_batch = Variable(torch.ones([batch_size, 1]), requires_grad=False).float().cuda()
                zeros_batch = Variable(torch.zeros([batch_size, 1]), requires_grad=False).float().cuda()

                X = Variable(torch.from_numpy(np.asarray([item[0] for item in data]))).float().cuda()
                Y = [item[1] for item in data]
                S_target = Variable(torch.from_numpy(target_att_feats[Y])).float().cuda()
                S, agent_weights, policy_values = semantic_transform(S_target)
                
                Yc = get_negative_samples(Y, target_classes['train'])
                Sc_target = Variable(torch.from_numpy(target_att_feats[Yc])).float().cuda()
                Sc, sc_weights, sc_values = semantic_transform(Sc_target)

                # get fake data
                Xp, _, _ = cvae.forward(X, S)
                Xp = Xp.detach()
                Xpp = cvae.sample(S).detach()
                Sp_target = regressor.forward(X).detach()
                Sp, sp_weights, sp_values = semantic_transform(Sp_target)

                # get scores
                true_scores = discriminator.forward(X, S_target)
                fake_scores = discriminator.forward(Xp, S_target)
                fake_scores2 = discriminator.forward(Xpp, S_target)
                reg_scores = discriminator.forward(X, Sp_target)
                ctrl_scores = discriminator.forward(X, Sc_target)

                # calculate loss using batch-specific ones and zeros
                d_loss = mse_loss(true_scores, ones_batch) + \
                        mse_loss(fake_scores, zeros_batch) + \
                        args.theta3 * mse_loss(reg_scores, zeros_batch) + \
                        mse_loss(ctrl_scores, zeros_batch)

                d_loss.backward()
                dis_opt.step()

                d_total_loss += d_loss.cpu().data.numpy()

                # 记录agent权重 - 修改这部分
                transformed, agent_weights, policy_values = semantic_transform(S_target)
                # 确保转换为numpy数组并保持一致的维度
                weights_np = agent_weights.detach().cpu().numpy()
                # 如果weights_np是2D数组，取平均值使其变为1D
                if len(weights_np.shape) > 1:
                    weights_np = np.mean(weights_np, axis=0)
                batch_weights.append(weights_np)
                
                # 计算每个agent的单独性能
                agent_performances = []
                for i in range(args.num_agents):
                    agent_output, _ = semantic_transform.agents[i](S_target)
                    agent_output = semantic_transform.dim_transform(agent_output)
                    recon_loss = F.mse_loss(agent_output, S.detach()).item()
                    agent_performances.append(recon_loss)
                batch_performances.append(np.array(agent_performances))  # 转换为numpy数组

            for i in range(args.g_iter):
                cvae.zero_grad()
                regressor.zero_grad()
                semantic_transform.zero_grad()

                data = train_manager.get_batch()
                batch_size = len(data)  # 获取实际的批次大小
                
                # 根据实际批次大小创建ones和zeros
                ones_batch = Variable(torch.ones([batch_size, 1]), requires_grad=False).float().cuda()
                zeros_batch = Variable(torch.zeros([batch_size, 1]), requires_grad=False).float().cuda()

                X = Variable(torch.from_numpy(np.asarray([item[0] for item in data]))).float().cuda()
                Y = [item[1] for item in data]
                S_target = Variable(torch.from_numpy(target_att_feats[Y])).float().cuda()
                S, agent_weights, policy_values = semantic_transform(S_target)

                Yc = get_negative_samples(Y, target_classes['train'])
                Sc_target = Variable(torch.from_numpy(target_att_feats[Yc])).float().cuda()
                Sc, sc_weights, sc_values = semantic_transform(Sc_target)

                # get fake data
                Xp, mu, log_sigma = cvae.forward(X, S)
                Xp2 = cvae.sample(S)
                Sp_target = regressor.forward(X)
                Sp, sp_weights, sp_values = semantic_transform(Sp_target)
                Spp_target = regressor.forward(Xp)
                Spp, spp_weights, spp_values = semantic_transform(Spp_target)
                Xpp, _, _ = cvae.forward(X, Sp)

                # get scores (使用目标域维度的特征)
                fake_scores = discriminator.forward(Xp, S_target)
                fake_scores2 = discriminator.forward(Xp2, S_target)
                reg_scores = discriminator.forward(X, Sp_target)

                # calculate loss
                vae_loss = cvae.vae_loss(X=X, Xp=Xp, mu=mu, log_sigma=log_sigma)
                cyc_loss = mse_loss(Spp_target, S_target) + mse_loss(Xpp, X)

                g_loss = mse_loss(fake_scores, ones_batch)
                r_loss = mse_loss(Sp, S)
                rd_loss = mse_loss(reg_scores, ones_batch)

                # 获取源域和目标域语义特征的投影
                source_proj = semantic_transform.get_projection(S, is_target=False)
                target_proj = semantic_transform.get_projection(S_target, is_target=True)

                # 计算对比损失
                contr_loss = contrastive_loss(source_proj, target_proj)

                # 计算分布对齐正则化损失
                kl_loss = torch.mean(torch.pow(torch.mean(source_proj, dim=0) - 
                                              torch.mean(target_proj, dim=0), 2))

                # 获取最后一次前向传播的注意力权重和门控信号
                attention_weights = semantic_transform.get_last_attention_weights()
                gates_list = semantic_transform.get_last_gates()

                # 计算语义一致性损失
                semantic_consistency_loss = 0
                semantic_scores = []
                for agent in semantic_transform.agents:
                    # 使用新的compute_semantic_score方法
                    source_score = agent.compute_semantic_score(S)
                    target_score = agent.compute_semantic_score(S_target)
                    semantic_consistency_loss += F.mse_loss(source_score, target_score)
                    semantic_scores.append(source_score)

                # 计算路由多样性损失 - 对所有层的门控信号计算
                routing_diversity_loss = 0
                for gates in gates_list:
                    routing_diversity_loss += -torch.mean(torch.sum(gates * torch.log(gates + 1e-6), dim=1))
                routing_diversity_loss /= len(gates_list)

                # 添加预热阶段
                warmup_epochs = 100
                if epoch < warmup_epochs:
                    # 第一阶段：只使用基本的转换损失
                    total_loss = vae_loss + g_loss + args.theta1 * cyc_loss + \
                                 args.theta2 * r_loss + args.theta3 * rd_loss
                    
                    # 逐步引入其他损失
                    if epoch > warmup_epochs // 2:
                        # 第二阶段：添加对比学习和分布对齐
                        weight = (epoch - warmup_epochs // 2) / (warmup_epochs // 2)
                        total_loss += weight * (args.theta4 * contr_loss + args.theta5 * kl_loss)
                else:
                    # 最终阶段：完整的损失函数
                    total_loss = vae_loss + g_loss+ args.theta3 * rd_loss + \
                                 args.theta4 * contr_loss + args.theta5 * kl_loss + \
                                 args.theta6 * semantic_consistency_loss + \
                                 args.theta7 * routing_diversity_loss
                     #+ args.theta1 * cyc_loss +args.theta2 * r_loss 

                # 添加额外的损失项
                agent_loss = 0
                # 定义目标值 - 可以根据任务调整
                target_value = torch.ones_like(policy_values[0]) * 0.5  # 设置为0.5作为中间目标
                for policy_value in policy_values:
                    agent_loss += F.mse_loss(policy_value, target_value)

                collaboration_entropy = -(agent_weights * torch.log(agent_weights + 1e-6)).sum(1).mean()
                
                total_loss += args.agent_weight * agent_loss + args.entropy_weight * collaboration_entropy
                total_loss += args.semantic_consistency_weight * semantic_consistency_loss

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
                batch_weights = np.array(batch_weights)  # 转换为numpy数组
                batch_performances = np.array(batch_performances)
                batch_avg_weights = np.mean(batch_weights, axis=0)
                batch_avg_performances = np.mean(batch_performances, axis=0)
                epoch_weights.append(batch_avg_weights)
                epoch_performance.append(batch_avg_performances)

        # 在epoch结束时计算整个epoch的平均值
        if epoch_weights:  # 确保列表不为空
            epoch_weights = np.array(epoch_weights)  # 转换为numpy数组
            epoch_performance = np.array(epoch_performance)
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
            states['epoch'] = epoch + 1
            states['cvae'] = cvae.state_dict()
            states['discriminator'] = discriminator.state_dict()
            states['regressor'] = regressor.state_dict()
            states['semantic_transform'] = semantic_transform.state_dict()
            states['semantic_transform_hidden_dims'] = semantic_transform.hidden_dims
            states['enc_layers'] = args.enc
            states['dec_layers'] = args.dec
            states['reg_layers'] = args.reg
            states['dis_layers'] = args.dis
            states['z_dim'] = args.z_dim
            states['x_dim'] = args.x_dim
            states['source_s_dim'] = args.source_s_dim
            states['target_s_dim'] = args.target_s_dim
            states['cvae_opt'] = cvae_opt.state_dict()
            states['dis_opt'] = dis_opt.state_dict()
            states['reg_opt'] = reg_opt.state_dict()
            states['semantic_transform_opt'] = semantic_transform_opt.state_dict()
            states['theta1'] = args.theta1
            states['theta2'] = args.theta2
            states['theta3'] = args.theta3
            states['theta4'] = args.theta4
            states['theta5'] = args.theta5
            states['theta6'] = args.theta6
            states['theta7'] = args.theta7
            states['num_agents'] = args.num_agents
            states['agent_weights'] = agent_weights.cpu().data
            states['policy_values'] = [p.cpu().data for p in policy_values]
            states['loss_values'] = {
                'kl_loss': kl_loss.item(),
                'semantic_consistency_loss': semantic_consistency_loss.item(),
                'semantic_alignment_loss': (kl_loss + semantic_consistency_loss).item()
            }

            torch.save(states, str(save_path))
            logger.info(f'epoch: {epoch+1:4}, g_loss: {g_avg_loss: .4}, d_loss: {d_avg_loss: .4}, \n'
                        f'cyc_loss: {cyc_avg_loss: .4}, r_loss: {r_avg_loss: .4}, rd_loss: {rd_avg_loss: .4}, '
                        f'g_score: {g_avg_score:.4}, vae loss: {vae_avg_loss:.4}')

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
