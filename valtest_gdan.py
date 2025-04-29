import os
import time
import pprint
from pathlib import Path
import argparse
import yaml
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_distances
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from models.gdan import CVAE, Discriminator, Regressor
from models.semantic_transform import MultiAgentSemanticTransform
from utils.data_factory import DataManager
from utils.utils import load_data, update_values, get_datetime_str
from utils.logger import Logger, log_args


parser = argparse.ArgumentParser(description='argument parser')

parser.add_argument('-cfg', '--config', metavar='YAML', default=None,
                    help='path to yaml config file')

# files and directories
parser.add_argument('-dn', '--data_name', metavar='NAME', default='CUB',
                    choices=['CUB', 'SUN', 'APY', 'AWA1', 'AWA2', 'ImageNet'],
                    help='name of dataset')
parser.add_argument('-d', '--data_root', metavar='DIR', default='./ZSL-GBU/xlsa17/data',
                    help='path to data directory')
parser.add_argument('-r', '--result', metavar='DIR', default='./result',
                    help='path to result directory')
parser.add_argument('-f', '--logfile', metavar='DIR', default=None,
                    help='path to result directory')
parser.add_argument('-ckpt', '--ckpt_dir', metavar='STR', default='./checkpoints/',
                    help='checkpoint file')

parser.add_argument('-clf', '--classifier', metavar='STR', default='KNN', choices=['knn', 'svc'],
                    help='method for classification')

# hyper-parameters
parser.add_argument('-ns', '--num_samples', type=int, metavar='INT', default=500,
                    help='number of samples drawn for each unseen class')
parser.add_argument('-k', '--K', metavar='INT', type=int, default=1,
                    help='number of neighbors in kNN')
parser.add_argument('-c', '--C', metavar='FLOAT', type=float, default=1.0,
                    help='penalty for SVC')

# environment
parser.add_argument('-g', '--gpu', metavar='IDs', default='0',
                    help='what GPUs to use')

args = parser.parse_args()

# if yaml config exists, load and override default ones
if args.config is not None:
    with open(args.config, 'r',encoding="utf-8") as fin:
        options_yaml = yaml.load(fin,Loader=yaml.SafeLoader)
    update_values(options_yaml, vars(args))


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

ts = get_datetime_str()
if args.logfile is None:
    safe_ts = ts.replace(':', '_').replace('-', '_')
    args.logfile = f'log_valtest_{args.data_name}_{safe_ts}.txt'


# 修改数据加载部分
source_data_dir = Path(args.source_data_root)
target_data_dir = Path(args.target_data_root)

source_dir = source_data_dir / Path(args.source_data_name)
target_dir = target_data_dir / Path(args.target_data_name)

source_att_path = source_dir / Path('att_splits.mat')
source_res_path = source_dir / Path('res101.mat')
target_att_path = target_dir / Path('att_splits.mat')
target_res_path = target_dir / Path('res101.mat')

pprint.pprint(vars(args))

result_dir = Path(args.result)
if not result_dir.is_dir():
    result_dir.mkdir(parents=True)

val_acc_file = str(result_dir / Path('val_acc_' + args.data_name + '_' + ts + '.txt'))
logfile = result_dir / Path(args.logfile)
logMaster = Logger(str(logfile))
log_args(str(logfile), args)


def main():
    val_acc = []
    test_acc_history = []  # 新增:记录所有模型在测试集上的表现
    model_epochs = []      # 新增:记录对应的epoch数

    logger = logMaster.get_logger('main')
    logger.info('loading data...')
    
    # 加载源域数据和属性
    source_x, source_y, source_att = load_data(args.source_data_name, args.source_data_root, 'train')
    
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
    target_test_seen_x, target_test_seen_y, _ = load_data(args.target_data_name, args.target_data_root, 'test_seen')
    target_test_unseen_x, target_test_unseen_y, _ = load_data(args.target_data_name, args.target_data_root, 'test_unseen')
    
    # 确保特征维度正确
    args.x_dim = source_x.shape[1]
    
    att_feats = {
        'train': source_att,
        'val': target_att,
        'test_seen': target_test_seen_x,
        'test_unseen': target_test_unseen_x
    }

    ckpt_dir = Path(args.ckpt_dir)
    filenames = ckpt_dir.glob('gdan_*.pkl')

    def cmp_func(s):
        s = str(s).split('.')[0]
        num = int(s.split('_')[-1])
        return num

    filenames = sorted(filenames, key=cmp_func)
    
    # 对每个checkpoint都进行测试集评估
    for checkpoint in filenames:
        epoch = int(str(checkpoint).split('_')[-1].split('.')[0])
        model_epochs.append(epoch)
        
        # 验证集评估
        macc = eval_model_val(checkpoint, logger, att_feats, source_x, target_x, source_y, target_y)
        val_acc.append(macc)
        
        # 测试集评估
        test_macc = eval_model_test(checkpoint, logger, att_feats, source_x, target_test_seen_x, target_test_seen_y, target_test_unseen_x, target_test_unseen_y)
        test_acc_history.append(test_macc)
        
        logger.info(f'Model at epoch {epoch}: val_acc: {macc:.5f}, test_acc: {test_macc:.5f}')

    # 保存完整的评估结果
    evaluation_results = {
        'model_epochs': model_epochs,
        'val_acc': val_acc,
        'test_acc': test_acc_history
    }
    
    # 确保result目录存在
    result_dir = Path(args.result)
    if not result_dir.exists():
        result_dir.mkdir(parents=True)
        
    # 保存结果
    save_path = result_dir / 'full_evaluation_results.pt'
    torch.save(evaluation_results, str(save_path))
    logger.info(f'Evaluation results saved to {save_path}')


def eval_model_val(checkpoint, logger, att_feats, source_x, target_x, source_y, target_y):
    """评估模型在验证集上的性能"""
    # 加载模型
    cvae = CVAE(x_dim=args.x_dim, s_dim=args.source_s_dim, z_dim=args.z_dim,
                enc_layers=args.enc, dec_layers=args.dec)
    discriminator = Discriminator(x_dim=args.x_dim, s_dim=args.target_s_dim, layers=args.dis)
    regressor = Regressor(x_dim=args.x_dim, s_dim=args.target_s_dim, layers=args.reg)
    
    semantic_transform = MultiAgentSemanticTransform(
        source_dim=args.source_s_dim,
        target_dim=args.target_s_dim,
        num_agents=args.num_agents,
        hidden_dims=[512, 256],
        dropout_rate=0.3
    )
    
    # 加载检查点
    states = torch.load(checkpoint)
    cvae.load_state_dict(states['cvae'])
    discriminator.load_state_dict(states['discriminator'])
    regressor.load_state_dict(states['regressor'])
    semantic_transform.load_state_dict(states['semantic_transform'])
    
    cvae.cuda().eval()
    discriminator.cuda().eval()
    regressor.cuda().eval()
    semantic_transform.cuda().eval()
    
    # 创建目标域数据
    target_data = list(zip(target_x, target_y))
    
    # 获取唯一类别
    unique_classes = np.unique(target_y)
    
    # 确保类别索引在有效范围内
    valid_classes = [c for c in unique_classes if c < len(att_feats['train'])]
    
    # 生成样本并评估
    generated_samples = generate_samples(cvae, semantic_transform, args.num_samples, 
                                        att_feats['train'], valid_classes)
    
    # 使用KNN分类器评估
    X_test = np.asarray([item[0] for item in target_data])
    Y_test = np.asarray([item[1] for item in target_data])
    X_gen = np.asarray([item[0] for item in generated_samples])
    Y_gen = np.asarray([item[1] for item in generated_samples])
    
    # 使用生成的样本作为训练集，目标域数据作为测试集
    knn = KNeighborsClassifier(n_neighbors=args.K)
    knn.fit(X_gen, Y_gen)
    Y_pred = knn.predict(X_test)
    
    # 计算准确率
    acc = accuracy_score(Y_test, Y_pred)
    
    return acc


def eval_model_test(checkpoint, logger, att_feats, source_x, target_test_seen_x, target_test_seen_y, target_test_unseen_x, target_test_unseen_y):
    """评估模型在测试集上的性能"""
    # 加载模型
    cvae = CVAE(x_dim=args.x_dim, s_dim=args.source_s_dim, z_dim=args.z_dim,
                enc_layers=args.enc, dec_layers=args.dec)
    discriminator = Discriminator(x_dim=args.x_dim, s_dim=args.target_s_dim, layers=args.dis)
    regressor = Regressor(x_dim=args.x_dim, s_dim=args.target_s_dim, layers=args.reg)
    
    semantic_transform = MultiAgentSemanticTransform(
        target_dim=args.target_s_dim,
        source_dim=args.source_s_dim,
        num_agents=args.num_agents,
        hidden_dims=[512, 256],
        dropout_rate=0.3
    )
    
    # 加载检查点
    states = torch.load(checkpoint)
    cvae.load_state_dict(states['cvae'])
    discriminator.load_state_dict(states['discriminator'])
    regressor.load_state_dict(states['regressor'])
    semantic_transform.load_state_dict(states['semantic_transform'])
    
    cvae.cuda().eval()
    discriminator.cuda().eval()
    regressor.cuda().eval()
    semantic_transform.cuda().eval()
    
    # 创建测试数据
    target_test_seen_data = list(zip(target_test_seen_x, target_test_seen_y))
    target_test_unseen_data = list(zip(target_test_unseen_x, target_test_unseen_y))
    
    # 获取唯一类别
    seen_classes = np.unique(target_test_seen_y)
    unseen_classes = np.unique(target_test_unseen_y)
    
    # 确保类别索引在有效范围内
    valid_seen_classes = [c for c in seen_classes if c < len(att_feats['train'])]
    valid_unseen_classes = [c for c in unseen_classes if c < len(att_feats['train'])]
    
    # 生成样本并评估
    generated_seen_samples = generate_samples(cvae, semantic_transform, args.num_samples, 
                                             att_feats['train'], valid_seen_classes)
    
    generated_unseen_samples = generate_samples(cvae, semantic_transform, args.num_samples, 
                                               att_feats['train'], valid_unseen_classes)
    
    # 使用生成的样本作为训练集，目标域数据作为测试集
    knn = KNeighborsClassifier(n_neighbors=args.K)
    
    # 处理已见类别
    X_gen_seen = np.asarray([item[0] for item in generated_seen_samples])
    Y_gen_seen = np.asarray([item[1] for item in generated_seen_samples])
    X_test_seen = np.asarray([item[0] for item in target_test_seen_data])
    Y_test_seen = np.asarray([item[1] for item in target_test_seen_data])
    
    knn.fit(X_gen_seen, Y_gen_seen)
    Y_pred_seen = knn.predict(X_test_seen)
    
    # 处理未见类别
    X_gen_unseen = np.asarray([item[0] for item in generated_unseen_samples])
    Y_gen_unseen = np.asarray([item[1] for item in generated_unseen_samples])
    X_test_unseen = np.asarray([item[0] for item in target_test_unseen_data])
    Y_test_unseen = np.asarray([item[1] for item in target_test_unseen_data])
    
    knn.fit(X_gen_unseen, Y_gen_unseen)
    Y_pred_unseen = knn.predict(X_test_unseen)
    
    # 计算准确率
    acc_seen = accuracy_score(Y_test_seen, Y_pred_seen)
    acc_unseen = accuracy_score(Y_test_unseen, Y_pred_unseen)
    
    return (acc_seen + acc_unseen) / 2


def generate_samples(net, semantic_transform, num_samples, class_emb, labels):
    """
    生成样本
    
    参数:
    - net: CVAE模型
    - semantic_transform: 语义转换模型
    - num_samples: 每个类别生成的样本数
    - class_emb: 类别嵌入
    - labels: 类别标签
    
    返回:
    - 生成的样本列表，每个元素是(特征, 标签)
    """
    with torch.no_grad():  # 确保在生成样本时不计算梯度
        data = []
        # 确保类别索引在有效范围内
        for i in range(len(labels)):
            if i >= len(class_emb):
                print(f"警告: 类别索引 {i} 超出范围，跳过")
                continue
                
            for _ in range(num_samples):
                feats = Variable(torch.from_numpy(class_emb[i].reshape(1, -1)).float()).cuda()
                # 使用动态语义转换，解包返回值
                feats_transformed, _, _ = semantic_transform(feats)  # 只使用转换后的特征
                sample = net.sample(feats_transformed).cpu().data.numpy().reshape(-1)
                data.append((sample, labels[i]))
        return data


def cal_macc(*, truth, pred):
    assert len(truth) == len(pred)
    count = {}
    total = {}
    labels = list(set(truth))
    for label in labels:
        count[label] = 0
        total[label] = 0

    for y in truth:
        total[y] += 1

    correct = np.nonzero(np.asarray(truth) == np.asarray(pred))[0]

    for c in correct:
        idx = truth[c]
        count[idx] += 1

    macc = 0
    num_class = len(labels)
    for key in count.keys():
        if total[key] == 0:
            num_class -= 1
        else:
            macc += count[key] / total[key]
    macc /= num_class
    return macc


if __name__ == '__main__':
    main()
