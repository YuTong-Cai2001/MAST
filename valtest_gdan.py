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
    test_acc_history = []
    model_epochs = []

    logger = logMaster.get_logger('main')
    logger.info('loading data...')
    att_feats, train_data, val_data, test_data, test_s_data, classes = load_data(
        att_path=target_att_path,  # 这里会使用CUB的数据路径
        res_path=target_res_path
    )

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
        macc = eval_model_val(checkpoint, logger, att_feats, train_data, val_data, classes)
        val_acc.append(macc)
        
        # 测试集评估
        test_macc = eval_model_test(checkpoint, logger, att_feats, train_data, test_data, classes)
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


def eval_model_val(checkpoint, logger, att_feats, train_data, val_data, classes):
    logger.info('building model...')

    states = torch.load(checkpoint)

    # 使用源域的维度初始化CVAE
    net = CVAE(x_dim=states['x_dim'], 
               s_dim=states['source_s_dim'],
               z_dim=states['z_dim'], 
               enc_layers=states['enc_layers'],
               dec_layers=states['dec_layers'])
               
    # 使用目标域的维度初始化判别器和回归器
    discriminator = Discriminator(x_dim=states['x_dim'], 
                                s_dim=states['target_s_dim'],
                                layers=states['dis_layers'])
    regressor = Regressor(x_dim=states['x_dim'], 
                         s_dim=states['target_s_dim'],
                         layers=states['reg_layers'])
    
    # 使用新的语义转换网络
    semantic_transform = MultiAgentSemanticTransform(
        target_dim=states['target_s_dim'],  # 这里会自动使用CUB的312维
        source_dim=states['source_s_dim'],
        num_agents=3,
        hidden_dims=[512, 256],
        dropout_rate=0.3
    ).cuda()

    net.cuda()
    discriminator.cuda()
    regressor.cuda()

    # 设置所有模型为评估模式
    net.eval()
    discriminator.eval()
    regressor.eval()
    semantic_transform.eval()

    logger.info(f'loading model from checkpoint: {checkpoint}')

    net.load_state_dict(states['cvae'])
    discriminator.load_state_dict(states['discriminator'])
    regressor.load_state_dict(states['regressor'])
    semantic_transform.load_state_dict(states['semantic_transform'])

    # 使用torch.no_grad()包装评估过程
    with torch.no_grad():
        logger.info('generating synthetic samples...')
        samples = generate_samples(net, semantic_transform, args.num_samples, att_feats[classes['val']], classes['val'])

        new_train_data = train_data + samples
        X, Y = zip(*new_train_data)
        X = np.array(X)
        Y = np.array(Y)

        if args.classifier == 'svc':
            clf = LinearSVC(C=args.C)
            logger.info('training linear SVC...')
        else:
            clf = KNeighborsClassifier(n_neighbors=args.K)
            logger.info('training kNN classifier')

        clf.fit(X=X, y=Y)

        test_X, test_Y = zip(*val_data)

        logger.info('predicting...')

        pred_Y = clf.predict(test_X)
        macc_u = cal_macc(truth=test_Y, pred=pred_Y)

        logger.info(f'gzsl macc_u: {macc_u:.5f}')
        return macc_u


def eval_model_test(checkpoint, logger, att_feats, train_data, test_data, classes):
    logger.info('building model...')

    states = torch.load(checkpoint)

    # 使用源域的维度初始化CVAE
    net = CVAE(x_dim=states['x_dim'], 
               s_dim=states['source_s_dim'],  # 使用源域的语义维度
               z_dim=states['z_dim'], 
               enc_layers=states['enc_layers'],
               dec_layers=states['dec_layers'])
               
    # 使用目标域的维度初始化判别器和回归器
    discriminator = Discriminator(x_dim=states['x_dim'], 
                                s_dim=states['target_s_dim'],  # 使用目标域的语义维度
                                layers=states['dis_layers'])
    regressor = Regressor(x_dim=states['x_dim'], 
                         s_dim=states['target_s_dim'],  # 使用目标域的语义维度
                         layers=states['reg_layers'])
    
    # 使用新的语义转换网络
    semantic_transform = MultiAgentSemanticTransform(
        target_dim=states['target_s_dim'],  # 这里会自动使用CUB的312维
        source_dim=states['source_s_dim'],
        num_agents=3,
        hidden_dims=[512, 256],
        dropout_rate=0.3
    ).cuda()

    net.cuda()
    discriminator.cuda()
    regressor.cuda()

    # 设置所有模型为评估模式
    net.eval()
    discriminator.eval()
    regressor.eval()
    semantic_transform.eval()

    logger.info(f'loading model from checkpoint: {checkpoint}')

    net.load_state_dict(states['cvae'])
    discriminator.load_state_dict(states['discriminator'])
    regressor.load_state_dict(states['regressor'])
    semantic_transform.load_state_dict(states['semantic_transform'])

    # 使用torch.no_grad()包装评估过程
    with torch.no_grad():
        logger.info('generating synthetic samples...')
        samples = generate_samples(net, semantic_transform, args.num_samples, att_feats[classes['test']], classes['test'])

        new_train_data = train_data + samples
        X, Y = zip(*new_train_data)
        X = np.array(X)
        Y = np.array(Y)

        if args.classifier == 'svc':
            clf = LinearSVC(C=args.C)
            logger.info('training linear SVC...')
        else:
            clf = KNeighborsClassifier(n_neighbors=args.K)
            logger.info('training kNN classifier')

        clf.fit(X=X, y=Y)

        test_X, test_Y = zip(*test_data)

        logger.info('predicting...')

        pred_Y = clf.predict(test_X)
        macc_u = cal_macc(truth=test_Y, pred=pred_Y)
        logger.info(f'gzsl unseen: {macc_u:.5f}\n')
        return macc_u


def generate_samples(net, semantic_transform, num_samples, class_emb, labels):
    with torch.no_grad():  # 确保在生成样本时不计算梯度
        class_emb = list(class_emb)
        data = []
        for i in range(len(class_emb)):
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
