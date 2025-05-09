import numpy as np
from pathlib import Path
import scipy.io as sio
from sklearn.metrics.pairwise import cosine_distances
import datetime
import os


def load_data(data_name, data_root, split='all'):
    """加载数据集"""
    # 检查是否是医学数据集
    if data_name in ['HAM10000', 'ISIC2019']:
        # 加载医学数据集
        res_path = os.path.join(data_root, data_name, 'res101.mat')
        att_path = os.path.join(data_root, data_name, 'att_splits.mat')
        
        res = sio.loadmat(res_path)
        att = sio.loadmat(att_path)
        
        features = res['features']
        labels = res['labels'].reshape(-1)
        attributes = att['att']
        
        if split == 'train':
            idx = att['train_loc'].reshape(-1)
        elif split == 'val':
            idx = att['val_loc'].reshape(-1)
        elif split == 'test_seen':
            idx = att['test_seen_loc'].reshape(-1)
        elif split == 'test_unseen':
            idx = att['test_unseen_loc'].reshape(-1)
        elif split == 'all':
            idx = np.arange(features.shape[0])
        
        # 确保索引从0开始
        if idx.size > 0 and idx.min() == 1:
            idx = idx - 1
            
        return features[idx], labels[idx], attributes
    else:
        # 原有的数据加载逻辑
        att_feats_dat = sio.loadmat(str(att_path))
        res_feats_dat = sio.loadmat(str(res_path))

        att_feats = att_feats_dat['att'].transpose()
        id_train = att_feats_dat['train_loc'].squeeze() - 1
        id_val = att_feats_dat['val_loc'].squeeze() - 1
        id_test_unseen = att_feats_dat['test_unseen_loc'].squeeze() - 1

        try:
            id_test_seen = att_feats_dat['test_seen_loc'].squeeze() - 1
        except KeyError:
            id_test_seen = None

        num_class = att_feats.shape[0]

        features = res_feats_dat['features'].transpose()
        labels = res_feats_dat['labels'].squeeze().astype(int) - 1

        train_class = np.unique(labels[id_train])
        val_class = np.unique(labels[id_val])
        test_class = np.unique(labels[id_test_unseen])

        if id_test_seen is not None:
            test_class_s = np.unique(labels[id_test_seen])
        else:
            test_class_s = []

        train_x = features[id_train]
        train_y = labels[id_train]
        train_data = list(zip(train_x, train_y))

        val_x = features[id_val]
        val_y = labels[id_val]
        val_data = list(zip(val_x, val_y))

        test_x = features[id_test_unseen]
        test_y = labels[id_test_unseen]
        test_data = list(zip(test_x, test_y))

        if id_test_seen is not None:
            test_s_x = features[id_test_seen]
            test_s_y = labels[id_test_seen]
            test_data_s = list(zip(test_s_x, test_s_y))
        else:
            test_data_s = []

        class_label = {}
        class_label['train'] = list(train_class)
        class_label['val'] = list(val_class)
        class_label['test'] = list(test_class)
        class_label['test_s'] = list(test_class_s)
        class_label['num_class'] = num_class
        return att_feats, train_data, val_data, test_data, test_data_s, class_label


def vectorized_l2(a, b):
    """
    computes the euclidean distance for every row vector in a against all b row vectors
    :param a: shape=[x, y]
    :param b: shape=[z, y]
    :return:  shape=[x, z]
    """
    return np.sqrt((np.square(a[:, np.newaxis] - b).sum(axis=2)))


def kNN_classify(*, x, y):
    """
    return the index of y that is closest to each x
    :param x: n*d matrix
    :param y: m*d matrix
    :return: n-dim vector
    """
    ds = cosine_distances(x, y)
    idx = y[np.argmin(ds, axis=1)]
    return idx


def update_values(dict_from, dict_to):
    for key, value in dict_from.items():
        if isinstance(value, dict):
            update_values(dict_from[key], dict_to[key])
        elif value is not None:
            dict_to[key] = dict_from[key]


def get_datetime_str():
    """返回一个安全的时间戳字符串，避免使用特殊字符"""
    from datetime import datetime
    now = datetime.now()
    return now.strftime('%Y%m%d_%H%M%S')  # 使用下划线替代特殊字符


def get_negative_samples(Y:list, classes):
    Yp = []
    for y in Y:
        yy = y
        while yy == y:
            yy = np.random.choice(classes, 1)
        Yp.append(yy[0])
    return Yp


