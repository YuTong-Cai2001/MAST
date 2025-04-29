import numpy as np
import scipy.io as sio
import os


def fix_indices():
    """修复类别索引问题"""
    print("开始修复类别索引...")

    # 加载当前数据
    res_path = 'data/medical/processed/HAM10000/res101.mat'
    att_path = 'data/medical/processed/HAM10000/att_splits.mat'

    res = sio.loadmat(res_path)
    att = sio.loadmat(att_path)

    # 获取数据
    features = res['features']
    labels = res['labels'].reshape(-1)
    image_files = res.get('image_files', [])
    att_matrix = att['att']
    train_loc = att['train_loc'].reshape(-1)
    test_seen_loc = att['test_seen_loc'].reshape(-1)

    # 获取唯一类别
    unique_classes = np.unique(labels)
    print(f"原始唯一类别: {unique_classes}")

    # 创建类别重映射
    label_map = {old_label: new_label for new_label, old_label in enumerate(unique_classes)}
    print(f"类别映射: {label_map}")

    # 重映射标签
    new_labels = np.array([label_map[label] for label in labels])
    print(f"新唯一类别: {np.unique(new_labels)}")

    # 更新数据
    res['labels'] = new_labels.reshape(labels.shape)

    # 保存修复后的数据
    print("保存修复后的数据...")
    sio.savemat(res_path, res)

    print("类别索引修复完成!")
    return True


if __name__ == "__main__":
    fix_indices()