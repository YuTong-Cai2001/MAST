import os
import numpy as np
import scipy.io as sio
import json


def inspect_class_indices():
    """检查类别索引"""
    print("开始检查类别索引...")

    # 检查原始数据是否存在
    original_res_path = 'data/medical/processed/HAM10000/res101_original.mat'
    original_att_path = 'data/medical/processed/HAM10000/att_splits_original.mat'

    if os.path.exists(original_res_path) and os.path.exists(original_att_path):
        # 加载原始数据
        res = sio.loadmat(original_res_path)
        labels_original = res['labels'].reshape(-1)

        # 打印原始数据信息
        print("\n原始数据信息:")
        print(f"样本数量: {labels_original.shape[0]}")
        unique_original = np.unique(labels_original)
        print(f"唯一类别: {unique_original}")

        # 打印类别分布
        print("\n原始数据类别分布:")
        for i in sorted(unique_original):
            print(f"类别 {i}: {np.sum(labels_original == i)}")
    else:
        print("未找到原始备份数据文件")

    # 加载当前数据
    res_path = 'data/medical/processed/HAM10000/res101.mat'
    att_path = 'data/medical/processed/HAM10000/att_splits.mat'

    if not os.path.exists(res_path) or not os.path.exists(att_path):
        print("未找到当前数据文件")
        return

    res = sio.loadmat(res_path)
    att = sio.loadmat(att_path)

    labels_new = res['labels'].reshape(-1)
    att_matrix = att['att']

    # 打印当前数据信息
    print("\n当前数据信息:")
    print(f"样本数量: {labels_new.shape[0]}")
    unique_new = np.unique(labels_new)
    print(f"唯一类别: {unique_new}")
    print(f"语义属性矩阵形状: {att_matrix.shape}")

    # 打印类别分布
    print("\n当前数据类别分布:")
    for i in sorted(unique_new):
        print(f"类别 {i}: {np.sum(labels_new == i)}")

    # 检查是否有类别索引超出语义属性矩阵范围
    if np.max(unique_new) >= att_matrix.shape[0]:
        print(f"\n警告: 最大类别索引 {np.max(unique_new)} 超出语义属性矩阵大小 {att_matrix.shape[0]}")
        print("这可能导致索引错误!")
    else:
        print("\n类别索引在语义属性矩阵范围内，不会导致索引错误")

    # 检查类别是否连续
    expected_classes = set(range(len(unique_new)))
    actual_classes = set(unique_new)

    if expected_classes == actual_classes:
        print("类别索引是连续的")
    else:
        print("类别索引不连续")
        print(f"缺失的类别: {expected_classes - actual_classes}")
        print(f"多余的类别: {actual_classes - expected_classes}")

    # 加载语义属性文件
    try:
        with open('data/medical/semantic_attributes/ham10000_attributes.json', 'r') as f:
            semantic_attributes = json.load(f)

        print("\n语义属性文件中的类别:")
        for cls in semantic_attributes.keys():
            print(f"- {cls}")
    except Exception as e:
        print(f"无法加载语义属性文件: {e}")

    # 建议解决方案
    print("\n建议解决方案:")
    if np.max(unique_new) >= att_matrix.shape[0]:
        print("1. 修改pretrain_gdan.py，添加类别重映射逻辑")
        print("2. 重新运行process_ham10000.py，确保生成连续的类别索引")
        print("3. 手动调整att_splits.mat文件中的语义属性矩阵大小")
    else:
        print("当前数据设置正常，不需要额外修改")


if __name__ == "__main__":
    inspect_class_indices()