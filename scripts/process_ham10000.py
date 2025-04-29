import os
import pandas as pd
import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm
import scipy.io as sio
import json
import shutil
import glob
from sklearn.model_selection import train_test_split

# 创建目录结构
os.makedirs('data/medical/raw/ham10000/images', exist_ok=True)
os.makedirs('data/medical/processed/HAM10000', exist_ok=True)
os.makedirs('data/medical/semantic_attributes', exist_ok=True)

# 设置随机种子以确保结果可重现
np.random.seed(42)

# 定义要排除的罕见疾病类别（这些将作为目标域的未见类别）
EXCLUDE_FROM_SOURCE = ['df', 'bkl', 'vasc']  # 皮肤纤维瘤(DF)、良性角化病(BKL)、血管病变(VASC)

def process_ham10000_without_rare_diseases():
    """处理HAM10000数据集，排除指定的罕见疾病类别"""
    # 使用原始代码中的路径
    metadata_path = 'data/medical/raw/ham10000/metadata.tab'
    output_dir = 'data/medical/processed/HAM10000'
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取元数据
    print(f"读取元数据: {metadata_path}")
    metadata = pd.read_csv(metadata_path, sep='\t')
    
    # 提取特征（使用您原始代码中的函数）
    features, labels, image_files, metadata, unique_classes, class_to_idx = extract_features()
    
    # 创建类别映射
    print("创建类别映射...")
    
    # 找出要排除的类别索引
    exclude_indices = [class_to_idx[cls] for cls in EXCLUDE_FROM_SOURCE if cls in class_to_idx]
    print(f"排除的类别索引: {exclude_indices}")
    print(f"类别映射: {class_to_idx}")
    
    # 分离要保留的数据和要排除的数据
    keep_indices = [i for i, label in enumerate(labels) if label not in exclude_indices]
    exclude_indices_actual = [i for i, label in enumerate(labels) if label in exclude_indices]
    
    keep_features = features[keep_indices]
    keep_labels = labels[keep_indices]
    keep_image_files = [image_files[i] for i in keep_indices]
    
    print(f"保留的样本数: {len(keep_indices)}")
    print(f"排除的样本数: {len(exclude_indices_actual)}")
    
    # 从保留的数据中划分训练集和测试集
    indices = np.arange(len(keep_labels))
    np.random.shuffle(indices)
    
    train_size = int(0.8 * len(indices))
    
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    train_features = keep_features[train_indices]
    train_labels = keep_labels[train_indices]
    train_image_files = [keep_image_files[i] for i in train_indices]
    
    test_features = keep_features[test_indices]
    test_labels = keep_labels[test_indices]
    test_image_files = [keep_image_files[i] for i in test_indices]
    
    # 加载语义属性
    attributes_file = 'data/medical/semantic_attributes/ham10000_attributes.json'
    with open(attributes_file, 'r') as f:
        semantic_attributes = json.load(f)
    
    attribute_names = list(next(iter(semantic_attributes.values())).keys())
    
    # 创建属性矩阵（排除罕见疾病后的）
    valid_classes = [cls for cls in unique_classes if cls not in EXCLUDE_FROM_SOURCE]
    num_classes = len(valid_classes)
    att_matrix = np.zeros((num_classes, len(attribute_names)))
    
    for i, cls in enumerate(valid_classes):
        for j, attr in enumerate(attribute_names):
            att_matrix[i, j] = semantic_attributes[cls].get(attr, 0)
    
    # 创建res101.mat（只包含非罕见疾病）
    res101 = {
        'features': keep_features,
        'labels': keep_labels,
        'image_files': keep_image_files
    }
    
    # 创建att_splits.mat（只包含非罕见疾病）
    att_splits = {
        'att': att_matrix,
        'train_loc': train_indices,
        'val_loc': np.array([]),  # 这里可以根据需要调整
        'test_loc': test_indices,
        'test_seen_loc': test_indices,
        'test_unseen_loc': np.array([]),
        'attribute_names': attribute_names
    }
    
    # 保存为.mat文件
    sio.savemat(os.path.join(output_dir, 'res101_no_rare.mat'), res101)
    sio.savemat(os.path.join(output_dir, 'att_splits_no_rare.mat'), att_splits)
    
    # 保存为numpy文件（与您的新函数兼容）
    np.save(os.path.join('data/medical/processed', 'HAM10000_train_features.npy'), train_features)
    np.save(os.path.join('data/medical/processed', 'HAM10000_train_labels.npy'), train_labels)
    np.save(os.path.join('data/medical/processed', 'HAM10000_test_seen_features.npy'), test_features)
    np.save(os.path.join('data/medical/processed', 'HAM10000_test_seen_labels.npy'), test_labels)
    
    # 保存类别映射（排除罕见疾病后的）
    filtered_class_to_idx = {cls: idx for idx, cls in enumerate(valid_classes)}
    with open(os.path.join('data/medical/processed', 'HAM10000_class_mapping.json'), 'w') as f:
        json.dump(filtered_class_to_idx, f, indent=2)
    
    print("HAM10000数据集处理完成（已排除罕见疾病）")
    
    # 返回处理后的数据，以便后续使用
    return {
        'train_features': train_features,
        'train_labels': train_labels,
        'test_features': test_features,
        'test_labels': test_labels,
        'att_matrix': att_matrix,
        'class_mapping': filtered_class_to_idx
    }

# 检查数据是否已存在
def check_data_exists():
    # 检查图像目录是否存在 - 修改为实际路径
    base_dir = 'data/medical/raw/ham10000/images'
    part1_dir = os.path.join(base_dir, 'HAM10000_images_part_1')
    part2_dir = os.path.join(base_dir, 'HAM10000_images_part_2')
    metadata_file = 'data/medical/raw/ham10000/metadata.tab'
    
    if not os.path.exists(part1_dir) and not os.path.exists(part2_dir):
        print(f"图像目录 {part1_dir} 或 {part2_dir} 不存在")
        return False
    
    if not os.path.exists(metadata_file):
        print(f"元数据文件 {metadata_file} 不存在")
        return False
    
    # 计算两个目录中的图像文件数量
    part1_count = len(glob.glob(os.path.join(part1_dir, '*.jpg'))) if os.path.exists(part1_dir) else 0
    part2_count = len(glob.glob(os.path.join(part2_dir, '*.jpg'))) if os.path.exists(part2_dir) else 0
    
    print(f"数据已存在:")
    print(f"- {part1_dir}: {part1_count}个图像文件")
    print(f"- {part2_dir}: {part2_count}个图像文件")
    print(f"- 元数据文件: {metadata_file}")
    
    return True

# 提取图像特征
def extract_features():
    # 读取元数据
    metadata_path = 'data/medical/raw/ham10000/metadata.tab'
    metadata = pd.read_csv(metadata_path, sep='\t')
    
    # 查看类别分布
    print("疾病类别分布:")
    class_distribution = metadata['dx'].value_counts()
    print(class_distribution)
    
    # 查看每个类别的全名
    print("疾病类别全名:")
    class_names = metadata[['dx', 'dx_type']].drop_duplicates()
    print(class_names)
    
    # 加载预训练的ResNet101
    model = models.resnet101(weights='IMAGENET1K_V1')
    # 移除最后的全连接层
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    
    # 定义图像转换
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # 为所有图像提取特征
    features = []
    labels = []
    image_files = []
    
    # 修改图像目录路径
    base_dir = 'data/medical/raw/ham10000/images'
    part1_dir = os.path.join(base_dir, 'HAM10000_images_part_1')
    part2_dir = os.path.join(base_dir, 'HAM10000_images_part_2')
    
    # 获取唯一类别并创建映射
    unique_classes = metadata['dx'].unique()
    class_to_idx = {cls: i for i, cls in enumerate(unique_classes)}
    
    # 获取所有图像文件路径
    all_image_files = []
    if os.path.exists(part1_dir):
        all_image_files.extend(glob.glob(os.path.join(part1_dir, '*.jpg')))
    if os.path.exists(part2_dir):
        all_image_files.extend(glob.glob(os.path.join(part2_dir, '*.jpg')))
    
    print(f"找到 {len(all_image_files)} 个jpg文件")
    
    # 创建图像ID到文件路径的映射
    image_id_to_path = {}
    for img_path in all_image_files:
        img_name = os.path.basename(img_path)
        # 处理ISIC_XXXXX.jpg格式
        if img_name.startswith('ISIC_') and img_name.endswith('.jpg'):
            img_id = img_name[5:-4]  # 移除ISIC_前缀和.jpg后缀
            image_id_to_path[img_id] = img_path
    
    print(f"创建了 {len(image_id_to_path)} 个图像ID到路径的映射")
    
    # 检查前几个图像ID是否在映射中
    sample_ids = metadata['image_id'].iloc[:5].tolist()
    print(f"元数据中的前5个图像ID: {sample_ids}")
    for img_id in sample_ids:
        if img_id in image_id_to_path:
            print(f"图像ID {img_id} 在映射中，对应路径: {image_id_to_path[img_id]}")
        else:
            print(f"图像ID {img_id} 不在映射中")
    
    print(f"开始提取特征，共{len(metadata)}张图像...")
    for index, row in tqdm(metadata.iterrows(), total=len(metadata)):
        image_id = row['image_id']
        
        # 尝试直接使用image_id查找图像
        if image_id in image_id_to_path:
            image_path = image_id_to_path[image_id]
        else:
            # 尝试不同的文件名格式和路径
            potential_paths = [
                os.path.join(part1_dir, f"ISIC_{image_id}.jpg"),
                os.path.join(part2_dir, f"ISIC_{image_id}.jpg"),
                os.path.join(part1_dir, f"{image_id}.jpg"),
                os.path.join(part2_dir, f"{image_id}.jpg")
            ]
            
            image_path = None
            for path in potential_paths:
                if os.path.exists(path):
                    image_path = path
                    break
        
        if image_path and os.path.exists(image_path):
            try:
                img = Image.open(image_path).convert('RGB')
                img_tensor = preprocess(img)
                img_tensor = img_tensor.unsqueeze(0)
                if torch.cuda.is_available():
                    img_tensor = img_tensor.cuda()
                
                with torch.no_grad():
                    feature = model(img_tensor)
                
                features.append(feature.cpu().numpy().reshape(-1))
                labels.append(class_to_idx[row['dx']])
                image_files.append(image_id)
            except Exception as e:
                print(f"处理图像{image_path}时出错: {e}")
        else:
            if index < 10:  # 只打印前10个未找到的图像，避免输出过多
                print(f"未找到图像: {image_id}")
    
    # 转换为numpy数组
    features = np.array(features)
    labels = np.array(labels)
    
    print(f"特征提取完成，共处理{len(features)}张图像")
    print(f"特征维度: {features.shape}")
    
    return features, labels, image_files, metadata, unique_classes, class_to_idx

# 创建语义属性
def create_semantic_attributes(metadata, unique_classes):
    # 这里我们假设已经通过ChatGPT生成了语义属性，并保存在一个JSON文件中
    # 如果没有，您需要手动创建这个文件
    
    attributes_file = 'data/medical/semantic_attributes/ham10000_attributes.json'
    
    if not os.path.exists(attributes_file):
        print("语义属性文件不存在，请先生成语义属性")
        print("您可以使用ChatGPT为每个类别生成64个二元语义属性，并保存为JSON格式")
        print("格式示例: {\"akiec\": {\"attr1\": 1, \"attr2\": 0, ...}, ...}")
        return None
    
    with open(attributes_file, 'r') as f:
        semantic_attributes = json.load(f)
    
    # 验证属性
    for cls in unique_classes:
        if cls not in semantic_attributes:
            print(f"警告: 类别{cls}在语义属性中不存在")
            return None
    
    attribute_names = list(next(iter(semantic_attributes.values())).keys())
    print(f"加载了{len(attribute_names)}个语义属性")
    
    return semantic_attributes, attribute_names

# 创建.mat文件
def create_mat_files(features, labels, image_files, semantic_attributes, attribute_names, unique_classes):
    # 创建属性矩阵
    num_classes = len(unique_classes)
    att_matrix = np.zeros((num_classes, len(attribute_names)))
    
    for i, cls in enumerate(unique_classes):
        for j, attr in enumerate(attribute_names):
            att_matrix[i, j] = semantic_attributes[cls].get(attr, 0)
    
    # 划分训练/验证/测试集
    np.random.seed(42)
    indices = np.arange(len(labels))
    np.random.shuffle(indices)
    
    train_size = int(0.8 * len(indices))
    val_size = int(0.1 * len(indices))
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size+val_size]
    test_indices = indices[train_size+val_size:]
    
    # 创建res101.mat
    res101 = {
        'features': features,
        'labels': labels,
        'image_files': image_files
    }
    
    # 创建att_splits.mat
    att_splits = {
        'att': att_matrix,
        'train_loc': train_indices,
        'val_loc': val_indices,
        'test_loc': test_indices,
        'test_seen_loc': test_indices,
        'test_unseen_loc': np.array([]),
        'attribute_names': attribute_names
    }
    
    # 保存为.mat文件
    sio.savemat('data/medical/processed/HAM10000/res101.mat', res101)
    sio.savemat('data/medical/processed/HAM10000/att_splits.mat', att_splits)
    
    print("数据处理完成，已保存为.mat文件")

def main():
    # 检查数据是否已存在
    if not check_data_exists():
        print("请确保数据已正确放置")
        return
    
    # 处理HAM10000数据集，排除罕见疾病
    process_ham10000_without_rare_diseases()
    
    # 如果您仍然需要原始的处理方式，可以取消下面的注释
    # 提取特征
    # features, labels, image_files, metadata, unique_classes, class_to_idx = extract_features()
    
    # 创建或加载语义属性
    # result = create_semantic_attributes(metadata, unique_classes)
    # if result is None:
    #     return
    # semantic_attributes, attribute_names = result
    
    # 创建.mat文件
    # create_mat_files(features, labels, image_files, semantic_attributes, attribute_names, unique_classes)

if __name__ == "__main__":
    main() 