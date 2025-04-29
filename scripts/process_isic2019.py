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
import zipfile

# 创建目录结构
os.makedirs('data/medical/raw/isic2019/images', exist_ok=True)
os.makedirs('data/medical/processed/ISIC2019', exist_ok=True)
os.makedirs('data/medical/semantic_attributes', exist_ok=True)

# 解压缩图像文件
def extract_zip_file():
    zip_path = 'ISIC_2019_Training_Input.zip'
    extract_dir = 'data/medical/raw/isic2019/images'
    
    if not os.path.exists(zip_path):
        print(f"ZIP文件 {zip_path} 不存在")
        return False
    
    # 检查是否已经解压
    if os.path.exists(extract_dir) and len(os.listdir(extract_dir)) > 0:
        print(f"图像目录 {extract_dir} 已存在，跳过解压步骤")
        return True
    
    print(f"正在解压 {zip_path} 到 {extract_dir}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    
    print("解压完成")
    return True

# 检查数据是否已存在
def check_data_exists():
    # 检查图像目录和元数据文件
    image_dir = 'data/medical/raw/isic2019/images'
    metadata_file = 'ISIC_2019_Training_Metadata.csv'
    groundtruth_file = 'ISIC_2019_Training_GroundTruth.csv'
    
    if not os.path.exists(image_dir) or len(os.listdir(image_dir)) == 0:
        print(f"图像目录 {image_dir} 不存在或为空")
        return False
    
    if not os.path.exists(metadata_file):
        print(f"元数据文件 {metadata_file} 不存在")
        return False
    
    if not os.path.exists(groundtruth_file):
        print(f"标签文件 {groundtruth_file} 不存在")
        return False
    
    # 计算图像文件数量
    image_count = len(glob.glob(os.path.join(image_dir, '**/*.jpg'), recursive=True))
    
    print(f"数据已存在:")
    print(f"- 图像目录: {image_dir} ({image_count}个图像文件)")
    print(f"- 元数据文件: {metadata_file}")
    print(f"- 标签文件: {groundtruth_file}")
    
    return True

# 创建语义属性
def create_semantic_attributes():
    # 读取HAM10000的语义属性作为模板
    ham_attributes_file = 'data/medical/semantic_attributes/ham10000_attributes.json'
    isic_attributes_file = 'data/medical/semantic_attributes/isic2019_attributes.json'
    
    if not os.path.exists(ham_attributes_file):
        print(f"HAM10000语义属性文件 {ham_attributes_file} 不存在")
        return None
    
    # 如果ISIC2019属性文件已存在，直接加载
    if os.path.exists(isic_attributes_file):
        with open(isic_attributes_file, 'r') as f:
            semantic_attributes = json.load(f)
        
        attribute_names = list(next(iter(semantic_attributes.values())).keys())
        print(f"加载了{len(attribute_names)}个语义属性")
        return semantic_attributes, attribute_names
    
    # 读取标签文件，确定类别
    groundtruth_file = 'ISIC_2019_Training_GroundTruth.csv'
    groundtruth = pd.read_csv(groundtruth_file)
    
    # 获取类别列名（除了image列）
    class_names = [col for col in groundtruth.columns if col != 'image']
    
    # 读取HAM10000属性作为模板
    with open(ham_attributes_file, 'r') as f:
        ham_attributes = json.load(f)
    
    # 获取属性名称
    attribute_names = list(next(iter(ham_attributes.values())).keys())
    
    # 创建ISIC2019的语义属性
    # 这里我们需要手动映射ISIC2019的类别到HAM10000的类别
    # 或者为每个类别创建新的属性
    
    # ISIC2019类别到HAM10000类别的映射
    # 这需要根据实际情况调整
    class_mapping = {
        'MEL': 'mel',  # 黑色素瘤
        'NV': 'nv',    # 黑色素细胞痣
        'BCC': 'bcc',  # 基底细胞癌
        'AK': 'akiec', # 光化性角化病
        'BKL': 'bkl',  # 良性角化病
        'DF': 'df',    # 皮肤纤维瘤
        'VASC': 'vasc', # 血管病变
        'SCC': 'akiec', # 鳞状细胞癌 - 映射到光化性角化病
        'UNK': 'nv'     # 未知 - 映射到最常见的类别
    }
    
    # 创建ISIC2019的语义属性
    semantic_attributes = {}
    for isic_class in class_names:
        if isic_class in class_mapping and class_mapping[isic_class] in ham_attributes:
            # 使用映射的HAM10000类别的属性
            semantic_attributes[isic_class] = ham_attributes[class_mapping[isic_class]]
        else:
            # 如果没有映射，使用默认属性（全0）
            semantic_attributes[isic_class] = {attr: 0 for attr in attribute_names}
            print(f"警告: 类别{isic_class}没有映射到HAM10000类别，使用默认属性")
    
    # 保存ISIC2019的语义属性
    with open(isic_attributes_file, 'w') as f:
        json.dump(semantic_attributes, f, indent=2)
    
    print(f"创建了ISIC2019的语义属性，保存到{isic_attributes_file}")
    return semantic_attributes, attribute_names

# 提取图像特征
def extract_features():
    # 读取元数据和标签
    metadata_file = 'ISIC_2019_Training_Metadata.csv'
    groundtruth_file = 'ISIC_2019_Training_GroundTruth.csv'
    
    metadata = pd.read_csv(metadata_file)
    groundtruth = pd.read_csv(groundtruth_file)
    
    # 合并元数据和标签
    data = pd.merge(metadata, groundtruth, on='image')
    
    # 获取类别列名（除了image和元数据列）
    class_columns = [col for col in groundtruth.columns if col != 'image']
    
    # 为每个图像分配一个类别标签（选择概率最高的类别）
    data['label'] = data[class_columns].idxmax(axis=1)
    
    # 查看类别分布
    print("疾病类别分布:")
    class_distribution = data['label'].value_counts()
    print(class_distribution)
    
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
    
    # 图像目录
    image_dir = 'data/medical/raw/isic2019/images'
    
    # 获取唯一类别并创建映射
    unique_classes = data['label'].unique()
    class_to_idx = {cls: i for i, cls in enumerate(unique_classes)}
    
    print(f"开始提取特征，共{len(data)}张图像...")
    for index, row in tqdm(data.iterrows(), total=len(data)):
        image_id = row['image']
        
        # 尝试不同的文件路径格式
        potential_paths = [
            os.path.join(image_dir, f"{image_id}.jpg"),
            os.path.join(image_dir, f"ISIC_{image_id}.jpg"),
            os.path.join(image_dir, f"ISIC_2019_Training_Input", f"{image_id}.jpg"),
            os.path.join(image_dir, f"ISIC_2019_Training_Input", f"ISIC_{image_id}.jpg")
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
                labels.append(class_to_idx[row['label']])
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
    
    return features, labels, image_files, unique_classes, class_to_idx

# 创建.mat文件
def create_mat_files(features, labels, image_files, semantic_attributes, attribute_names, unique_classes, class_to_idx):
    # 创建属性矩阵
    num_classes = len(unique_classes)
    att_matrix = np.zeros((num_classes, len(attribute_names)))
    
    for i, cls in enumerate(unique_classes):
        for j, attr in enumerate(attribute_names):
            att_matrix[i, j] = semantic_attributes[cls].get(attr, 0)
    
    # 获取每个类别的样本数量
    class_counts = {}
    for cls in unique_classes:
        class_counts[cls] = np.sum(labels == class_to_idx[cls])

    print("各类别样本数量:")
    for cls, count in sorted(class_counts.items(), key=lambda x: x[1]):
        print(f"{cls}: {count}")

    # 基于医学知识的罕见疾病列表（从最罕见到较罕见）
    # 根据ChatGPT的医学建议排序
    medical_rare_diseases = ['DF', 'BKL', 'VASC', 'AK']  # 最罕见的四种皮肤病

    # 选择前3种最罕见的疾病（如果在数据集中存在）
    num_rare_classes = 3
    rare_diseases = [cls for cls in medical_rare_diseases if cls in class_to_idx][:num_rare_classes]

    # 如果没有足够的罕见疾病，自动补充样本数量最少的类别
    if len(rare_diseases) < num_rare_classes:
        additional_rare = [cls for cls in sorted(class_counts.keys(), key=lambda x: class_counts[x]) 
                          if cls not in rare_diseases][:num_rare_classes-len(rare_diseases)]
        rare_diseases.extend(additional_rare)

    print(f"选择的罕见疾病类别: {rare_diseases}")
    
    # 获取罕见疾病的索引
    rare_indices = [class_to_idx[cls] for cls in rare_diseases if cls in class_to_idx]
    
    print(f"对应的类别索引: {rare_indices}")
    
    # 划分数据集
    np.random.seed(42)
    
    # 分离罕见疾病样本和常见疾病样本
    unseen_indices = np.where(np.isin(labels, rare_indices))[0]
    seen_indices = np.where(~np.isin(labels, rare_indices))[0]
    
    print(f"常见疾病样本数: {len(seen_indices)}")
    print(f"罕见疾病样本数: {len(unseen_indices)}")
    
    # 打乱顺序
    np.random.shuffle(seen_indices)
    np.random.shuffle(unseen_indices)
    
    # 划分常见疾病样本为训练/验证/测试集
    train_size = int(0.8 * len(seen_indices))
    val_size = int(0.1 * len(seen_indices))
    
    train_indices = seen_indices[:train_size]
    val_indices = seen_indices[train_size:train_size+val_size]
    test_seen_indices = seen_indices[train_size+val_size:]
    
    # 罕见疾病样本全部作为未见测试集
    test_unseen_indices = unseen_indices
    
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
        'test_loc': np.concatenate((test_seen_indices, test_unseen_indices)),
        'test_seen_loc': test_seen_indices,
        'test_unseen_loc': test_unseen_indices,
        'attribute_names': attribute_names
    }
    
    # 保存为.mat文件
    sio.savemat('data/medical/processed/ISIC2019/res101.mat', res101)
    sio.savemat('data/medical/processed/ISIC2019/att_splits.mat', att_splits)
    
    print("数据处理完成，已保存为.mat文件")
    print(f"训练集大小: {len(train_indices)}")
    print(f"验证集大小: {len(val_indices)}")
    print(f"测试集(已见类别)大小: {len(test_seen_indices)}")
    print(f"测试集(未见类别)大小: {len(test_unseen_indices)}")

def main():
    # 解压缩图像文件
    if not extract_zip_file():
        return
    
    # 检查数据是否已存在
    if not check_data_exists():
        print("请确保数据已正确放置")
        return
    
    # 创建语义属性
    result = create_semantic_attributes()
    if result is None:
        return
    semantic_attributes, attribute_names = result
    
    # 提取特征
    features, labels, image_files, unique_classes, class_to_idx = extract_features()
    
    # 创建.mat文件
    create_mat_files(features, labels, image_files, semantic_attributes, attribute_names, unique_classes, class_to_idx)

if __name__ == "__main__":
    main() 