import json
import pandas as pd
import os

# 确保目录存在
os.makedirs('data/medical/semantic_attributes', exist_ok=True)

def generate_attributes_manually():
    # 读取元数据
    metadata_path = 'data/medical/raw/ham10000/metadata.tab'
    if not os.path.exists(metadata_path):
        print(f"元数据文件{metadata_path}不存在")
        return
    
    metadata = pd.read_csv(metadata_path, sep='\t')
    
    # 获取唯一的类别
    unique_classes = metadata[['dx', 'dx_type']].drop_duplicates().reset_index(drop=True)
    
    print("HAM10000数据集包含以下类别:")
    for index, row in unique_classes.iterrows():
        print(f"{index+1}. {row['dx']} - {row['dx_type']}")
    
    print("\n请为每个类别生成64个二元语义属性(0-1值)")
    print("您可以使用ChatGPT，提示如下:")
    print("---")
    print("为皮肤病\"[类别名称]\"(代码:[类别代码])生成64个二元语义属性(0-1值)。")
    print("属性应包括:")
    print("1. 颜色特征(如红色、棕色、黑色等)")
    print("2. 形态特征(如对称性、边界规则性、隆起等)")
    print("3. 纹理特征(如粗糙、光滑、鳞片状等)")
    print("4. 分布特征(如局部、弥漫、多发等)")
    print("5. 临床特征(如瘙痒、疼痛、生长速度等)")
    print("6. 组织学特征(如角化过度、色素沉着等)")
    print("7. 风险因素(如年龄相关、阳光暴露等)")
    print("8. 预后特征(如恶性潜能、复发倾向等)")
    print("")
    print("请以JSON格式返回，格式为:")
    print("{")
    print("    \"attribute_name_1\": 1,")
    print("    \"attribute_name_2\": 0,")
    print("    ...")
    print("}")
    print("")
    print("确保生成正好64个属性，其中1表示该疾病通常具有该特征，0表示通常不具有该特征。")
    print("---")
    
    print("\n获取所有类别的属性后，将它们合并为一个JSON文件，格式如下:")
    print("{")
    print("    \"akiec\": {\"attr1\": 1, \"attr2\": 0, ...},")
    print("    \"bcc\": {\"attr1\": 0, \"attr2\": 1, ...},")
    print("    ...")
    print("}")
    
    print("\n然后将此JSON保存为: data/medical/semantic_attributes/ham10000_attributes.json")

if __name__ == "__main__":
    generate_attributes_manually() 