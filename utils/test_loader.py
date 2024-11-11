import os
import shutil
from sklearn.model_selection import train_test_split


# 假设你的数据集文件夹结构如下：
# dataset/
#   ├── class1/
#   ├── class2/
#   └── ...

def split_dataset(data_dir, output_dir, test_size=0.2):
    """
    划分数据集，并将训练集和测试集保存到不同的文件夹中。

    :param data_dir: 原始数据集文件夹
    :param output_dir: 划分后的数据集保存路径
    :param test_size: 测试集所占比例
    """
    # 获取所有类别的文件夹
    classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

    for cls in classes:
        class_dir = os.path.join(data_dir, cls)
        images = [f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))]

        # 划分数据集
        train_files, test_files = train_test_split(images, test_size=test_size, random_state=42)

        # 创建训练集和测试集文件夹
        train_dir = os.path.join(output_dir, 'train', cls)
        test_dir = os.path.join(output_dir, 'test', cls)
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        # 移动文件到相应的文件夹
        for file in train_files:
            shutil.copy(os.path.join(class_dir, file), os.path.join(train_dir, file))

        for file in test_files:
            shutil.copy(os.path.join(class_dir, file), os.path.join(test_dir, file))


# 示例用法
data_dir = 'path_to_your_dataset'
output_dir = 'path_to_output_dataset'
split_dataset(data_dir, output_dir, test_size=0.2)