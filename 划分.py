import os
import shutil
from sklearn.model_selection import train_test_split

# 设置源数据路径和目标数据路径
import os
import shutil
from sklearn.model_selection import train_test_split

# 设置源数据路径和目标数据路径
source_dir = '/media/totem_disk/totem/huang/EBHI-SEG'
train_img_dir = '/media/totem_disk/totem/huang/8-2-png/Training_Images'
val_img_dir = '/media/totem_disk/totem/huang/8-2-png/Test_Images'
train_label_dir = '/media/totem_disk/totem/huang/8-2-png/Training_Labels'
val_label_dir = '/media/totem_disk/totem/huang/8-2-png/Test_Labels'

# 创建目标数据文件夹
os.makedirs(train_img_dir, exist_ok=True)
os.makedirs(val_img_dir, exist_ok=True)
os.makedirs(train_label_dir, exist_ok=True)
os.makedirs(val_label_dir, exist_ok=True)

# 获取所有类别文件夹
classes = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]

for cls in classes:
    img_src = os.path.join(source_dir, cls, 'image')
    label_src = os.path.join(source_dir, cls, 'label')

    # 确保 img 和 label 文件夹存在
    if not os.path.isdir(img_src) or not os.path.isdir(label_src):
        print(f"Missing img or label folder in class {cls}. Skipping.")
        continue

    # 获取图像和标签文件名
    img_files = sorted(os.listdir(img_src))
    label_files = sorted(os.listdir(label_src))

    if len(img_files) != len(label_files):
        if len(img_files) > len(label_files):
            for name in img_files:
                if name not in label_files:
                    img_files.remove(name)
                    print(f"Removed image {name} due to missing label.")

    # 划分数据集
    img_train, img_val = train_test_split(img_files, test_size=0.2, random_state=42)
    label_train, label_val = train_test_split(label_files, test_size=0.2, random_state=42)

    # 创建类别文件夹
    os.makedirs(os.path.join(train_img_dir, cls), exist_ok=True)
    os.makedirs(os.path.join(val_img_dir, cls), exist_ok=True)
    os.makedirs(os.path.join(train_label_dir, cls), exist_ok=True)
    os.makedirs(os.path.join(val_label_dir, cls), exist_ok=True)

    # 复制训练集和验证集文件
    for file_name in img_train:
        shutil.copy(os.path.join(img_src, file_name), os.path.join(train_img_dir, cls, file_name))
    for file_name in img_val:
        shutil.copy(os.path.join(img_src, file_name), os.path.join(val_img_dir, cls, file_name))

    for file_name in label_train:
        shutil.copy(os.path.join(label_src, file_name), os.path.join(train_label_dir, cls, file_name))
    for file_name in label_val:
        shutil.copy(os.path.join(label_src, file_name), os.path.join(val_label_dir, cls, file_name))

print("数据和标签划分完成！")