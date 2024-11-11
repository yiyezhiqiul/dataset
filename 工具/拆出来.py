import os
import shutil

# 原始数据集路径
dataset_path = '/media/totem_disk/totem/huang/test'  # 修改为你的数据集路径
# 新的数据集路径
new_dataset_path = '/media/totem_disk/totem/huang/PANDA'

# 遍历原始数据集
for category in os.listdir(dataset_path):
    category_path = os.path.join(dataset_path, category)

    if os.path.isdir(category_path):  # 确保是目录
        # 创建新的类别文件夹
        os.makedirs(os.path.join(new_dataset_path, 'Test_Images',category), exist_ok=True)
        os.makedirs(os.path.join(new_dataset_path, 'Test_Labels',category), exist_ok=True)

        for folder in os.listdir(category_path):
            folder_path = os.path.join(category_path, folder)

            if os.path.isdir(folder_path):  # 确保是子目录
                for file in os.listdir(folder_path):
                    if file.endswith('.jpg'):  # 假设图像格式是 .png
                        # 构造图像和掩码的路径
                        img_path = os.path.join(folder_path, file)
                        mask_path = os.path.join(folder_path, file.replace('.jpg', '_mask.png'))

                        # 检查原图和掩码是否都存在
                        if os.path.exists(img_path) and os.path.exists(mask_path):
                            # 复制图像到新文件夹
                            shutil.copy(img_path, os.path.join(new_dataset_path,'Test_Images', category, file))
                            # 复制掩码到新文件夹
                            shutil.copy(mask_path, os.path.join(new_dataset_path,'Test_Labels', category, file.replace('.jpg', '_mask.png')))

                        # 如果原图存在但掩码不存在，原图也不复制
                        elif os.path.exists(img_path):
                            continue

                        # 如果掩码存在但原图不存在，掩码也不复制
                        elif os.path.exists(mask_path):
                            continue
