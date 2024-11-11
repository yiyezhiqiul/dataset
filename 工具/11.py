import os

from skimage import io

img_path ='/media/totem_disk/totem/huang/SICAPv2/SICAPv2/images'
for img in os.listdir(img_path):
    img = io.imread(os.path.join(img_path,img))
    print(img.shape)












# import os
# import shutil
# import random
#
# # 设置路径
# img_path = '/media/totem_disk/totem/huang/SICAPv2/SICAPv2/images'  # 图像文件夹路径
# label_path = '/media/totem_disk/totem/huang/SICAPv2/SICAPv2/masks'  # 标签文件夹路径
# train_img_path = '/media/totem_disk/totem/huang/SICAPv2/Training_Images'  # 训练图像存储路径
# train_label_path = '/media/totem_disk/totem/huang/SICAPv2/Training_Labels'  # 训练标签存储路径
# test_img_path = '/media/totem_disk/totem/huang/SICAPv2/Test_Images'  # 测试图像存储路径
# test_label_path = '/media/totem_disk/totem/huang/SICAPv2/Test_Lables'  # 测试标签存储路径
#
# # 创建文件夹
# os.makedirs(train_img_path, exist_ok=True)
# os.makedirs(train_label_path, exist_ok=True)
# os.makedirs(test_img_path, exist_ok=True)
# os.makedirs(test_label_path, exist_ok=True)
#
# # 获取所有图像文件和标签文件
# image_files = [f for f in os.listdir(img_path) if f.endswith(('.jpg', '.png'))]
# label_files = [f for f in os.listdir(label_path) if f.endswith(('.jpg', '.png'))]
#
# # 打乱文件列表
# random.shuffle(image_files)
#
# # 计算分割点
# split_index = int(len(image_files) * 0.8)
#
# # 划分训练集和测试集
# train_images = image_files[:split_index]
# test_images = image_files[split_index:]
#
# # 移动训练集图像和标签
# for img in train_images:
#     shutil.copy(os.path.join(img_path, img), os.path.join(train_img_path, img))
#     shutil.copy(os.path.join(label_path, img), os.path.join(train_label_path, img))
#
# # 移动测试集图像和标签
# for img in test_images:
#     shutil.copy(os.path.join(img_path, img), os.path.join(test_img_path, img))
#     shutil.copy(os.path.join(label_path, img), os.path.join(test_label_path, img))
#
# print("数据集划分完成！")