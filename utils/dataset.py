
from torch.utils.data import Dataset
import os
import random
import glob
import numpy as np
import cv2
from torch.utils.data import Dataset
import torch
import torch
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
import numpy as np
import random

from PIL import Image, ImageFilter

class Loader(Dataset):
    def __init__(self, data_path, total_samples, transform=None):
        self.data_path = data_path
        self.total_samples_per_epoch = total_samples
        self.transform = transform
        self.class_to_index = {}
        self.class_images = {}  # 新增：存储每个类别的图像路径
        self.class_labels = {}  # 新增：存储每个类别的标签路径
        self.class_indices = {}

        # 图像和标签的根目录
        img_root = os.path.join(data_path, 'Training_Images')
        label_root = os.path.join(data_path, 'Training_Labels')

        # 遍历类别文件夹
        for class_folder in os.listdir(img_root):
            class_img_dir = os.path.join(img_root, class_folder)
            class_label_dir = os.path.join(label_root, class_folder)

            if not os.path.isdir(class_img_dir) or not os.path.isdir(class_label_dir):
                continue

            # 获取图像路径
            imgs_in_class = glob.glob(os.path.join(class_img_dir, '*.jpg'))
            self.class_images[class_folder] = imgs_in_class

            # 获取标签路径
            # labels_in_class = [path.replace('Training_Images', 'Training_Labels') for path in imgs_in_class]
            # self.class_labels[class_folder] = labels_in_class

            # 创建类别到索引的映射
            self.class_to_index[class_folder] = len(self.class_to_index)

            # 创建类别到索引列表的映射
            self.class_indices[class_folder] = list(range(len(imgs_in_class)))
            # 打印每个类别的图像数量
            print(f"类别 {class_folder} 的图像数量: {len(imgs_in_class)}")

        # 计算每个类别应该有的样本数量
        num_classes = len(self.class_to_index)
        samples_per_class = self.total_samples_per_epoch // num_classes
        remainder = self.total_samples_per_epoch % num_classes

        # 对每个类别进行采样
        self.sampled_paths = []  # 新增：存储采样的图像路径
        self.samples_per_class = {}
        self.resampled_counts = {}  # 新增：记录每个类别的重采样数量
        for class_folder, indices in self.class_indices.items():
            if len(indices) >= samples_per_class:
                sampled = random.sample(indices, samples_per_class)
                resampled_count = 0  # 当前类别不需要重采样
            else:
                # 如果类别样本不足，则进行重采样
                sampled = random.choices(indices, k=samples_per_class)
                resampled_count = samples_per_class - len(indices)  # 计算重采样的数量

            # 从每个类别中采样图像路径
            sampled_image_paths = [self.class_images[class_folder][i] for i in sampled]
            self.sampled_paths.extend(sampled_image_paths)
            self.samples_per_class[class_folder] = len(sampled)
            self.resampled_counts[class_folder] = resampled_count  # 记录重采样数量

        # 将剩余的样本分配给类别
        if remainder > 0:
            extra_samples = random.choices(self.sampled_paths, k=remainder)
            self.sampled_paths.extend(extra_samples)

        # 输出每个类别对应的数字及采样数目
        print("每个类别对应的数字及采样数目:")
        for class_folder, class_index in self.class_to_index.items():
            # 输出每个类别对应的数字及采样数目
            print(f"{class_folder}: 数字 {class_index+1}, 采样数目 {self.samples_per_class[class_folder]}, 重采样数量 {self.resampled_counts.get(class_folder, 0)}")

        # # 打印最终采样路径
        # print("最终采样路径:", self.sampled_paths)

    def augment(self, image, label, flipCode):
        if flipCode != 0:
            image = cv2.flip(image, flipCode)
            label = cv2.flip(label, flipCode)
        return image, label



    def add_gaussian_noise(self, image, mean=0, std=0.1):
        noise = np.random.normal(mean, std, image.shape)
        noisy = image + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)

    def scale_image(self, image, label,scale_factor):
        new_size = (int(image.shape[1] * scale_factor), int(image.shape[0] * scale_factor))
        scaled = cv2.resize(image, new_size)
        scaled_label = cv2.resize(label,new_size,interpolation=cv2.INTER_NEAREST)
        return scaled,scaled_label

    def rotate_image(self, image, label, angle):
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv2.warpAffine(image, M, (w, h))
        rotated_label = cv2.warpAffine(label, M, (w, h))
        return rotated_image, rotated_label

    def random_crop(self, image, label, crop_size=(64, 64)):
        h, w = image.shape[:2]
        start_x = random.randint(0, w - crop_size[0])
        start_y = random.randint(0, h - crop_size[1])
        cropped_image = image[start_y:start_y + crop_size[1], start_x:start_x + crop_size[0]]
        cropped_label = label[start_y:start_y + crop_size[1], start_x:start_x + crop_size[0]]
        return cropped_image, cropped_label


    def __getitem__(self, index):
        image_path = self.sampled_paths[index]  # 使用采样路径
        label_path = image_path.replace('Training_Images', 'Training_Labels').replace('.jpg','_mask.png')

        image = cv2.imread(image_path)
        label = cv2.imread(label_path)

        if image is None or label is None:
            raise FileNotFoundError(f"Image or label file not found: {image_path} or {label_path}")



        # 灰度
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
        # RGB
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = np.transpose(image, (2, 0, 1))

        class_folder = os.path.basename(os.path.dirname(image_path))
        class_index = self.class_to_index[class_folder]
        # label = (label > 0).astype(int) * (class_index + 1)
        # class_name = class_folder
        # if class_name == 'Serrated adenoma':
        #     label[label > 0] = 1
        # if class_name == 'Adenocarcinoma':
        #     label[label > 0] = 2
        # if class_name == 'Low-grade IN':
        #     label[label > 0] = 3
        # if class_name == 'Normal':
        #     label[label > 0] = 4
        # if class_name == 'High-grade IN':
        #     label[label > 0] = 5
        # if class_name == 'Polyp':
        #     label[label > 0] = 6
        label[label > 0] = 1

        if random.random() < 0.5:  # 50%的概率应用增强
            angle = random.randint(-10, 10)
            image, label = self.rotate_image(image, label, angle)
        if random.random() < 0.5:
            image = self.add_gaussian_noise(image)
        if random.random() < 0.5:
            image, label = self.scale_image(image, label, scale_factor=random.uniform(0.8, 1.2))
        if random.random() < 0.5:
            crop_size = (64, 64)
            image, label = self.random_crop(image, label, crop_size)

        image = cv2.resize(image, (512, 512))
        label = cv2.resize(label, (512, 512), interpolation=cv2.INTER_NEAREST)


        label = label.astype(np.int64)  # 确保 label 是 int64 类型
        # label = torch.from_numpy(label).long()  # 转换为 PyTorch Long 类型
        image = torch.from_numpy(image).float().unsqueeze(0)  # 添加通道维度
        # label = torch.from_numpy(label).float().unsqueeze(0)  # 添加通道维度
        # 将标签转换为 one-hot 编码

        num_classes = 2  # 假设有7个类别
        label = torch.nn.functional.one_hot(torch.from_numpy(label), num_classes=num_classes)
        label = label.permute(2, 0, 1).float()



        if self.transform:
            image,label = self.transform(image,label)


        return image, label



    def __len__(self):
        return len(self.sampled_paths)

if __name__ == "__main__":
    isbi_dataset = Loader("/media/totem_disk/totem/huang/8-2-png",total_samples=1000)
    print("数据个数：", len(isbi_dataset))
    train_loader = torch.utils.data.DataLoader(dataset=isbi_dataset,
                                               batch_size=64,
                                               shuffle=True)
    for image, label in train_loader:
        # print(image.shape)
        print(label.shape)
        # print(label)
        # print(image.shape)