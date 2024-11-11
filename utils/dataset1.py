import matplotlib.pyplot as plt
import torch.nn.functional as F
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
from unet_42.utils.RandStainNA.randstainna import RandStainNA
from PIL import Image, ImageFilter
from torchvision import transforms
from .tissue import get_tissue
COLORS = np.array([
[100,100,100],
[0, 0, 0], #black
[0, 255, 255],#青色
[255, 0, 255], #purple
[128, 0, 0], #maroon
[255, 255, 0], #yellow
[255, 0, 0], #red
[0,0,255], #blue
[255, 255, 255],
[255, 255, 255],
[255, 255, 255],
[255, 255, 255],
], dtype=np.uint8)

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
    def __call__(self, tensor):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError('Input tensor should be a torch tensor. Got {}.'.format(type(tensor)))

        noise = torch.randn(tensor.size()) * self.std + self.mean
        noisy = tensor + noise
        return torch.clamp(noisy, 0, 1)  # 确保像素值在 [0, 1] 范围内
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
class Loader(Dataset):
    def __init__(self, data_path,channel, num_class,transform=None,size=(256,256)):
        self.data_path = data_path
        self.class_to_index = {}
        self.class_images = {}  # 新增：存储每个类别的图像路径
        self.class_labels = {}  # 新增：存储每个类别的标签路径
        self.class_indices = {}
        self.channel = channel
        self.num_class = num_class
        self.size = size
        self.window_size = size
        self.stride = (self.size[0] // 2, self.size[1] // 2)  # 对每个维度使用整数除法

        # 图像和标签的根目录
        img_root = os.path.join(data_path, 'Training_Images')
        label_root = os.path.join(data_path, 'Training_Labels')
        min_count = float('inf')
        max_count = 0


        # 遍历类别文件夹
        for class_folder in os.listdir(img_root):
            class_img_dir = os.path.join(img_root, class_folder)
            # 对应的图片类别文件夹名称
            class_label_dir = os.path.join(label_root, class_folder)

            if not os.path.isdir(class_img_dir) or not os.path.isdir(class_label_dir):
                print("数据有问题")
                continue

            # 获取图像路径
            imgs_in_class = glob.glob(os.path.join(class_img_dir, '*.png')) + glob.glob(os.path.join(class_img_dir, '*.jpg'))
            self.class_images[class_folder] = imgs_in_class

            # 获取标签路径
            labels_in_class = [path.replace('Training_Images', 'Training_Labels') for path in imgs_in_class]
            self.class_labels[class_folder] = labels_in_class

            # 创建类别到索引的映射
            self.class_to_index[class_folder] = len(self.class_to_index)

            # 创建类别到索引列表的映射
            self.class_indices[class_folder] = list(range(len(imgs_in_class)))
            # 打印每个类别的图像数量
            print(f"类别 {class_folder} 的图像数量: {len(imgs_in_class)}")
            count= len(imgs_in_class)
            min_count= min(min_count,count)
            max_count = max(max_count,count)

        # 计算每个类别应该有的样本数量
        if self.num_class == len(self.class_to_index)+1:
            print('自动预处理')
        elif self.num_class == 2:
            print('二值化分割')

        num_classes = len(self.class_to_index)
        avg = max_count-min_count
        if avg<min_count:
            total_samples_per_epoch = min_count+int(avg/num_classes)
        else:
            total_samples_per_epoch = min_count+int(min_count/num_classes)
        samples_per_class = int(total_samples_per_epoch)
        print(total_samples_per_epoch)
        remainder = int(total_samples_per_epoch % num_classes)

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
        if transform is not None:
            if isinstance(transform, transforms.Compose):
                pass  # 保持原样
            else:
                raise ValueError("Transform should be an instance of torchvision.transforms.Compose.")
        else:
            transform = transforms.Compose([
                RandStainNA(
                    yaml_file="/media/totem_disk/totem/huang/code/u-net/unet_42/utils/RandStainNA/CRC_LAB_randomTrue_n0.yaml",
                    std_hyper=-0.3,
                    probability=0.5,
                    distribution="normal",
                    is_train=True,
                ),
                transforms.ToPILImage(),  # 将 NumPy 数组转换为 PIL.Image
                transforms.ToTensor(),  # 转换为 Tensor
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 归一化
                AddGaussianNoise(std=0.1),  # 添加高斯噪声
            ])
        self.transform = transform

    def fill_holes(mask):
        # 填充孔洞
        mask_filled = mask.copy()
        contours, _ = cv2.findContours(mask_filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(mask_filled, contours, -1, 255, cv2.FILLED)
        # 平滑边缘
        mask_smoothed = cv2.GaussianBlur(mask_filled, (5, 5), 0)  # 使用高斯模糊
        return mask_smoothed

    def generate_patches(self, image, label):
        svs_regin_mask, _ = get_tissue(image, 2000)
        mask_filled = svs_regin_mask.copy()
        contours, _ = cv2.findContours(mask_filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(mask_filled, contours, -1, 255, cv2.FILLED)
        # 平滑边缘
        mask_smoothed = cv2.GaussianBlur(mask_filled, (5, 5), 0)  # 使用高斯模糊

        image_height, image_width = image.shape[:2]
        patches = []
        label_patches = []

        # 计算需要的窗口数量
        num_rows = (image_height - self.window_size[1]) // self.stride[1] + 1
        num_cols = (image_width - self.window_size[0]) // self.stride[0] + 1

        # 处理最后一行和最后一列
        if (image_height - self.window_size[1]) % self.stride[1] != 0:
            num_rows += 1
        if (image_width - self.window_size[0]) % self.stride[0] != 0:
            num_cols += 1

        for row in range(num_rows):
            for col in range(num_cols):
                start_y = row * self.stride[1]
                start_x = col * self.stride[0]

                # 处理最后一行和最后一列的特殊情况
                if start_y + self.window_size[1] > image_height:
                    start_y = image_height - self.window_size[1]
                if start_x + self.window_size[0] > image_width:
                    start_x = image_width - self.window_size[0]

                # 检查 mask_smoothed 是否大于 0
                threshold = 0.9  # 50% 的像素大于 0
                patch_mask = mask_smoothed[start_y:start_y + self.window_size[1], start_x:start_x + self.window_size[0]]
                if np.mean(patch_mask > 0) > threshold:
                    patch = image[start_y:start_y + self.window_size[1], start_x:start_x + self.window_size[0]]
                    label_patch = label[start_y:start_y + self.window_size[1], start_x:start_x + self.window_size[0]]
                    patches.append(patch)
                    label_patches.append(label_patch)
        if not patches:
            center_y = (image_height - self.window_size[1]) // 2
            center_x = (image_width - self.window_size[0]) // 2
            patch = image[center_y:center_y + self.window_size[1], center_x:center_x + self.window_size[0]]
            label_patch = label[center_y:center_y + self.window_size[1], center_x:center_x + self.window_size[0]]
            patches.append(patch)
            label_patches.append(label_patch)

        return patches, label_patches
    def __getitem__(self, index):
        image_path = self.sampled_paths[index]  # 使用采样路径
        label_path = image_path.replace('Training_Images', 'Training_Labels').replace('.jpg','.png')
        if not os.path.exists(label_path):
            label_path = label_path.replace('.png','.jpg')
        if not os.path.exists(label_path):
            label_path = label_path.replace('.png','_mask.png').replace('.jpg','_mask.png')
        if not os.path.exists(label_path):
            print(label_path)
            print(image_path)
            print('注意label和img图片格式')
        channel = self.channel
        num_class = self.num_class
        size = self.size

        image = cv2.imread(image_path)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)  # 直接读取为灰度图

        if channel == 1:
            # 灰度
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
        elif channel == 3:
            # RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        class_folder = os.path.basename(os.path.dirname(image_path))
        if num_class == max(self.class_to_index.values())+2:
            class_index = self.class_to_index[class_folder]
            # 将多通道的 label 转换成单一通道，并根据类别索引进行编码
            label = (label > 0).astype(int) * (class_index + 1)
        # 你的代码逻辑
        if num_class == 2:
            label[label > 0] = 1
        elif num_class !=2 and num_class != max(self.class_to_index.values())+2:
            print("要么类别错误了，记得加背景0")

        # 根据图像尺寸判断是否需要裁剪
            # 获取图像的高度和宽度

        image_height, image_width = image.shape[:2]
        if image_height > size[1] or image_width > size[0]:
            # 遍历图片
            patches,label_patches = self.generate_patches(image,label)
        else:
            # 否则，直接调整大小
            image = cv2.resize(image, (size[1], size[0]))
            # image = np.transpose(image, (2, 0, 1))
            label = cv2.resize(label, (size[1], size[0]), interpolation=cv2.INTER_NEAREST)
            patches = [image]
            label_patches = [label]
        transformed_images = []
        transformed_labels = []

        for patch, label_patch in zip(patches, label_patches):
            if self.transform and random.random() < 0.5:
                # 进行数据增强
                patch = self.transform(patch)
                label_patch = torch.tensor(label_patch, dtype=torch.long)
            else:
                transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                patch = transform(patch)
                label_patch = torch.tensor(label_patch, dtype=torch.long)
            transformed_images.append(patch)
            transformed_labels.append(label_patch)
        transformed_images = torch.stack(transformed_images)
        transformed_labels = torch.stack(transformed_labels)

        return transformed_images, transformed_labels
    def __len__(self):
        return len(self.sampled_paths)

if __name__ == "__main__":
    isbi_dataset = Loader("/media/totem_disk/totem/huang/PANDA",channel=3, num_class=2,size=(256,256))
    print("数据个数：", len(isbi_dataset))
    # 全局缓冲区
    # 全局缓冲区
    image_buffer = []
    label_buffer = []
    def dynamic_collate_fn(batch, max_total_patches=16):
        global image_buffer, label_buffer
        images, labels = zip(*batch)
        # 展平所有图像块并添加到缓冲区
        for sample_images, sample_labels in zip(images, labels):
            image_buffer.extend(sample_images)
            label_buffer.extend(sample_labels)
        print(len(image_buffer))
        # 从缓冲区中随机取出最多 max_total_patches 个图像块
        if len(image_buffer) >= max_total_patches:
            indices = np.random.choice(len(image_buffer), max_total_patches, replace=False)
            images = [image_buffer[i] for i in indices]
            labels = [label_buffer[i] for i in indices]
        else:
            images = image_buffer
            labels = label_buffer
        # 更新缓冲区
        image_buffer = image_buffer[max_total_patches:]
        label_buffer = label_buffer[max_total_patches:]
        images = torch.stack(images)
        labels = torch.stack(labels)
        return images, labels
    train_loader = torch.utils.data.DataLoader(dataset=isbi_dataset,
                                               batch_size=1,
                                               shuffle=True,collate_fn=lambda x: dynamic_collate_fn(x, max_total_patches=16))
    for image, label in train_loader:
        # print(image.shape)
        print(label.shape)
        print(image.shape)

        # print(label)
        # print(image.shape)