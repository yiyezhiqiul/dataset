import os
import random
import numpy as np
import cv2
import albumentations as A
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import tifffile  # 确保已安装 tifffile

def create_augmentations(cfg: dict):
    """
    根据配置创建数据增强管道。
    """
    tf = [Space()]
    if cfg['level'] == 'strong':
        tf += strong()
    elif cfg['level'] == 'weak':
        tf += weak()
    elif cfg['level'] == 'off':
        # If off, no transfer added
        return A.Compose([])
    else:
        raise ValueError(f'Transfer-level not legal')
    if cfg['elastic'] == 'on':
        tf += elastic()
    if cfg['cut'] == 'on':
        tf += cut()
    return A.Compose(tf)


class Space:
    def __call__(self, image: np.ndarray, mask: np.ndarray):
        code = random.randint(0, 7)
        # 二分之一概率翻转
        if code % 2:
            image = cv2.flip(image, 0)
            mask = cv2.flip(mask, 0)
        # 四分之一概率不转
        code = code // 2
        if code != 3:
            image = cv2.rotate(image, code)
            mask = cv2.rotate(mask, code)
        return {
            'image': image,
            'mask': mask,
        }


def weak():
    return [
        A.OneOf([
            A.RandomGamma(gamma_limit=(75, 135), p=1),
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=1
            ),
        ], p=0.8),
        A.OneOf([
            A.GaussNoise(),
        ], p=0.8),
    ]


def strong():
    return [
        A.RandomGamma(gamma_limit=(75, 135), p=0.5),
        A.Equalize(p=0.5),
        A.Solarize(p=0.5),
        A.Posterize(p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=0.3,
            contrast_limit=0.3,
            p=0.5,
        ),
        A.GaussNoise(p=0.5),
        A.GaussianBlur(blur_limit=3, p=0.4),  # 使用高斯模糊，blur_limit 可以调整模糊程度
    ]


def elastic():
    return [A.ElasticTransform(
        alpha=500,
        sigma=50,
        alpha_affine=0,
        p=0.8,
    )]


def cut():
    return [
        A.OneOf([
            A.GridDropout(
                unit_size_min=2,
                unit_size_max=5,
                p=1,
            ),
            A.CoarseDropout(
                mask_fill_value=0,
                min_holes=1,
                max_holes=3,
                min_width=50,
                max_width=200,
                min_height=50,
                max_height=200,
                p=1,
            ),
        ], p=0.2),
    ]


class MedicalImageDataset(Dataset):
    def __init__(self, image_dir, label_dir, image_format=['.tif'], transform=None, augmentations=None):
        """
        初始化医学图像数据集。

        参数:
        - image_dir (str): 包含原始图像的目录路径。
        - label_dir (str): 包含标签图像的目录路径。
        - image_format (list of str): 支持的图像文件格式，默认为 ['.tif']。
        - transform (callable, optional): 可选的图像转换函数。
        - augmentations (callable, optional): 可选的数据增强函数。
        """
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.augmentations = augmentations
        self.image_format = image_format

        # 获取所有图像文件名，并确保每个图像都有对应的标签文件
        self.images = []
        for img_format in image_format:
            self.images.extend([img for img in os.listdir(image_dir) if img.endswith(img_format) and os.path.exists(
                os.path.join(label_dir, img))])  # 标签文件名与图像文件名相同

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        label_path = os.path.join(self.label_dir, self.images[index])  # 标签文件名与图像文件名相同

        try:
            if img_path.endswith('.tif'):
                image = tifffile.imread(img_path)  # 使用 tifffile 读取 .tif 文件
                label = tifffile.imread(label_path)  # 读取标签为灰度图像

            if self.augmentations is not None:
                augmented = self.augmentations(image=image, mask=label)
                image = augmented['image']
                label = augmented['mask']

            # 如果需要，可以在这里进行额外的处理，例如将标签转换为 Tensor 并确保其为单通道
            if isinstance(label, np.ndarray):
                label = torch.from_numpy(label).unsqueeze(0)  # 添加通道维度

            if self.transform is not None:
                image = self.transform(image)

        except IOError as e:
            raise IOError(f"Error loading file: {e}")

        return image, label


# 使用示例
if __name__ == '__main__':
    cfg = {
        'level': 'strong',  # 或者 'weak' 或 'off'
        'elastic': 'on',
        'cut': 'on',
    }

    augmentations = create_augmentations(cfg)

    image_dir = '/media/totem_disk/totem/huang/tif/high-3/slide'
    label_dir = '/media/totem_disk/totem/huang/tif/high-3/label'

    # 图像预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # 创建数据集实例
    dataset = MedicalImageDataset(image_dir=image_dir, label_dir=label_dir, image_format=['.tif'],
                                  transform=transform, augmentations=augmentations)

    print(f"Number of images found: {len(dataset)}")

    # 随机选择 num_samples 个样本进行展示
    num_samples = min(5, len(dataset))  # 确保 num_samples 不超过数据集大小
    sample_indices = random.sample(range(len(dataset)), num_samples)

    for index in sample_indices:
        # 读取原始图像
        img_path = os.path.join(image_dir, dataset.images[index])
        original_image = tifffile.imread(img_path)
        label_path = os.path.join(label_dir, dataset.images[index])
        original_label = tifffile.imread(label_path)

        # 获取增强后的图像
        image, label = dataset[index]

        # 将 tensor 转换回 numpy 数组以便展示
        image = image.permute(1, 2, 0).numpy() * 255  # 将张量转换为 numpy 数组，并缩放至 [0, 255] 范围
        # label = label.squeeze().numpy()

        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(original_image.astype(np.uint8))
        ax[0].set_title('Original Image')
        ax[1].imshow(image.astype(np.uint8))
        ax[1].set_title('Augmented Image')
        ax[2].imshow(label, cmap='gray')
        ax[2].set_title('Label')
        plt.show()