import random

from tiffslide.tiffslide import TF


class SegmentationTransforms:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, label):
        if random.random() > self.p:
            return image, label

        # 随机水平翻转
        if random.random() > 0.5:
            image = TF.hflip(image)
            label = TF.hflip(label)

        # 随机垂直翻转
        if random.random() > 0.5:
            image = TF.vflip(image)
            label = TF.vflip(label)

        # 随机旋转
        angle = random.randint(-10, 10)
        image = TF.rotate(image, angle)
        label = TF.rotate(label, angle)

        # 随机裁剪和缩放
        i, j, h, w = TF.RandomResizedCrop.get_params(image, scale=(0.8, 1.0), ratio=(1.0, 1.0))
        image = TF.resized_crop(image, i, j, h, w, size=(480, 320))
        label = TF.resized_crop(label, i, j, h, w, size=(480, 320))

        # 弹性变换
        image, label = self.elastic_transform(image, label)

        # 高斯模糊
        image = self.gaussian_blur(image)

        return image, label