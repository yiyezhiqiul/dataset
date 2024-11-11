
import os

import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.utils_metrics import compute_mIoU, show_results
import glob
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
import os
import cv2
from model.unet_model import UNet
from model.Net24 import Net
from model.Unet import NestedUnet
from torchvision import transforms
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
], dtype=np.uint8)




label_mapping = {
    'Serrated adenoma':  1,
    'Adenocarcinoma': 2,
    'Low-grade IN':  3,
    'Normal':  4,
    'High-grade IN': 5,
    'Polyp':  6
}
def cal_miou(test_dir="/media/totem_disk/totem/huang/8-2-png-EBHI/Test_Images",
             pred_dir="/media/totem_disk/totem/huang/code/u-net/unet_42/results", gt_dir="/media/totem_disk/totem/huang/8-2-png-EBHI/Test_Labels"):
    # ---------------------------------------------------------------------------#
    #   miou_mode用于指定该文件运行时计算的内容
    #   miou_mode为0代表整个miou计算流程，包括获得预测结果、计算miou。
    #   miou_mode为1代表仅仅获得预测结果。
    #   miou_mode为2代表仅仅计算miou。
    # ---------------------------------------------------------------------------#
    miou_mode = 0
    num_classes = 7
    size = 512
    if miou_mode == 0 or miou_mode == 1:
        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)

        print("Load model.")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 加载网络，图片单通道，分类为1。
        net = UNet(n_channels=3, n_classes=num_classes)
        # net = NestedUnet(in_channels=3, out_channels=7)    #U-net++
        # net = Net(3, 7)
        # 将网络拷贝到deivce中
        net.to(device=device)
        # 加载模型参数
        net.load_state_dict(torch.load('/media/totem_disk/totem/huang/code/u-net/unet_42/权重/U-net-EBHI80.pth', map_location=device)) # todo
        # 测试模式
        net.eval()
        name_classes = ["bg"]
        for name in os.listdir(test_dir):
            name_classes.append(name)
        #自动获取名称

            # 图像预处理
            transform = transforms.Compose([
                transforms.ToPILImage(),  # 将 NumPy 数组转换为 PIL.Image
                transforms.ToTensor(),  # 转换为 Tensor
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 归一化
            ])

            # 遍历测试图像
            for class_name in tqdm(os.listdir(test_dir)):
                class_dir = os.path.join(test_dir, class_name)
                for image_id in os.listdir(class_dir):
                    image_path = os.path.join(test_dir, class_name, image_id)
                    img = cv2.imread(image_path)
                    origin_shape = img.shape
                    stride = size
                    if img.shape[0] > size or img.shape[1] > size:
                        # 计算需要的窗口数量
                        num_rows = (img.shape[0] - size) // stride + 1
                        num_cols = (img.shape[1] - size) // stride + 1
                        # 处理最后一行和最后一列
                        if (img.shape[0] - size) % stride != 0:
                            num_rows += 1
                        if (img.shape[1] - size) % stride != 0:
                            num_cols += 1
                        # 初始化结果图像和权重矩阵
                        result = np.zeros((img.shape[0], img.shape[1], num_classes), dtype=np.float32)
                        weight_matrix = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
                        for row in range(num_rows):
                            for col in range(num_cols):
                                start_y = row * stride
                                start_x = col * stride
                                # 处理最后一行和最后一列的特殊情况
                                if start_y + size > img.shape[0]:
                                    start_y = img.shape[0] - size
                                if start_x + size > img.shape[1]:
                                    start_x = img.shape[1] - size
                                patch = img[start_y:start_y + size, start_x:start_x + size]
                                patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
                                patch = transform(patch).unsqueeze(0).float().to(device)
                                with torch.no_grad():
                                    output = net(patch)
                                    output = torch.softmax(output, dim=1).squeeze(0).cpu().numpy()
                                result[start_y:start_y + size, start_x:start_x + size] += output
                                weight_matrix[start_y:start_y + size, start_x:start_x + size] += 1

                        # 取平均值
                        result /= np.expand_dims(weight_matrix, axis=-1)
                        result = result.argmax(axis=-1)
                    else:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = cv2.resize(img, (size, size))
                        img = transform(img).unsqueeze(0).float().to(device)
                        with torch.no_grad():
                            output = net(img)
                            output = torch.softmax(output, dim=1).squeeze(0).cpu().numpy()

                        result = output.argmax(axis=0)
                    # 保存预测结果
                    pred_path = os.path.join(pred_dir, class_name, image_id)
                    os.makedirs(os.path.dirname(pred_path), exist_ok=True)
                    cv2.imwrite(pred_path, result.astype(np.uint8))

        if miou_mode == 0 or miou_mode == 2:
            print("Get miou.")
            print(gt_dir)
            print(pred_dir)
            print(num_classes)
            print(name_classes)
            hist, IoUs, PA_Recall, Precision = compute_mIoU(gt_dir, pred_dir, num_classes,
                                                            name_classes,label_mapping)  # 执行计算mIoU的函数
            print("Get miou done.")
            miou_out_path = "results/"
            show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes)

if __name__ == '__main__':
    cal_miou()