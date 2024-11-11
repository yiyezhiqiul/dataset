import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from model.unet_model import UNet
from utils.dataset1 import Loader
from utils.train_loader import MedicalImageDataset, create_augmentations
import os
from model.loss_3 import BceAndGeneralizedDiceLoss
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from model.Net24 import Net
import torch.nn.functional as F
from model.Unet import NestedUnet
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

def get_acc(y_pred, y_true):
    # 获取每个像素的最大值对应的类别索引
    pred_classes = torch.argmax(y_pred, dim=1)  # shape: [batch_size, height, width]
    true_classes = torch.argmax(y_true, dim=1)  # shape: [batch_size, height, width]

    # 比较y_true和pred_classes
    equal_mask = torch.eq(pred_classes, true_classes)
    equal_count = torch.sum(equal_mask)
    num_elements = y_true.numel() // y_true.shape[1]  # 总像素数量

    acc = equal_count / num_elements
    return acc
def calculate_dice(pred, target, smooth=1e-5):
    # 将预测结果和目标转换为二进制形式
    pred = torch.round(torch.sigmoid(pred))
    # 计算交集
    intersection = (pred * target).sum(dim=(1, 2, 3))
    # 计算Dice系数
    dice = (2. * intersection + smooth) / (pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + smooth)
    # 返回平均Dice系数
    return dice.mean().item()


def train_net(net, device, data_path,output, classes,epochs=80, batch_size=32, lr=0.00001):
    # 加载训练集
    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    # 定义 Loss 函数
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.BCEWithLogitsLoss()
    # best_loss 统计，初始化为正无穷
    best_loss = float('inf')
    best_precison = 0

    # 训练epochs次
    for epoch in range(epochs):
        if epoch % 20 == 0 or epoch == 0:
            isbi_dataset = Loader(data_path, channel=3, num_class=classes, size=(256, 256))
            train_loader = DataLoader(dataset=isbi_dataset, batch_size=batch_size, shuffle=True)
        # 初始化损失和正确预测的数量
        running_loss = 0.0
        correct_predictions = 0
        dice_score = 0
        # 训练模式
        net.train()
        # 按照batch_size开始训练
        for image, label in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}'):
            optimizer.zero_grad()
            # 将数据拷贝到device中
            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.long)

            # 使用网络参数，输出预测结果
            pred = net(image)
            # _, pred_classes = torch.max(pred, dim=1)
            # pred1 = pred_classes.squeeze().cpu().numpy()
            # pred1 = COLORS[pred1]
            # plt.imshow(pred1)
            # plt.show()
            # 计算loss
            loss = criterion(pred, label)

            # 更新损失总和
            running_loss += loss.item() * image.size(0)
            if pred.shape != label.shape:
                label_one_hot = F.one_hot(label,num_classes=classes)  # 结果是 [batch_size, height, width, num_classes]
                label_one_hot = label_one_hot.permute(0, 3, 1, 2)  # 调整为 [num_classes, height, width]
            # 计算Dice系数
            dice = calculate_dice(pred, label_one_hot)
            dice_score += dice * image.size(0)
            # 计算精度
            accuary = get_acc(pred,label_one_hot)
            correct_predictions += (accuary*image.size(0))

            # 更新参数
            loss.backward()
            optimizer.step()

        # 计算整个epoch的平均损失
        epoch_loss = running_loss / len(isbi_dataset)
        # 计算整个epoch的平均精度
        epoch_accuracy = correct_predictions / len(isbi_dataset)
        # 将精度转换为百分比形式
        epoch_accuracy_percentage = epoch_accuracy *100
        # 计算整个epoch的平均Dice系数
        epoch_dice = dice_score / len(isbi_dataset)
        name = 'U-net++-EBHI'
        output_loss = output+'/'+name+'loss.pth'
        output_pre = output +'/'+name+'pre.pth'

        # 检查并保存最佳模型
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(net.state_dict(), output_loss)
        if best_precison < epoch_accuracy_percentage:
            best_precison = epoch_accuracy_percentage
            torch.save(net.state_dict(), output_pre)


        # 打印每个epoch的平均损失和精度
        print(
            f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy_percentage:.2f}%,Dice: {epoch_dice:.4f}')

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
if __name__ == "__main__":
    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    classes = 7

    # 加载网络，图片单通道1，分类为7。
    net = NestedUnet(in_channels=3, out_channels=7)    #U-net++
    # net = UNet(n_channels=3, n_classes=classes)  # todo edit input_channels n_classes
    # net = Net(3,7)               #EifficinetNet
    # 将网络拷贝到device中
    net.to(device=device)

    # 指定训练集地址，开始训练
    out_path ='/media/totem_disk/totem/huang/code/u-net/unet_42/权重'
    data_path = "/media/totem_disk/totem/huang/8-2-png-EBHI"  # todo 修改为你本地的数据集位置
    print("进度条出现卡着不动不是程序问题，是他正在计算，请耐心等待")
    train_net(net, device, data_path,out_path, epochs=100, batch_size=16,classes=classes)