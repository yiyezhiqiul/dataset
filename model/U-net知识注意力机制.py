import torch
import torch.nn as nn
import torchvision.models as models
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.efficientnet import EfficientNet_V2_S_Weights, efficientnet_v2_s
from unet_model import UNet

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.efficientnet import EfficientNet_V2_S_Weights, efficientnet_v2_s
from unet_model import UNet

class EfficientNetV2sFeatureExtractor(nn.Module):
    def __init__(self, pretrained=True):
        super(EfficientNetV2sFeatureExtractor, self).__init__()
        weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
        self.model = efficientnet_v2_s(weights=weights)
        self.model.classifier = nn.Identity()  # 移除分类头

    def forward(self, x):
        features = self.model(x)
        return features

class UNetWithRNN(nn.Module):
    def __init__(self, feature_extractor, rnn_type='lstm', hidden_size=256, num_layers=1):
        super(UNetWithRNN, self).__init__()
        self.feature_extractor = feature_extractor
        self.block_size = 256
        self.channels = 3  # 假设输入是RGB图像
        self.feature_size = self.feature_extractor.model.features[-1].out_channels  # 获取EfficientNetV2s的输出特征维度

        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_size=self.feature_size, hidden_size=hidden_size,
                               num_layers=num_layers, batch_first=True)
        elif rnn_type == 'gru':
            self.rnn = nn.GRU(input_size=self.feature_size, hidden_size=hidden_size,
                              num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, self.block_size * self.block_size * self.channels)

    def forward(self, x):
        # x: (batch_size, sequence_length, block_size, block_size, channels)
        batch_size, seq_len, block_size, block_size, channels = x.size()
        x = x.view(batch_size, seq_len, block_size, block_size, channels).permute(0, 1, 4, 2, 3)  # (batch_size, sequence_length, channels, block_size, block_size)

        # Apply EfficientNetV2s to each block
        features = []
        for t in range(seq_len):
            block = x[:, t, :, :, :]
            feature = self.feature_extractor(block)
            features.append(feature)
        features = torch.stack(features, dim=1)  # (batch_size, sequence_length, feature_size)

        # Apply RNN/LSTM
        rnn_output, _ = self.rnn(features)

        # Final fully connected layer
        final_output = self.fc(rnn_output)
        final_output = final_output.view(batch_size, seq_len, block_size, block_size, channels)
        return final_output

class UNetWithEfficientNetV2sAndRNN(nn.Module):
    def __init__(self, unet_model, rnn_type='lstm', hidden_size=256, num_layers=1, unet_batch_size=8):
        super(UNetWithEfficientNetV2sAndRNN, self).__init__()
        self.feature_extractor = EfficientNetV2sFeatureExtractor(pretrained=True)
        self.rnn_model = UNetWithRNN(self.feature_extractor, rnn_type, hidden_size, num_layers)
        self.unet = unet_model
        self.unet_batch_size = unet_batch_size

    def forward(self, x):
        # x: (batch_size, height, width, block_size, block_size, channels)
        batch_size, height, width, block_size, block_size, channels = x.size()
        x = x.view(batch_size, -1, block_size, block_size, channels)  # (batch_size, sequence_length, block_size, block_size, channels)

        # Apply RNN/LSTM to the sequence of blocks
        rnn_output = self.rnn_model(x)

        # Reshape back to original dimensions
        rnn_output = rnn_output.view(batch_size, height, width, block_size, block_size, channels)

        # Apply U-Net to each block in batches
        unet_outputs = []
        num_blocks = height * width
        for i in range(0, num_blocks, self.unet_batch_size):
            batch_blocks = rnn_output[:, i:i+self.unet_batch_size, :, :, :, :]
            batch_blocks = batch_blocks.view(-1, block_size, block_size, channels).permute(0, 3, 1, 2)  # (batch_size * unet_batch_size, channels, block_size, block_size)
            unet_output = self.unet(batch_blocks)
            unet_outputs.append(unet_output)

        # Combine the U-Net outputs back into a single tensor
        combined_output = torch.cat(unet_outputs, dim=0)
        combined_output = combined_output.view(batch_size, height, width, channels, block_size, block_size)
        return combined_output


# 假设有一个基本的U-Net模型

def split_image(image, block_size=256, stride=128):
    blocks = []
    for y in range(0, image.shape[1] - block_size + 1, stride):
        for x in range(0, image.shape[2] - block_size + 1, stride):
            block = image[:, y:y+block_size, x:x+block_size, :]
            blocks.append(block)
    blocks = torch.stack(blocks, dim=1)  # (batch_size, num_blocks, block_size, block_size, channels)
    return blocks
# 创建模型实例
# 假设有一个基本的U-Net模型
# 假设有一个基本的U-Net模型
unet_model = UNet(3, 3)  # 定义您的U-Net模型
# 创建模型实例
model = UNetWithEfficientNetV2sAndRNN(unet_model, rnn_type='lstm', hidden_size=256, num_layers=1, unet_batch_size=8)
# 将模型移到GPU上
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
# 假设输入图像
image = torch.randn((1, 512, 512, 3)).to(device)  # (batch_size, height, width, channels)
# 划分图像块
blocks = split_image(image, block_size=256, stride=128)
# 调整形状以适应模型输入

blocks = blocks.unsqueeze(0).to(device)  # (batch_size, height, width, block_size, block_size, channels)
print(blocks.shape)# 前向传播
output = model(blocks)
print(output.shape)  # 输出形状应为 (batch_size, height, width, channels, block_size, block_size)