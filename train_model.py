import os
import torch
from config import cfg
import torch.optim as optim
from torchvision import transforms, datasets


from nice import NICE

# PyTorch implementation of NICE


# 数据
transform = transforms.ToTensor()  # 转化为张量

# 保存路径、创建训练集，转换，下载
dataset = datasets.MNIST(root='../data/mnist', 
                        train=True, 
                        transform=transform, 
                        download=True)

# 数据加载器    pin_memory 数据加载器将张量复制到CUDA固定的内存中，然后返回它们
dataloader = torch.utils.data.DataLoader(dataset, 
                                        batch_size=cfg['TRAIN_BATCH_SIZE'], 
                                        shuffle=True,
                                        pin_memory=True)

# 加载模型
model = NICE(data_dim=784, num_coupling_layers=cfg['NUM_COUPLING_LAYERS'])

