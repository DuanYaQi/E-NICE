import numpy as np
import torch
import torch.nn as nn
from typing_extensions import IntVar
from config import cfg

from modules import CouplingLayer, ScalingLayer, LogisticDistribution



class NICE(nn.Module): # 继承自nn.Module类。
    def __init__(self, data_dim, num_coupling_layers=3):# 耦合层3层
        super().__init__()   # 继承父类的__init__()方法

        self.data_dim = data_dim # 数据维度 784

        # 连续耦合层的交替遮罩方向
        masks = [self.__get_mask(data_dim, orientation=(i % 2 == 0))
                                                    for i in range(num_coupling_layers)]

        # general 耦合层，
        self.coupling_layers = nn.ModuleList([CouplingLayer(data_dim=data_dim,
                                                            hidden_dim=cfg['NUM_HIDDEN_UNITS'],
                                                            mask=masks[i],
                                                            num_layers=cfg['NUM_NET_LAYERS'])
                                                            for i in range(num_coupling_layers)])             
        # 尺度变化层
        self.scaling_layer = ScalingLayer(data_dim=data_dim)
        # 先验分布
        self.prior = LogisticDistribution()


    def forward(self, x, invert=False):
        if not invert: #正向
            z, log_det_jacobian = self.f(x) # 输入真实数据
            log_likelihood = torch.sum(self.prior.log_prob(z), dim=1) + log_det_jacobian
            return z, log_likelihood
        
        return self.f_inverse(x) # 逆向


    def f(self, x):
        z = x                   # 保证维数相同
        log_det_jacobian = 0    # 雅克比矩阵的行列式的对数
        for i, coupling_layer in enumerate(self.coupling_layers):     # 过耦合层
            z, log_det_jacobian = coupling_layer(z, log_det_jacobian)
        z, log_det_jacobian = self.scaling_layer(z, log_det_jacobian) # 过尺度变化层
        return z, log_det_jacobian


    def f_inverse(self, z):
        x = z
        x, _ = self.scaling_layer(x, 0, invert=True) #逆尺度变化层
        for i, coupling_layer in reversed(list(enumerate(self.coupling_layers))): # 逆耦合层，注意耦合层要反转
            x, _ = coupling_layer(x, 0, invert=True)
        return x


    def sample(self, num_samples):
        # 采样
        z = self.prior.sample([num_samples, self.data_dim]).view(self.samples, self.data_dim)
        return self.f_inverse(z)


    def __get_mask(self, dim, orientation=True):
        mask = np.zeros(dim)    # 按照维度创建
        mask[::2] = 1.          # list[start:end:step] start:默认起始位置 end:默认结束位置 step:默认步长为1 中间隔1个step赋值 [0 1 0 1 0 1] 
        if orientation:         # 直接反转
            mask = 1. - mask
        mask = torch.tensor(mask) # 转张量
        if cfg['USE_CUDA']:
            mask = mask.cuda()  # 放入cuda内存中
        return mask.float()     # 浮点数