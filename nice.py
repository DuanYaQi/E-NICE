import numpy as np
import torch
import torch.nn as nn
from config import cfg

from modules import CouplingLayer, ScalingLayer



class NICE(nn.Module): # 继承自nn.Module类。
    def __init__(self, data_dim, num_coupling_layers=3):
        super.__init__()   # 继承父类的__init__()方法

        self.data_dim = data_dim

        # 连续耦合层的交替遮罩方向
        masks = [self.__get_mask(data_dim, orientation=(i % 2 == 0))
                                                    for i in range(num_coupling_layers)]

        # general 耦合层，
        self.coupling_layers = nn.ModuleList([CouplingLayer(data_dim=data_dim,
                                                            hidden_dim=cfg['NUM_HIDDEN_UNITS'],
                                                            mask=masks[i],
                                                            num_layers=cfg['NUM_NET_LAYERS'])
                                                            for i in range(num_coupling_layers)])             
        self.scaling_layer = ScalingLayer(data_dim=data_dim)



        def __get_mask(self, dim, orientation=True):
            mask = np.zeros(dim)    # 按照维度创建
            mask[::2] = 1.          # list[start:end:step] start:默认起始位置 end:默认结束位置 step:默认步长为1
            if orientation:         # 直接反转
                mask = 1. - mask
            mask = torch.tensor(mask) # 转张量
            if cfg['USE_CUDA']:
                mask = mask.cuda()  # 放入cuda内存中
            return mask.float()     # 浮点数