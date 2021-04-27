import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Distribution, Uniform
from config import cfg


class CouplingLayer(nn.Module):
    
    def __init__(self, data_dim, hidden_dim, mask, num_layers=4): # 这里的num_layers 是全连接的层数
        '''
        NICE论文中的第3.2节         general coupling
        data_dim    数据维度
        hideen_dim  隐藏节点维度
        mask        遮蔽模板
        num_layers  耦合次数
        '''
        super().__init__()

        assert data_dim % 2 == 0  #断言数据维度都是偶数 方便后边切割

        self.mask = mask   

        modules = [nn.Linear(data_dim, hidden_dim), nn.LeakyReLU(0.2)]  # 线性层+leakyrelu激活函数 784维升到1000维度

        for i in range(num_layers - 2):
            modules.append(nn.Linear(hidden_dim, hidden_dim))
            modules.append(nn.LeakyReLU(0.2))
        modules.append(nn.Linear(hidden_dim, data_dim))

        self.m = nn.Sequential(*modules)


    def forward(self, x, logdet, invert=False):
        '''
        x           特征向量
        logdet      行列式的对数
        invert      正向训练还是反向生成
        '''

        if not invert:
            # 通过遮蔽模板 分成两部分 直接赋值为0
            x1, x2 = self.mask * x, (1. - self.mask) * x
            # 一部分直接恒等赋值  y1 = x1
            # 另一部分进行m复杂线性运算 y2 = x2 + m(x1)
            y1, y2 = x1, x2 + (self.m(x1) * (1. - self.mask)) 
            return y1 + y2, logdet
        
        # 逆过程 生成过程
        # 通过遮蔽模板返回y1和y2
        y1, y2 = self.mask * x, (1. - self.mask) * x

        # 前部分直接恒等赋值 x1 = y1
        # 后半部分为 x2 = y2 - m(x1)       x2 = y2 - m(y1)
        x1, x2 = y1, y2 - (self.m(y1) * (1. - self.mask))
        return x1 + x2, logdet


class ScalingLayer(nn.Module):
    
    def __init__(self, data_dim):
        '''
        NICE论文中的第3.3节         尺度变换层
        '''
        super().__init__()

        # 随机标准正态分布   size [1, data_dim]
        self.log_scale_vector = nn.Parameter(torch.randn(1 ,
                                                        data_dim, 
                                                        requires_grad=True))

    def forward(self, x, logdet, invert=False):
        '''
        x           特征向量
        logdet      行列式的对数
        invert      正向训练还是反向生成
        '''
        # 尺度变换层的雅克比行列式，就是其对角阵元素之积  log_scale_vector [1, 784]
        log_det_jacobian = torch.sum(self.log_scale_vector)  # 因为是独立分量 log后可相加
        
        if invert:
            return torch.exp(- self.log_scale_vector) * x, logdet - log_det_jacobian

        return torch.exp(self.log_scale_vector) * x, logdet + log_det_jacobian
       
class LogisticDistribution(Distribution):
    '''
    3.4小节中的公式log(p(Hd))
    '''
    def __init__(self):
        super().__init__()

    def log_prob(self, x):
        # - 1/b * [ log(1+e^(b*x)) + * log(1+e^(b*-x)) ] beta 默认为 1
        # - [ log(1+e^(x)) + * log(1+e^(-x)) ] 
        return -(F.softplus(x) + F.softplus(-x))     

    def sample(self, size):
        # 采样 用于生成
        if cfg['USE_CUDA']:
            z = Uniform(torch.cuda.FloatTensor([0.]), torch.cuda.FloatTensor([1.])).sample(size)
        else:
            z = Uniform(torch.FloatTensor([0.]), torch.FloatTensor([1.])).sample(size)

        return torch.log(z) - torch.log(1. - z)
