import torch
import torch.nn as nn
import torch.nn.functional as float

class CouplingLayer(nn.modules):
    
    def __init__(self, data_dim, hidden_dim, mask, num_layers=4):
        '''
        NICE论文中的第3.2节         general coupling
        data_dim    数据维度
        hideen_dim  隐藏节点维度
        mask        遮蔽模板
        num_layers  耦合次数
        '''
        super.__init__()

        assert data_dim % 2 == 0  #断言数据维度都是偶数 方便后边切割

        self.mask = mask   

        modules = [nn.Linear(data_dim, hidden_dim), nn.LeakyReLU(0.2)]  # 线性层+leakyrelu激活函数

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


class ScalingLayer(nn.modules):
    
    def __init__(self, data_dim):
        '''
        NICE论文中的第3.3节         尺度变换层
        '''
        super().__init__()
        # 随机标准正态分布   aize [1, data_dim]
        self.log_scale_vector = nn.Parameter(torch.randn(1 ,
                                                        data_dim, 
                                                        requires_grad=True))

    def forward(self, x, logdet, invert=False):
        '''
        x           特征向量
        logdet      行列式的对数
        invert      正向训练还是反向生成
        '''
        log_det_jacobian = torch.sum(self.log_scale_vector)
        
        if invert:
            return torch.exp(- self.log_scale_vector) * x, logdet - log_det_jacobian

        return torch.exp(self.log_scale_vector) * x, logdet + log_det_jacobian
       
