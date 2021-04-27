# E-NICE
PyTorch implementation of NICE: NON-LINEAR INDEPENDENT COMPONENTS ESTIMATION

## Overrall Dimension Processing


```python
input 	[1, 784]
output 	[1, 784]
4 coupling layer + 1 scaling layer
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Linear-1              [-1, 1, 1000]         785,000
         LeakyReLU-2              [-1, 1, 1000]               0
            Linear-3              [-1, 1, 1000]       1,001,000
         LeakyReLU-4              [-1, 1, 1000]               0
            Linear-5              [-1, 1, 1000]       1,001,000
         LeakyReLU-6              [-1, 1, 1000]               0
            Linear-7              [-1, 1, 1000]       1,001,000
         LeakyReLU-8              [-1, 1, 1000]               0
            Linear-9              [-1, 1, 1000]       1,001,000
        LeakyReLU-10              [-1, 1, 1000]               0
           Linear-11               [-1, 1, 784]         784,784
----------------------------------------------------------------
           Linear-12              [-1, 1, 1000]         785,000
        LeakyReLU-13              [-1, 1, 1000]               0
           Linear-14              [-1, 1, 1000]       1,001,000
        LeakyReLU-15              [-1, 1, 1000]               0
           Linear-16              [-1, 1, 1000]       1,001,000
        LeakyReLU-17              [-1, 1, 1000]               0
           Linear-18              [-1, 1, 1000]       1,001,000
        LeakyReLU-19              [-1, 1, 1000]               0
           Linear-20              [-1, 1, 1000]       1,001,000
        LeakyReLU-21              [-1, 1, 1000]               0
           Linear-22               [-1, 1, 784]         784,784
----------------------------------------------------------------
           Linear-23              [-1, 1, 1000]         785,000
        LeakyReLU-24              [-1, 1, 1000]               0
           Linear-25              [-1, 1, 1000]       1,001,000
        LeakyReLU-26              [-1, 1, 1000]               0
           Linear-27              [-1, 1, 1000]       1,001,000
        LeakyReLU-28              [-1, 1, 1000]               0
           Linear-29              [-1, 1, 1000]       1,001,000
        LeakyReLU-30              [-1, 1, 1000]               0
           Linear-31              [-1, 1, 1000]       1,001,000
        LeakyReLU-32              [-1, 1, 1000]               0
           Linear-33               [-1, 1, 784]         784,784
----------------------------------------------------------------
           Linear-34              [-1, 1, 1000]         785,000
        LeakyReLU-35              [-1, 1, 1000]               0
           Linear-36              [-1, 1, 1000]       1,001,000
        LeakyReLU-37              [-1, 1, 1000]               0
           Linear-38              [-1, 1, 1000]       1,001,000
        LeakyReLU-39              [-1, 1, 1000]               0
           Linear-40              [-1, 1, 1000]       1,001,000
        LeakyReLU-41              [-1, 1, 1000]               0
           Linear-42              [-1, 1, 1000]       1,001,000
        LeakyReLU-43              [-1, 1, 1000]               0
           Linear-44               [-1, 1, 784]         784,784
----------------------------------------------------------------
     ScalingLayer-45               [-1, 1, 784]             784
================================================================
Total params: 22,295,920
Trainable params: 22,295,920
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.002991
Forward/backward pass size (MB): 0.335083
Params size (MB): 85.052185
Estimated Total Size (MB): 85.390259
----------------------------------------------------------------
```



## Coupling_layer Dimension Processing

```python
input 	[1, 784]
output 	[1, 784]
5 FCN-leakyrelu + 1 FCN(Fully Connection Layer)
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Linear-1              [-1, 1, 1000]         785,000
         LeakyReLU-2              [-1, 1, 1000]               0
            Linear-3              [-1, 1, 1000]       1,001,000
         LeakyReLU-4              [-1, 1, 1000]               0
            Linear-5              [-1, 1, 1000]       1,001,000
         LeakyReLU-6              [-1, 1, 1000]               0
            Linear-7              [-1, 1, 1000]       1,001,000
         LeakyReLU-8              [-1, 1, 1000]               0
            Linear-9              [-1, 1, 1000]       1,001,000
        LeakyReLU-10              [-1, 1, 1000]               0
           Linear-11               [-1, 1, 784]         784,784
================================================================
Total params: 5,573,784
Trainable params: 5,573,784
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.002991
Forward/backward pass size (MB): 0.082275
Params size (MB): 21.262299
Estimated Total Size (MB): 21.347565
----------------------------------------------------------------
```


## Loss Visulization in real-time

```python
from torch.utils.tensorboard import SummaryWriter
# default dir ./runs 
writer = SummaryWriter() 
# run
$ tensorboard --logdir ./runs
```
