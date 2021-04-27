import os
import torch
from argparse import ArgumentParser
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.tensorboard import SummaryWriter
# 默认创建目录 ./runs 
writer = SummaryWriter() 
# 执行命令 tensorboard --logdir ./runs

from nice import NICE

# PyTorch implementation of NICE

def train(model, dataloader, args, continueIndex):
    # 训练模型
    model.train()

    opt  = optim.Adam(model.parameters())

    for epoch in range(args.TRAIN_EPOCHS):
        mean_likelihood = 0.0
        num_minibatches = 0

        for batch_id, (x, _) in enumerate(dataloader): # 取出输入数据 x （256，1，28，28）
            # x (256, 784) 因为channel为1 
            x = x.view(-1, 784) + torch.rand(784) / 256. 

            if args.USE_CUDA: # 转到cuda
                x = x.cuda()

            x = torch.clamp(x, 0, 1) # 将input张量每个元素的夹紧到区间 [0, 1]
            
            z, likelihood = model(x)
            loss = -torch.mean(likelihood)

            loss.backward()
            opt.step()
            model.zero_grad()
            
            mean_likelihood -= loss
            num_minibatches += 1

        mean_likelihood /= num_minibatches
        writer.add_scalar('training loss', mean_likelihood, epoch)
        print('Epoch {} completed. Log Likelihood :{}'.format(epoch, mean_likelihood))

        if epoch % 5 == 0:
            if int(continueIndex):
                save_path = os.path.join(args.MODEL_SAVE_PATH, '{}.pt'.format(epoch + int(continueIndex) ) )
            else:
                save_path = os.path.join(args.MODEL_SAVE_PATH, '{}.pt'.format(epoch))
            torch.save(model.state_dict(), save_path)

def test(model, dataloader, args):
    # 训练模型
    model.eval()

    opt  = optim.Adam(model.parameters())

    for epoch in range(args.TRAIN_EPOCHS):
        mean_likelihood = 0.0
        num_minibatches = 0

        for batch_id, (x, _) in enumerate(dataloader): # 取出输入数据 x （256，1，28，28）
            # x (256, 784) 因为channel为1 
            x = x.view(-1, 784) + torch.rand(784) / 256. 

            if args.USE_CUDA: # 转到cuda
                x = x.cuda()

            x = torch.clamp(x, 0, 1) # 将input张量每个元素的夹紧到区间 [0, 1]
            
            z, likelihood = model(x)
            loss = -torch.mean(likelihood)

            loss.backward()
            opt.step()
            model.zero_grad()
            
            mean_likelihood -= loss
            num_minibatches += 1

        mean_likelihood /= num_minibatches
        writer.add_scalar('training loss', mean_likelihood, epoch)
        print('Epoch {} completed. Log Likelihood :{}'.format(epoch, mean_likelihood))

        if epoch % 5 == 0:
            save_path = os.path.join(args.MODEL_SAVE_PATH, '{}.pt'.format(epoch))
            torch.save(model.state_dict(), save_path)            

# -----------------------------------------------------------------------------------------
def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--MODEL_SAVE_PATH', type=str, help='Dir of mnist dataset', default = './saved_models/')
    parser.add_argument('--USE_CUDA', type=bool, default = True)
    parser.add_argument('--TRAIN_BATCH_SIZE', type=int, default = 256)
    parser.add_argument('--TRAIN_EPOCHS', type=int, default = 75)
    parser.add_argument('--NUM_COUPLING_LAYERS', type=int, default = 4)
    parser.add_argument('--NUM_NET_LAYERS', type=int, default = 6)
    parser.add_argument('--NUM_HIDDEN_UNITS', type=int, default = 1000)
    return parser.parse_args()

# -----------------------------------------------------------------------------------------
def main(phase='Train', checkpoint_path: str=None):
    args = parse_arguments()
    # 数据
    transform = transforms.ToTensor()  # 转化为张量

    # 保存路径、创建训练集，转换，下载
    dataset = datasets.MNIST(root='../data/mnist', 
                            train=True, 
                            transform=transform, 
                            download=True)

    # 数据加载器    pin_memory 数据加载器将张量复制到CUDA固定的内存中，然后返回它们
    dataloader = torch.utils.data.DataLoader(dataset, 
                                            batch_size=args.TRAIN_BATCH_SIZE, 
                                            shuffle=True,
                                            pin_memory=True)

    # 加载模型
    model = NICE(data_dim=784, num_coupling_layers=args.NUM_COUPLING_LAYERS)
    if args.USE_CUDA:
        device = torch.device('cuda')
        model = model.to(device)


    if phase == 'Train':
        train(model, dataloader, args, 0)
            
    elif phase == 'continueTrain':
        model.load_state_dict(torch.load(checkpoint_path))
        a = checkpoint_path.split('/')[2].split('.')[0]
        train(model, dataloader, args, a)

    elif phase == 'Test':
        model.load_state_dict(torch.load(checkpoint_path))
        test(model, dataloader, args)

    else:
        print('Error')        

# -----------------------------------------------------------------------------------------
if __name__ == "__main__":
    checkpoint_path = './saved_models/70.pt'
    main('continueTrain', checkpoint_path)