from network import Net,resnet
from dataset import train_data,val_data

import os
import time

import torch
from torchvision import models, transforms
from torch import optim, nn
from torch.autograd import Variable
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

batch_size=8
epochs =500
train_root='D:\\lhq\\catdog\\train\\'
val_root='D:\\lhq\\catdog\\val\\'

train_dataset=train_data(train_root)
val_dataset=val_data(val_root)
print()

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,num_workers=0)


if torch.cuda.is_available()==True:
    #net=resnet(3, 2, False).cuda()
    net = Net().cuda()


cirterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)

writer = SummaryWriter()


for epoch in range(epochs):
    running_loss = 0.0
    train_correct = 0
    train_total = 0
    for i, data in enumerate(train_loader):
        net.train()
        inputs, train_labels = data
        inputs, labels = Variable(inputs.cuda()), Variable(train_labels.cuda())
        optimizer.zero_grad()
        outputs = net(inputs)
        _, train_predicted = torch.max(outputs.data, 1)#输出每一行最大的元素
        train_correct += (train_predicted == labels.data).sum()
        loss = cirterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        train_total += train_labels.size(0)

    val_loss = 0.0
    val_correct = 0
    val_total = 0
    for i, data in enumerate(val_loader):
        net.eval()
        inputs, val_labels = data
        inputs, labels = Variable(inputs.cuda()), Variable(val_labels.cuda())
        outputs = net(inputs)
        _, val_predicted = torch.max(outputs.data, 1)  # 输出每一行最大的元素
        val_correct += (val_predicted == labels.data).sum()
        loss = cirterion(outputs, labels)
        val_loss += loss.item()
        val_total += val_labels.size(0)


    print('train %d epoch loss: %.3f  acc: %.3f val loss: %.3f  val acc: %.3f ' % (
        epoch + 1, running_loss / train_total, 100 * train_correct / train_total,val_loss /val_total,100 * val_correct / val_total))
    with open("log_Lenet.txt", "a") as f:
        f.write('train %d epoch loss: %.3f  acc: %.3f val loss: %.3f  val acc: %.3f \n' % (
        epoch + 1, running_loss / train_total, 100 * train_correct / train_total,val_loss /val_total,100 * val_correct / val_total))


    torch.save(net, 'model_'+str(epoch)+'.pt')
    # ====================== 使用 tensorboard ==================
    writer.add_scalars('train_Loss', {'train': running_loss / train_total},epoch)
    writer.add_scalars('train_Acc', {'train': 100*train_correct / train_total}, epoch)
    writer.add_scalars('val_Loss', {'val': val_loss /val_total}, epoch)
    writer.add_scalars('val_Acc', {'val': 100 * val_correct / val_total}, epoch)

    # =========================================================

torch.save(net, 'model.pt')
