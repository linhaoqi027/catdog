import torch.utils.data as data
from PIL import Image
import os
import os.path
from torchvision.datasets import ImageFolder
from torchvision import models, transforms



data_transform = transforms.Compose([
    transforms.Resize((84,84)),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.5,0.5,0.5],std = [0.5,0.5,0.5])
])



# testdata_transform = transforms.Compose([
#     transforms.Resize((84,84)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
# ])


class train_data(data.Dataset):
    def __init__(self, root):
        img_array = ImageFolder(root,transform=data_transform)
        self.data=img_array
    def __getitem__(self, index):
        img,label = self.data[index]
        return img, label
    def __len__(self):
        # 返回图像的数量
        return len(self.data)


class val_data(data.Dataset):
    def __init__(self, root):
        img_array = ImageFolder(root,transform=data_transform)
        self.data=img_array
    def __getitem__(self, index):
        img,label = self.data[index]
        return img, label
    def __len__(self):
        # 返回图像的数量
        return len(self.data)

class test_data(data.Dataset):
    def __init__(self, root):
        img_array = ImageFolder(root,transform=data_transform)
        self.data=img_array
    def __getitem__(self, index):
        img, label= self.data[index]
        return img, label
    def __len__(self):
        # 返回图像的数量
        return len(self.data)