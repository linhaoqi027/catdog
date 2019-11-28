

import os

import torch
from torchvision import models, transforms
from torch.autograd import Variable
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from dataset import test_data
import pandas as pd
from PIL import Image


mysub = pd.read_csv('mysub.csv')

model = torch.load('model_122.pt')
model.eval()

data_transform = transforms.Compose([
    transforms.Resize((84,84)),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.5,0.5,0.5],std = [0.5,0.5,0.5])
])

my_predict=[]

for i in range(5000):
    image = Image.open('D:\\lhq\\catdog\\test\\pic\\'+str(i)+'.jpg')
    image_transformed = data_transform(image)
    images = image_transformed.unsqueeze(0)
    images= Variable(images.cuda())
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    predicted=predicted.cpu().numpy()
    my_predict.extend(predicted)

    
print(my_predict)
mysub.label=my_predict
mysub.to_csv('mysub.csv')