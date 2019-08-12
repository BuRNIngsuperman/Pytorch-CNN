import torch
import argparse
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--epochs','-e',default=1000,type=int,help='numbers of epochs top train')
parser.add_argument('--t',action='store_true',default=False)
args = parser.parse_args()
print(args.t)
print(args.epochs)
# train_transforms = transforms.Compose([
#     transforms.Resize(244),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
# ])
#
# train_set = torchvision.datasets.CIFAR10(root='./data',train=True,download=False,transform=train_transforms)
# train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2 )
#
# def imshow(img):
#     img = img / 2 + 0.5
#     npimg =img.numpy()
#     plt.imshow(np.transpose(npimg))
#
# print(train_set)