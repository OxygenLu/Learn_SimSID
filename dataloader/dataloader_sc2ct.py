import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import torch.optim as optim
import os
from matplotlib import pyplot as plt
import random


class Sc2ct(torch.utils.data.Dataset):
    def __init__(self, root, train=True, img_size=(256, 256), normalize=False, normalize_tanh=False, 
                 enable_transform=True, full=True, positive_ratio=1.0):

        self.data = []
        self.train = train
        self.root = root
        self.normalize = normalize
        self.img_size = img_size
        self.mean = 0.1307
        self.std = 0.3081
        self.full = full
        self.positive_ratio = positive_ratio
        self.total = 1

        # 训练时候
        if train:
            if enable_transform:
                self.transforms = [
                    transforms.RandomAffine(0, translate=(0.05, 0.05), scale=(0.95,1.05)),
                    transforms.ToTensor()
                ]
            else:
                self.transforms = [transforms.ToTensor()]
        # 没有训练
        else:
            self.transforms = [transforms.ToTensor()]

        if normalize_tanh:
            self.transforms.append(transforms.Normalize((0.5,), (0.5,))) 
        self.transforms = transforms.Compose(self.transforms)

        self.load_data()

    def load_data(self):
        self.fnames = list()

        # 训练
        if self.train:
            pos_items = os.listdir(os.path.join(self.root, 'pos'))
            neg_items = os.listdir(os.path.join(self.root, 'neg'))

            num_pos = len(pos_items)
            num_neg = len(neg_items)
           
            for item in pos_items[:num_pos]:
                image = Image.open(os.path.join(self.root, 'pos', item)).resize(self.img_size)
                self.data.append((image, 0))
                self.fnames.append(item)
            for item in neg_items[:num_neg]:
                image = Image.open(os.path.join(self.root, 'neg', item)).resize(self.img_size)
                self.data.append((image, 1))
                self.fnames.append(item)
            self.positive_ratio = len(self.data)/(len(pos_items)+len(neg_items))
            print('%d data loaded from: %s, positive rate %.2f' % (len(self.data), self.root, self.positive_ratio))
        
        # 没训练
        if not self.train:
            items_nor = os.listdir(os.path.join(self.root, 'pos')) # 正样本

            # 遍历正样本
            for idx, item in enumerate(items_nor):
                
                if not self.full and idx > 9:
                    break
                self.data.append((Image.open(os.path.join(self.root, 'pos', item)).resize(self.img_size), 0))
            self.fnames += items_nor

            items_pn = os.listdir(os.path.join(self.root, 'neg'))
            for idx, item in enumerate(items_pn):
                if not self.full and idx > 9:
                    break
                self.data.append((Image.open(os.path.join(self.root, 'neg', item)).resize(self.img_size), 1))
            self.fnames += items_pn
            self.positive_ratio = len(items_nor)/(len(items_pn)+len(items_nor))
            print('%d data loaded from: %s, positive rate %.2f' % (len(self.data), self.root, self.positive_ratio))
    

    def __getitem__(self, index):
        img, label = self.data[index]

        img = self.transforms(img)[[0]]
        if self.normalize:
            img -= self.mean
            img /= self.std
        return img, (torch.zeros((1,)) + label).long()

    def __len__(self):
        return len(self.data)

if __name__ == '__main__':
    dataset = Sc2ct('/media/administrator/1305D8BDB8D46DEE/jhu/ZhangLabData/CellData/chest_xray/val', train=False)
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    for i, (img, label) in enumerate(trainloader):
        if img.shape[1] == 3:
            plt.imshow(img[0,1], cmap='gray')
            plt.show()
        break