import os, sys, pickle
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.utils.data as Data
from torch.utils.data import DataLoader, Dataset

from PIL import Image


def make_dict():
    with open('training data dic.txt', "r") as fp:
        lines = fp.readlines()
    word_dict = {} #key:中文字 value:數字
    num = 1
    for line in lines:
        word_dict[line.strip()] = num
        num += 1
    return word_dict



def load_file(path, word_dict): # load pic and save as pickle
    dirs = os.listdir(path)
    data_x = []
    data_y = []
    for i, d in enumerate(dirs):
        file_name = f'{path}/{d}'
        word = d.split('_')
        file_value =  word_dict.get(word[1][0], 0)
        data_x.append(file_name)
        data_y.append(file_value)
    # with open(f'{path}/dataset.pickle', 'wb') as file:
    #     pickle.dump([data_x, data_y], file)
    return data_x, data_y


class MyDataset(Dataset):
    def __init__(self, data, label, transform=None):
        self.data = data
        self.label = label
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        file_name = self.data[index]
        input_image = Image.open(self.data[index]).convert('RGB')
        x = self.transform(input_image)
        y = y = torch.tensor(self.label[index], dtype = torch.long)
        return x, y

# Data Augmentation 