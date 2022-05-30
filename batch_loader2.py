# Author Hao Wang & Ziyu Zhan
# OAM spectrum detection based on hybrid optoeletronic neural network
#  ============================
import scipy.io
import os
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch

class TrainDataset(Dataset):
    def __init__(self, path_dir,  transform):
        self.path_dir = path_dir
        self.transform = transform
        self.image = os.listdir(self.path_dir)
    def __len__(self):
        return len(self.image)
    def __getitem__(self, item):
        image_item = self.image[item]
        image_path = os.path.join(self.path_dir, image_item)
        mat_vars = scipy.io.loadmat(image_path)
        image = mat_vars['E']
        label = mat_vars['OAM_s']
        image = torch.tensor(image)
        # for size match
        image_Re = torch.abs(image) * torch.cos(torch.angle(image))
        image_Re = torch.unsqueeze(image_Re, dim=0).float()
        image_Re = torch.unsqueeze(image_Re, dim=1).float()
        image_Re = F.interpolate(image_Re, size=[200, 200], mode='nearest')
        image_Re = torch.squeeze(image_Re)
        image_Im = torch.abs(image) * torch.sin(torch.angle(image))
        image_Im = torch.unsqueeze(image_Im, dim=0).float()
        image_Im = torch.unsqueeze(image_Im, dim=1).float()
        image_Im = F.interpolate(image_Im, size=[200, 200], mode='nearest')
        image_Im = torch.squeeze(image_Im)
        image1 = image_Re + 1.0j * image_Im
        image1 = torch.squeeze(image1)
        image1 = image1.detach().numpy()
        return image1, label

class ValDataset(Dataset):
    def __init__(self, path_dir,  transform):
        self.path_dir = path_dir
        self.transform = transform
        self.image = os.listdir(self.path_dir)
    def __len__(self):
        return len(self.image)
    def __getitem__(self, item):
        image_item = self.image[item]
        image_path = os.path.join(self.path_dir, image_item)
        mat_vars = scipy.io.loadmat(image_path)
        image = mat_vars['E']
        label = mat_vars['OAM_s']
        return image, label

class TestDataset(Dataset):
    def __init__(self, path_dir, transform):
        self.path_dir = path_dir
        self.transform = transform
        self.image = os.listdir(self.path_dir)
    def __len__(self):
        return len(self.image)
    def __getitem__(self, item):
        image_item = self.image[item]
        image_path = os.path.join(self.path_dir, image_item)
        mat_vars = scipy.io.loadmat(image_path)
        image = mat_vars['E']
        label = mat_vars['OAM_s']
        return image, label
