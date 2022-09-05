# Coding :  UTF-8
# Author : Ziyu Zhan & Hao Wang
import scipy.io
import os
from torch.utils.data import Dataset


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
        return image, label


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
