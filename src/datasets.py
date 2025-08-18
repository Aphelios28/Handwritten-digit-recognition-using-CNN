import torch
import torchvision
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import Dataset

class NumberMNIST(Dataset):
    def __init__(self, root = 'data', download = True, transform = transforms.ToTensor(), target_transform = None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
    def load_data_train(self):
        train_data = datasets.MNIST(root=self.root, train = True, download= self.download, transform=self.transform,
                                           target_transform=self.target_transform)
        return train_data
    def load_data_test(self):
        test_data = datasets.MNIST(root=self.root, train = False, download= self.download, transform=self.transform,
                                           target_transform=self.target_transform)
        return test_data
    @staticmethod
    def class_name(self):
        return self.load_data_train().classes
