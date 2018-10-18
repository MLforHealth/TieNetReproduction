from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from data_loader import CxrDataset

if __name__ == '__main__':
    transform = transforms.Compose([
                    transforms.Resize((224,224)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])
    cxr_dataset = CxrDataset(csv_file='./data/Data_Entry_2017.csv',img_dir='cxr/images/',transform=transform)

