
import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image
import PIL.ImageOps

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torchvision.utils
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch import optim
import torch.nn.functional as F




# create the Siamese Neural Network
class SiameseNetwork(nn.Module):

    def __init__(self, dim=2):
        super(SiameseNetwork, self).__init__()

        # Setting up the Sequential of CNN Layers
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=2 ,stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=1),

            nn.Conv2d(96, 256, kernel_size=2, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=1),

            nn.Conv2d(256, 384, kernel_size=1 ,stride=1),
            nn.ReLU(inplace=True),

        )

        # Setting up the Fully Connected Layers
        self.fc1 = nn.Sequential(
            nn.Linear(30720, 256),
            nn.ReLU(inplace=True),

            nn.Linear(256, 126),
            nn.ReLU(inplace=True),

            nn.Linear(126, dim)
        )

    def forward_once(self, x):
        # This function will be called for both images
        # It's output is used to determine the similiarity
        output = self.cnn1(x)
        output = output.contiguous().view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        # In this function we pass in both images and obtain both vectors
        # which are returned
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        return output1, output2