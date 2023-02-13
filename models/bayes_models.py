import torch.nn as nn
import torch
import torch.nn.functional as F
import torchbnn as bnn


class bayes_shotSiameseNetwork(nn.Module):
    def __init__(self, kernel, strd, conv_chnls, shp_aftr_conv):
        super(bayes_shotSiameseNetwork, self).__init__()

        # Setting up the Sequential of CNN Layers
        self.cnn1 = nn.Sequential(
            bnn.BayesConv2d(
                prior_mu=0,
                prior_sigma=0.1,
                in_channels=1,
                out_channels=conv_chnls,
                kernel_size=kernel,
                stride=strd,
            ),
            # nn.LeakyReLU(2, inplace=True),
            # nn.ELU(alpha=1.0, inplace=True)
            nn.ReLU(),
            nn.Flatten()  # nn.MaxPool2d(3, stride=1),
            # nn.Conv2d(256, 256, kernel_size=2, stride=1),
            # nn.LeakyReLU(2, inplace=True),
            # nn.MaxPool2d(2, stride=1),
            # nn.Conv2d(256, 256, kernel_size=2, stride=1),
            # nn.LeakyReLU(2, inplace=True)
        )

        # Setting up the Fully Connected Layers
        self.fc1 = nn.Sequential(
            bnn.BayesLinear(
                prior_mu=0, prior_sigma=0.1, in_features=shp_aftr_conv, out_features=100
            )
        )

    def forward_once(self, x):
        # This function will be called for both images
        # It's output is used to determine the similiarity
        output = self.cnn1(x)
        output = output.contiguous().view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2, input3):
        # In this function we pass in both images and obtain both vectors
        # which are returned
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        output3 = self.forward_once(input3)

        return output1, output2, output3
