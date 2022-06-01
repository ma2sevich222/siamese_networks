import torch.nn as nn
import torch
import torch.nn.functional as F


# create the Siamese Neural Network
class SiameseNetwork(nn.Module):

    def __init__(self, embedding_dim=2):
        super(SiameseNetwork, self).__init__()

        # Setting up the Sequential of CNN Layers
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 156, kernel_size=2, stride=1),
            nn.LeakyReLU(2, inplace=True),
            #nn.ELU(alpha=1.0, inplace=True)
            #nn.Sigmoid()

            #nn.MaxPool2d(3, stride=1),

            nn.Conv2d(256, 156, kernel_size=2, stride=1),
            nn.LeakyReLU(2, inplace=True),
            #nn.MaxPool2d(2, stride=1),

            nn.Conv2d(156, 156, kernel_size=2, stride=1),
            nn.LeakyReLU(2, inplace=True)

        )

        # Setting up the Fully Connected Layers
        self.fc1 = nn.Sequential(
            nn.LazyLinear(156),
            nn.LeakyReLU(2, inplace=True),
            #nn.ELU(alpha=1.0, inplace=True)
            #nn.Sigmoid()

            nn.Linear(156, 156),
            nn.LeakyReLU(2, inplace=True),

            nn.Linear(156, embedding_dim)
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


class shotSiameseNetwork(nn.Module):

    def __init__(self, embedding_dim=2):
        super(shotSiameseNetwork, self).__init__()

        # Setting up the Sequential of CNN Layers
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 256, kernel_size=2, stride=1),
            #nn.LeakyReLU(2, inplace=True),
            #nn.ELU(alpha=1.0, inplace=True)
            nn.Sigmoid()

            #nn.MaxPool2d(3, stride=1),

            #nn.Conv2d(256, 256, kernel_size=2, stride=1),
            #nn.LeakyReLU(2, inplace=True),
            #nn.MaxPool2d(2, stride=1),

            #nn.Conv2d(256, 256, kernel_size=2, stride=1),
            #nn.LeakyReLU(2, inplace=True)

        )

        # Setting up the Fully Connected Layers
        self.fc1 = nn.Sequential(
            nn.LazyLinear(embedding_dim),
            nn.LeakyReLU(2, inplace=True),
            #nn.ELU(alpha=1.0, inplace=True)
            #nn.Sigmoid()

            #nn.Linear(256, 256),
            #nn.LeakyReLU(2, inplace=True),

            #nn.Linear(256, embedding_dim)
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


class SiameseNetwork_extend(nn.Module):

    def __init__(self, base_model, embedding_dim=2):
        super(SiameseNetwork_extend, self).__init__()

        self.cnn1 = nn.Sequential(
            nn.ConvTranspose2d(1, 3, kernel_size=10, stride=1),
            nn.ReLU(inplace=True))
        self.model = base_model

        # Setting up the Fully Connected Layers
        self.fc1 = nn.Sequential(
            nn.LazyLinear(256),
            nn.ReLU(inplace=True),

            nn.Linear(256, 126),
            nn.ReLU(inplace=True),

            nn.Linear(126, embedding_dim)
        )

    def forward_once(self, x):
        # This function will be called for both images
        # It's output is used to determine the similiarity
        output = self.cnn1(x)
        output = self.model(output)
        output = output.contiguous().view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        # In this function we pass in both images and obtain both vectors
        # which are returned
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)


        return output1, output2,


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
      # Calculate the euclidian distance and calculate the contrastive loss
      euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)

      loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                    (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))


      return loss_contrastive


class SiameseNetwork_extend_triplet(nn.Module):

    def __init__(self, base_model, embedding_dim=2):
        super(SiameseNetwork_extend_triplet, self).__init__()

        self.cnn1 = nn.Sequential(
            nn.ConvTranspose2d(1, 3, kernel_size=10, stride=1),
            #nn.ReLU(inplace=True))
            nn.Sigmoid())
        self.model = base_model

        # Setting up the Fully Connected Layers
        self.fc1 = nn.Sequential(
            nn.LazyLinear(256),
            #nn.ReLU(inplace=True),
            nn.Sigmoid(),

            nn.Linear(256, 126),
            #nn.ReLU(inplace=True),
            nn.Sigmoid(),

            nn.Linear(126, embedding_dim)
        )

    def forward_once(self, x):
        # This function will be called for both images
        # It's output is used to determine the similiarity
        output = self.cnn1(x)
        output = self.model(output)
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


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
      # Calculate the euclidian distance and calculate the contrastive loss
      euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)

      loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                    (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))


      return loss_contrastive