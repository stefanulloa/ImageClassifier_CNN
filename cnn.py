
import torch
import torch.nn as nn
import torch.nn.functional as F

'model architecture representation on tutorial webpage'

class NeuralNetwork(nn.Module):
    'pytorch neural networks must be subclasses of nn.Module'

    def __init__(self):
        super(NeuralNetwork, self).__init__()

        'attributes with updatable data (w, b) such as filter or transformations in init'
        self.con1 = nn.Conv2d(1, 6, 5) #1 input channel, 6 output channels
        self.conv2 = nn.Conv2d(6, 16, 5) #third parameter is kernel size, if only one entry y, infers y x y

        self.fc1 = nn.Linear(16*5*5, 120) #linear transformation
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        'in forward set functional methods'
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2)) #max pool on 2x2
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_1D_features(x)) #resize matrix as vector for linear transformation
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    'get matrix dimension resized as as array (1, number)'
    def num_1D_features(self, x):
        size = x.size()[1:] #ignore batch dim
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
