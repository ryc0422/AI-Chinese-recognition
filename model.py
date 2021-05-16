import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class VGG16(nn.Module):
    def __init__(self, N_CLASS):
        super(VGG16, self).__init__()

        ## Convolutuin layers
        self.conv11 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        self.conv21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.conv31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv33 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.conv41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv43 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.conv51 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv52 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv53 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        ## Maxpooling
        self.Maxpool = nn.MaxPool2d(2, 2)
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((7, 7))

        ## Activation Function
        self.ReLu = nn.ReLU(inplace=False)
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((7, 7))

        ## Dropout
        self.dropout = nn.Dropout(0.5)

        ## Fully connected layers:
        self.fc1 = nn.Linear(512*7*7, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, N_CLASS)
    
    def forward(self, x, training=True):
        x = self.ReLu(self.conv11(x))
        x = self.ReLu(self.conv12(x))
        x = self.Maxpool(x)
        x = self.ReLu(self.conv21(x))
        x = self.ReLu(self.conv22(x))
        x = self.Maxpool(x)        
        x = self.ReLu(self.conv31(x))
        x = self.ReLu(self.conv32(x))
        x = self.ReLu(self.conv32(x))
        x = self.Maxpool(x)
        x = self.ReLu(self.conv41(x))
        x = self.ReLu(self.conv42(x))
        x = self.ReLu(self.conv42(x))
        x = self.Maxpool(x)
        x = self.ReLu(self.conv51(x))
        x = self.ReLu(self.conv52(x))
        x = self.ReLu(self.conv52(x))
        x = self.Maxpool(x)
        x = self.AdaptiveAvgPool(x)
        x = x.view(-1, 512*7*7)
        x = self.ReLu(self.fc1(x))
        x = self.dropout(x)
        x = self.ReLu(self.fc2(x))
        x = self.dropout(x)
        x = self.ReLu(self.fc3(x))
        x = F.softmax(x, dim=1)

        return x

    