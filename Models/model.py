import torch
import torch.nn as nn
import torchvision.models as models


class extractFeatures(nn.Module):
    def __init__(self, num_classes):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(extractFeatures, self).__init__()
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]          # delete the last fc layer
        self.resnet = nn.Sequential(*modules)
        # print('resnet.fc.in_features : ', resnet.fc.in_features)
        # print('resnet.fc shape  :', resnet.fc.shape)       # error - linear object has no shape attribute
        # self.linear = nn.Linear(resnet.fc.in_features, num_classes)
        # self.softmax = nn.Softmax()
        # self.bn = nn.BatchNorm1d(num_classes, momentum=0.01)
        # self.softmax = nn.Softmax()
        self.model = nn.Sequential(
            # nn.Sequential(*modules),
            # nn.Linear(resnet.fc.in_features, num_classes),
            # nn.Linear(2048, num_classes)
            # nn.Softmax()
            nn.Linear(resnet.fc.in_features, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_classes),
            nn.Softmax()
        )



    def forward(self, images):
        """Extract feature vectors from input images"""
        # print(1)
        with torch.no_grad():
            # features = self.resnet(images)
            features = self.resnet(images)
            # print(features.shape)
            # features = self.linear(features)
            # features = self.softmax(features)
        # features = self.model(features)
        # features = self.linear

        # print(1)
        features = features.reshape(features.size(0), -1)
        # print('Reshape : ', features.shape)
        features = self.model(features)
        # features = self.bn(self.linear(features))
        return features







