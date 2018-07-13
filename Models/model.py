import torch
import torch.nn as nn
import torchvision.models as models


class extractFeatures(nn.Module):
    def __init__(self, num_classes):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(extractFeatures, self).__init__()
        resnet = models.resnet152(preTrained=True)
        modules = list(resnet.children())[:-1]          # delete the last fc layer
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.linear(resnet.fc.in_features, num_classes)
        self.bn = nn.BatchNorm1d(num_classes, momentum=0.01)

    def forward(selfself, images):
        """Extract feature vectors from input images"""
        with torch.no_grad():
            features = self.resnet(images)
        features = features.reshape(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features






