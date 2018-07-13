import torch
import os
import argparse
import numpy as np
from model import extractFeatures
import torch.nn as nn


# Device Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):
    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    # data loader
    images = np.load('img_data1.npy')
    labels = np.load('labels1.npy')

    images = torch.from_numpy(images)
    images.unsqueeze(0)

    # Build the models
    extr_features = extractFeatures(args.num_classes)

    #Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    params = list(extr_features.linear.parameters()) + list(extr_features.bn.parameters())
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)

    print(images.shape)

    # Train the model
    total_step = len(images)
    for epoch in range(args.num_epochs):
        for image in images:
            image = image.to(device)
            # print(torch._infer_size(image))
            # print(type(image))
            # print(image.shape)

            # Forward, backward and optimize
            # features = extr_features(image)
            y_pred = extr_features(image)
            loss = criterion(y_pred, y)
            extr_features.zero_grad()
            loss.backward()
            optimizer.step()













if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models/', help='path for saving trained models')

    # Model parameters
    parser.add_argument('--num_classes', type=int, default=58, help='number of classes for classification')

    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--image_size', type=int, default=36)
    args = parser.parse_args()
    print(args)
    main(args)











