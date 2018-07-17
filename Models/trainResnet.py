import torch
import os
import argparse
import numpy as np
from model import extractFeatures
import torch.nn as nn
from torchvision import transforms


# Device Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):
    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    # data loader
    images = np.load('img_data1.npy')
    labels = np.load('labels1.npy')

    print(type(images))
    print(images.shape)
    # images = torch.from_numpy(images)
    # images.unsqueeze(0)

    # Data Augmentation and normalization for training
    data_transforms = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    ])

    images_tensor = torch.empty(12912,3,36,36)

    for i in range(12912):
        images_tensor[i] = data_transforms(images[i]).unsqueeze(0)
        print(type(images_tensor[i]))
        print(images_tensor[i].shape)

    print(images_tensor[12911])

    # images1 = [data_transforms(image) for image in images]
    # images = data_transforms(images[1]).unsqueeze(0)
    # images = data_transforms(images[1]).unsqueeze(0)
    # print(images.shape)

    dataloader = torch.utils.data.DataLoader(images_tensor,
                                             batch_size=args.batch_size,
                                             shuffle= True,
                                             num_workers=4,
                                             drop_last=False,
                                             timeout=0,
                                             worker_init_fn=None)
    print(len(dataloader))
    # Build the models
    extr_features = extractFeatures(args.num_classes).to(device)

    #Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    params = list(extr_features.linear.parameters()) + list(extr_features.bn.parameters())
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)

    print(images.shape)
    print(images_tensor.shape)
    print(images_tensor.size)

    print(dataloader)

    # Train the model
    total_step = len(dataloader)
    print(total_step)
    j = 0
    l=0
    for epoch in range(args.num_epochs):
        k=0
        l = l + 1
        for i,batch_images in enumerate(dataloader):
            print(batch_images)
            print(type(batch_images))
            print(batch_images.shape)
            print(batch_images.size())

            print('epoch : ', l)
            print('j : ', j+1)
            print('k : ', k+1)

            j = j+1
            k = k+1
            batch_images = batch_images.to(device)
            # print(torch._infer_size(image))
            # print(type(image))
            # print(image.shape)
            #
            # Forward, backward and optimize
            # features = extr_features(batch_images)
            # y_pred = extr_features(image)
            # loss = criterion(y_pred, y)
            # extr_features.zero_grad()
            # loss.backward()
            # optimizer.step()






if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models/', help='path for saving trained models')

    # Model parameters
    parser.add_argument('--num_classes', type=int, default=58, help='number of classes for classification')

    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=48)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--image_size', type=int, default=36)
    args = parser.parse_args()
    print(args)
    main(args)


#
# import cv2
#
# img = cv2.imread('./devanagari-character-dataset/consonants/1/001_01.jpg', 1)
# img = cv2.resize(img, (224,224))
#
# cv2.imshow('img', img)
# cv2.waitKey(0)



