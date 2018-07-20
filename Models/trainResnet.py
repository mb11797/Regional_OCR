import torch
import os
import argparse
import numpy as np
from model import extractFeatures
import torch.nn as nn
from torchvision import transforms
from sklearn.preprocessing import OneHotEncoder

# Device Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):
    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    # torch.set_default_tensor_type('torch.DoubleTensor')

    # data loader
    images = np.load('img_data2.npy')
    labels = np.load('labels2.npy')

    # print(labels.shape)
    y = torch.zeros(labels.shape[0], 58)
    # temp = "Start"
    prev = "C1"
    j = 0
    for i in range(labels.shape[0]):
        if labels[i] == prev:
            y[i][j] = 1
        else:
            j = j+1
            prev = labels[i]
            y[i][j] = 1

    # print(y)
    # print(y.shape)
    # print(y[1])
    # print(y[1].shape)
    # print(y[:, 0])
    # print(y[:, 0].shape)



    #
    #
    # print(y.shape)
    # print(labels)
    #
    # labels = np.unique(labels)
    #
    # print(labels)
    # print(labels.shape)
    # oneHot = OneHotEncoder(categorical_features=[0])
    # y = oneHot.fit_transform(labels)
    # print(type(y))


    # print(type(images))
    # print(images.shape)
    # images = torch.from_numpy(images)
    # images.unsqueeze(0)

    # Data Augmentation and normalization for training
    data_transforms = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    ])

    images_tensor = torch.empty(12912,3,224,224)

    for i in range(12912):
        images_tensor[i] = data_transforms(images[i]).unsqueeze(0)
        # print(type(images_tensor[i]))
        # print(images_tensor[i].shape)
    # print(images_tensor.shape)
    # print(images_tensor[12911])

    # images1 = [data_transforms(image) for image in images]
    # images = data_transforms(images[1]).unsqueeze(0)
    # images = data_transforms(images[1]).unsqueeze(0)
    # print(images.shape)

    # dataset_tensors = np.concatenate((images_tensor, y), axis=1)


    dataloader = torch.utils.data.DataLoader(images_tensor,
                                             batch_size=args.batch_size,
                                             shuffle= False,
                                             num_workers=4,
                                             drop_last=False,
                                             timeout=0,
                                             worker_init_fn=None)
    print('dataloader len : ', len(dataloader))
    # Build the models
    extr_features = extractFeatures(args.num_classes).to(device)

    labelloader = torch.utils.data.DataLoader(y,
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              num_workers=4,
                                              drop_last=False,
                                              timeout=0,
                                              worker_init_fn=None)

    #Loss and Optimizer
    # criterion = nn.CrossEntropyLoss().to(device)
    criterion = nn.MSELoss()
    # params = list(extr_features.linear.parameters()) + list(extr_features.bn.parameters())
    params = list(extr_features.linear.parameters())
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)

    # print(images.shape)
    # print(images_tensor.shape)
    # print(images_tensor.size)
    #
    # print(dataloader)

    # Train the model
    total_step = len(dataloader)
    print('batch size : ', total_step)
    print('label_batch_size : ', len(labelloader))
    j = 0
    l=0
    for epoch in range(args.num_epochs):
        k=0
        l = l + 1
        for batch_images,y in zip(dataloader,labelloader):
            # print(batch_images)
            # print(type(batch_images))
            # print(batch_images.shape)
            # print(batch_images.size())

            print('epoch : ', l)
            print('j : ', j+1)
            print('k : ', k+1)

            j = j+1
            k = k+1
            batch_images = batch_images.to(device)
            # y = torch.dtype(torch)
            y = y.to(device)
            # print(features)
            # print(torch._infer_size(image))
            # print(type(image))
            # print(image.shape)
            #
            # Forward, backward and optimize
            features = extr_features(batch_images)
            print('features shape : ', features.shape)
            print('Single image features extracted : ', features[1])
            print('Single image features extracted : ', type(features[1]))
            print('Single image features extracted : ', features[1].shape)
            max_prob, index = torch.max(features, 1)
            print('max_prob : ', max_prob)
            print('max_prob len : ', len(max_prob))
            print('max_prob type : ', type(max_prob))
            print('indices : ', index)
            print('indices len : ', len(index))
            print('indices type : ', type(index))

            y_pred = torch.zeros(269,58)

            loss = 0
            for i in range(269):
                # y_pred[i][index[i]] = 1
                loss += criterion(features[i], y[i])

            print('Cross Entropy loss for ', k, 'th batch ', loss)

            # y_pred = y_pred.to(device)
            # print('y_pred : ', y_pred)
            # print('y : ', y)
            # print('y shape : ', y.shape)

            # print(y_pred.long().type())
            # print(y.long().type())
            # print(y_pred.type())
            # print(y.type())

            # print(max_prob)
            # y_pred = extr_features(image)
            # loss = criterion(y_pred, y)
            # print('Mean Squared Error loss : ', loss)
            extr_features.zero_grad()
            loss.backward()
            optimizer.step()






if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models/', help='path for saving trained models')

    # Model parameters
    parser.add_argument('--num_classes', type=int, default=58, help='number of classes for classification')

    parser.add_argument('--num_epochs', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=269)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--image_size', type=int, default=112)
    args = parser.parse_args()
    # print(args)
    main(args)



# import cv2
#
# img = cv2.imread('./devanagari-character-dataset/consonants/1/001_01.jpg', 1)
# # img = cv2.resize(img, (224,224))
# # img = cv2.resize(img, (112, 112))
# # img = cv2.resize(img, (72, 72))
# img = cv2.resize(img, (36, 36))
#
# print(type(img))
#
# cv2.imshow('img', img)
# cv2.waitKey(0)
#
#

