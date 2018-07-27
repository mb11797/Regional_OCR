import split
import os
from torchvision import transforms
import torch
import torch.nn as nn
from model import extractFeatures
import argparse
import numpy as np

# Device Configuration
os.environ["CUDA_VISIBLE_Devices"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(args):
    xtrain, xtest, ytrain, ytest = split.load_dataset()

    print(xtrain[1].shape)
    print(xtest.shape)
    train_size = len(xtrain)
    test_size = len(xtest)
    print(train_size)
    print(test_size)

    data_transforms = transforms.Compose([

#        transforms.Scale(256),
#        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    ])

    train_images = torch.empty(train_size, 3, 224, 224)
    test_images = torch.empty(test_size, 3, 224, 224)

    for i in range(train_size):
        train_images[i] = torch.tensor(xtrain[i]).unsqueeze(0)
        # train_images[i] = data_transforms(train_images[i])

    for i in range(test_size):
        test_images[i] = torch.tensor(xtest[i]).unsqueeze(0)
        # test_images[i] = data_transforms(test_images[i])

#    train_images = torch.tensor(xtrain).unsqueeze(0)
#    test_images = torch.tensor(xtest).unsqueeze(0)

    train_imgs_batches = torch.utils.data.DataLoader(train_images,
                                                    batch_size=args.batch_size,
                                                    shuffle=False,
                                                    num_workers=4,
                                                    drop_last=False,
                                                    timeout=0,
                                                    worker_init_fn=None
                                                    )

    ytrain = torch.from_numpy(ytrain)
    ytest = torch.from_numpy(ytest)

    train_labels_batches = torch.utils.data.DataLoader(ytrain,
                                                       batch_size=args.batch_size,
                                                       shuffle=False,
                                                       num_workers=4,
                                                       drop_last=False,
                                                       timeout=0,
                                                       worker_init_fn=None
                                                       )


    # Build the Model
    extr_features = extractFeatures(args.num_classes).to(device)


    # Loss and Optimizer
    loss_fn = nn.CrossEntropyLoss()
    params = list(extr_features.model.parameters())
    optimizer = torch.optim.Adam(params, lr = args.learning_rate)

    avg_accuracy = 0
    # Train the Model
    for epoch in range(args.num_epochs):
        b = 0
        accuracy = np.zeros((args.batch_size,1))
        for imgs_batch, labels_batch in zip(train_imgs_batches, train_labels_batches):
            imgs_batch = imgs_batch.to(device)
            labels_batch = labels_batch.to(device)
            features = extr_features(imgs_batch)

            b = b + 1
            max_prob, index = torch.max(features, 1)
            print("\nEpoch : ", epoch+1)
            print("Batch : ", b)
            print("Indices : ", index)

            score=0
            for i in range(args.batch_size):
                if index[i] == labels_batch[i]:
                    score += 1

            accuracy[b] = (score / args.batch_size) * 100
            print("Training Accuracy : ", accuracy[b])

            loss = loss_fn(features, labels_batch)
            print("Loss (Cross Entropy loss) : ", loss)

            extr_features.zero_grad()
            loss.backward()
            optimizer.step()
#            with torch.no_grad():
#                for param in extr_features.model.parameters():
#                    param -= args.learning_rate * param.grad
        avg_accuracy = np.mean(accuracy)
        print("Maximum Accuracy in epoch : ", np.max(accuracy))
        print("Maximum Accuracy batch : ", np.argmax(accuracy))
        print("Average Accuracy : ", avg_accuracy)

        if epoch % 10 == 0:
            torch.save(extr_features, "devanagari_recognition_model_res152.pt")
    torch.save(extr_features, "devanagari_recognition_model_res152.pt")




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models/', help='path for saving trained models')

    # Model parameters
    parser.add_argument('--num_classes', type=int, default=58, help='number of classes for classification')

    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=140)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--image_size', type=int, default=224)
    args = parser.parse_args()
    # print(args)
    train(args)


