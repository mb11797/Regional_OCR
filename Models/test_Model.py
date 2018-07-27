import argparse
import split
import numpy as np
import os
import torch
import torch.utils.data

# Device Configuration
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test(args):
    xtrain, xtest, ytrain, ytest = split.load_dataset()

    test_size = len(xtest)
    print(test_size)

    test_images = torch.empty(test_size, 3, 224, 224)

    for i in range(test_size):
        test_images[i] = torch.tensor(xtest[i]).unsqueeze(0)

    # test_images = test_images.to(device)

    # test_images = test_images.unsqueeze(0).to(device)
    # print(test_images.shape)


    test_imgs_batches = torch.utils.data.DataLoader(test_images,
                                                    batch_size=args.batch_size,
                                                    shuffle=False,
                                                    num_workers=4,
                                                    drop_last=False,
                                                    timeout=0,
                                                    worker_init_fn=None
                                                    )

    ytest = torch.from_numpy(ytest)

    test_labels_batches = torch.utils.data.DataLoader(ytest,
                                                       batch_size=args.batch_size,
                                                       shuffle=False,
                                                       num_workers=4,
                                                       drop_last=False,
                                                       timeout=0,
                                                       worker_init_fn=None
                                                       )



    model = torch.load("devanagari_recognition_model_res152.pt").to(device)
    # model = model.to(device)

    for imgs_batches, labels_batch in zip(test_imgs_batches, test_labels_batches):

        imgs_batches = imgs_batches.to(device)
        labels_batch = labels_batch.to(device)

        features = model(imgs_batches)

        max_prob, index = torch.max(features, 1)

        score = 0
        for i in range(args.batch_size):
            if index[i] == labels_batch[i]:
                score += 1

        accuracy = (score / args.batch_size) * 100
        print("Test Accuracy : ", accuracy)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=323)

    args = parser.parse_args()
    # print(args)
    test(args)
