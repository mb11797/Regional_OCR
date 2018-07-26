import numpy as np
from sklearn.model_selection import train_test_split


def load_dataset():
    images = np.load("img_data4.npy")
    labels = np.load("labels4.npy")

    x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.1, random_state=42)

    print(type(x_train), type(x_test), type(y_train), type(y_test))
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

    print(y_train)

    return x_train, x_test, y_train, y_test

load_dataset()
