import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2


labels = pd.read_csv('labels.csv')

X = np.load('img_data.npy')
y = np.load('labels.npy')

print(X.shape)
print(type(X))
print(y.shape)
print(type(y))

def print_image(img_arr):
    # cv2.imshow('image_testing', img_arr)
    # plt.imshow(img_arr)         ###to print in rgb format
    plt.imshow(img_arr, cmap='gray')
    plt.show()
# print(y)
for i in range(289,300):
    print_image(X[i])
    print('label : ' + y[i])
# print(X[4].shape)
cv2.waitKey(0)

















