import os
import numpy as np
import pandas as pd
import scipy.ndimage



def readFiles(path):
    # os.walk(path)
    img_arr_data = np.ndarray([])
    i = 0
    for root, dirnames, filenames in os.walk(path):
        # print(1)
        # print(root)
        # print(dirnames)
        # print(filenames)
        for filename in filenames:
            path = os.path.join(root,filename)
            # img_arr = scipy.ndimage.imread(path, flatten=True)
            img_arr = scipy.ndimage.imread(path, flatten=True)
            print(path)
            print(img_arr.shape)
            i += 1
            print(i)


# def dataFrameFromDirectory(path, classification):
#     X = []
#     y = []
#     for filename, message in readFiles(path):
print('Hello')

readFiles("/home/surveillance6/PycharmProjects/Regional_OCR/Models/devanagari-character-dataset")

print('Hi')























