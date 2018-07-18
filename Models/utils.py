import os
import numpy as np
import pandas as pd
import scipy.ndimage
import cv2


# def readFiles(path):
#     # os.walk(path)
#     img_arr_data = []
#     dir_names = []
#     i = 0
#     for root, dirnames, filenames in os.walk(path):
#         # print(1)
#         # print(path)
#         # print(dirnames)
#         # print(dirnames[0])
#         # print(filenames)
#
#         for filename in filenames:
#             # print(dirnames)
#             # print(root)
#             # print(type(root))
#             dir_names.append(dirnames)
#             path = os.path.join(root,filename)
#             img_arr = scipy.ndimage.imread(path, flatten=True)
#             # print(path)
#             # print(img_arr.shape)
#             # i += 1
#             # print(i)
#             # np.append([img_arr_data], img_arr, axis = 1)
#             # print(img_arr_data.shape)
#             img_arr_data.append(img_arr)
#             print(len(img_arr_data))
#
#     # print(img_arr_data[12911].shape)
#
#
# # def dataFrameFromDirectory(path, classification):
# #     X = []
# #     y = []
# #     for filename, message in readFiles(path):
# print('Hello')
#
# readFiles("/home/surveillance6/PycharmProjects/Regional_OCR/Models/devanagari-character-dataset/consonants")
#
# print('Hi')

def read_n_save_Files(path, cat):
    # img_arr_data = np.ndarray((12912,36,36), dtype = np.float32)
    hin_char = [u"\u0915", u"\u0916", u"\u0917", u"\u0918", u"\u0919", u"\u091A", u"\u091B", u"\u091C", u"\u091D", u"\u091E",u"\u091F", u"\u0920", u"\u0921", u"\u0922", u"\u0923", u"\u0924", u"\u0925", u"\u0926", u"\u0927", u"\u0928", u"\u092A", u"\u092B", u"\u092C", u"\u092D", u"\u092E", u"\u092F", u"\u0930", u"\u0932", u"\u0935", u"\u0936", u"\u0937", u"\u0938", u"\u0939", u"\u0915\u094d\u0937", u"\u0924\u094D\u0930", u"\u091C\u094D\u091E"]
    img_arr_data = []
    # print(img_arr_data)
    # print(img_arr_data.shape)
    # labels = np.array([])
    labels = []
    # labels = np.array((12912,1))
    for ctg in cat:
        if ctg == 'consonants':
            path1 = path + '/' + ctg
            for i in range(1,37):
                path2 = path1 + '/' + str(i)
                for root,dirnames,filenames in os.walk(path2):
                    for filename in filenames:
                        # img_arr = scipy.ndimage.imread(root+'/'+filename, flatten=False).flatten()
                        # img_arr = scipy.ndimage.imread(root+'/'+filename, flatten=False, mode='RGB')
                        img_arr = cv2.imread(root+'/'+filename, 1)
                        img_arr = cv2.resize(img_arr, (224, 224))
                        print(img_arr.shape)
                        print(type(img_arr))
                        # print(img_arr)
                        # print(img_arr.shape)
                        # img_arr_data[img_arr] = 'C' + str(i)
                        # img_arr_data = np.vstack((img_arr_data, img_arr))
                        # print(img_arr_data)
                        img_arr_data.append(img_arr)
                        labels.append('C' + str(i))


        if ctg == 'numerals':
            path1 = path + '/' + ctg
            for i in range(0,10):
                path2 = path1 + '/' + str(i)
                for root,dirnames,filenames in os.walk(path2):
                    for filename in filenames:
                        # img_arr = scipy.ndimage.imread(root+'/'+filename, flatten=True).flatten()
                        # img_arr = scipy.ndimage.imread(root+'/'+filename, flatten=False, mode='RGB')
                        img_arr = cv2.imread(root+'/'+filename, 1)
                        img_arr = cv2.resize(img_arr, (224, 224))
                        print(img_arr.shape)
                        # img_arr_data[img_arr] = 'N' + str(i)
                        img_arr_data.append(img_arr)
                        # img_arr_data = np.vtsack((img_arr_data, img_arr))
                        labels.append('N' + str(i))

        if ctg == 'vowels':
            path1 = path + '/' + ctg
            for i in range(1,13):
                path2 = path1 + '/' + str(i)
                for root,dirnames,filenames in os.walk(path2):
                    for filename in filenames:
                        # img_arr = scipy.ndimage.imread(root+'/'+filename, flatten=True).reshape(36*36)
                        # img_arr = scipy.ndimage.imread(root+'/'+filename, flatten=False, mode='RGB')
                        img_arr = cv2.imread(root+'/'+filename, 1)
                        img_arr = cv2.resize(img_arr, (224, 224))
                        print(img_arr.shape)
                        # img_arr_data[img_arr] = 'V' + str(i)
                        img_arr_data.append(img_arr)
                        # img_arr_data = np.vstack((img_arr_data, img_arr))
                        labels.append('V' + str(i))

    # img_arr_data = np.ndarray(img_arr_data)
    # labels = np.ndarray(labels)
    # print(len(img_arr_data))
    # print(len(labels))
    img_arr_data = np.asarray(img_arr_data)
    labels = np.asarray(labels)

    print(img_arr_data)
    print(labels)
    print(img_arr_data.shape)
    print(labels.shape)
    np.save('/home/surveillance6/PycharmProjects/Regional_OCR/Models/img_data2.npy', img_arr_data)
    np.save('/home/surveillance6/PycharmProjects/Regional_OCR/Models/labels2.npy', labels)



def arraysFromDirectory(path):
    # X = []
    # y = []

    # path_con = path + '/consonants'
    # path_num = path + '/numerals'
    # path_vow = path + '/vowels'
    categ = ['consonants', 'numerals', 'vowels']
    read_n_save_Files(path, categ)
    # print(X.shape)
    # print(y.shape)





arraysFromDirectory("/home/surveillance6/PycharmProjects/Regional_OCR/Models/devanagari-character-dataset")




















