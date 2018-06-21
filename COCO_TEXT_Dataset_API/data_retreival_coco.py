import coco_text
import json



ct = coco_text.COCO_Text('COCO_Text.json')
ct.info()


# Some image retreival...To get a list of all those image ids from training set where at least one text instance is  there, i.e, all legible images and are machine printed
imgs = ct.getImgIds(imgIds=ct.train, catIds=[('legibility','legible'), ('class','machine printed')])
# imgs = ct.getImgIds(imgIds=ct.val, catIds=[('legibility','legible'), ('class','machine printed')])
# imgs = ct.getImgIds(imgIds=ct.train, catIds=[('legibility','legible')])
# imgs = ct.getImgIds(imgIds=ct.train)
# imgs = ct.getImgIds(imgIds=ct.val)
# imgs = ct.getImgIds(imgIds=ct.test)


print(type(imgs))
print(imgs)
print(len(imgs))

# Annotation Ids retreival form the validation set that are legible, machine printed and have an area between 0 and 200 pixel
anns = ct.getAnnIds(imgIds=ct.val, catIds=[('legibility','legible'),('class','machine printed')], areaRng=[0,200])
# anns = ct.getAnnIds(imgIds=ct.train, catIds=[('legibility','legible'),('class','machine printed')], areaRng=[0,200])
# anns = ct.getAnnIds(imgIds=ct.train, catIds=[('legibility','legible'),('class','machine printed')])



# print(type(anns))
# print(anns)
# print(len(anns))


#####   VISUALIZE COCO TEXT ANNOTATIONS
dataDir = '/media/surveillance6/SSVB2'
dataType = 'train2014'


import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
pylab.rcParams['figure.figsize'] = (10.0, 8.0)

#get all images containing at least one instance of legible text
imgIds = ct.getImgIds(imgIds = ct.val, catIds=[('legibility','legible')])

#pick one at random
img = ct.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]

#load and display text annotations
annIds = ct.getAnnIds(imgIds=img['id'])
anns = ct.loadAnns(annIds)
print(anns)
print(ct.showAnns(anns))

#now loading the image
I = io.imread('%s/%s/%s'%(dataDir,dataType,img['file_name']))
print('/%s/%s'%(dataType,img['file_name']))
# plt.figure()
# plt.imshow(I)
# plt.show()

##### Showing the bounding boxes on the image
import cv2

# print(type(img['file_name']))
# print(img['file_name'])
# print(type(I))

i=1
img_txt_bbox_loc = dict()
for item in anns:
    img_txt_bbox_loc[i] = item['bbox']
    # print(item['bbox'])
    print(img_txt_bbox_loc[i])
    x,y,w,h = item['bbox']
    x,y,w,h = int(x), int(y), int(w), int(h)
    cv2.rectangle(I,(x,y),(x+w,y+h),(0,255,0),2)
    cropped_img = I[y:y+h, x:x+w]
    cv2.imshow("cropped"+str(i), cropped_img)
    i += 1
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()



cv2.imshow('image+bbox',I)
# plt.show()
cv2.waitKey(0)
# cv2.destroyAllWindows()

# cv2.imshow('Image+bbox',I)
# plt.show()
print(type(img_txt_bbox_loc))







