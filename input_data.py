import os
import cv2
import numpy as np
import pandas as pd

files=os.listdir('./code/')

idx=0

images=[]
labels=[]

for file in files:
    img=cv2.imread('./code/{}'.format(file))
    img=img[:,:,0]
    img2 = img.flatten()
    abc = pd.value_counts(img2)
    for x in abc.index:
        if abc[x]>30:
            abc=abc.drop(x)
    for x in abc.index:
        img[img==x]=0.0

    img[img!=0]=1.0

    img=img[:,:,np.newaxis]

    images.append(img)

    label=[]
    filename=file[0:4]
    for no in filename:
        xx=[0]*10
        xx[int(no)]=1
        label.append(xx)

    labels.append(label)
labels=np.array(labels)
images=np.array(images)

print(labels.shape,images.shape)