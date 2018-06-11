import numpy as np
import cv2

def read_file(file):
    img=cv2.imread(file)
    img=np.array(img)

    # 0 46 47:93
    # 4 25
    img1=img[:,7:27,:]
    img2=img[:,27:47,:]
    img3=img[:,47:67,:]
    img4=img[:,67:87,:]

    cv2.imshow('1', img1)
    cv2.imshow('2', img2)
    cv2.imshow('3', img3)
    cv2.imshow('4', img4)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(img.shape)

# read_file('./code/0126.jpg')
# read_file('./code/0290.jpg')
# read_file('./code/2870.jpg')
# read_file('./code/3122.jpg')
# read_file('./code/5315.jpg')
read_file('./0000.png')