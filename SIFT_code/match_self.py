import gaussian_pyramid as GAU
import image_process

import numpy as np


origimg = image_process.pread("./vector3.jpg")
if len(origimg.shape) == 3:#如果是彩色图，就按照三通道取均值的方式转成灰度图
    img = origimg.mean(axis=-1)
else:
    img = origimg
# keyPoints, discriptors = SIFT(img)  # 用SIFT算法计算关键点（x坐标，y坐标，sigma，主方向，梯度幅值）和描述符（128维的向量）

filename2 = './data/KeyPoint1.txt'
keyPoints = GAU.load_variavle(filename2)
filename3 = './data/descriptors1.txt'
descriptors = GAU.load_variavle(filename3)

origimg2 = image_process.pread("./Vector_Tsoi.jpg")  # 读第二张图片
if len(origimg.shape) == 3:
    img2 = origimg2.mean(axis=-1)
else:
    img2 = origimg2

img2 = image_process.res(img, img2)

# keyPoints2, discriptors2 = SIFT(img2)  # 用SIFT算关键点和描述符
filename4 = './data/KeyPoint2.txt'
keyPoints2 = GAU.load_variavle(filename4)
filename5 = './data/descriptors2.txt'
descriptors2 = GAU.load_variavle(filename5)

indexs = []
deltas = []
for i in range(len(keyPoints2)):
    ds = descriptors2[i]
    mindetal = 10000000
    index = -1
    detal = 0
    for j in range(len(keyPoints)):
        ds0 = descriptors[j]
        d = np.array(ds)-np.array(ds0)
        detal = d.dot(d)
        if( detal <= mindetal):
            mindetal = detal
            index = j
    indexs.append(index)
    deltas.append(mindetal)


keyPoints = np.array(keyPoints)[:,:2]
keyPoints2 = np.array(keyPoints2)[:,:2]

keyPoints2[:, 1] = img.shape[1] + keyPoints2[:, 1]

result = image_process.resize(origimg, origimg2, img, img2)

keyPoints = keyPoints[indexs[:]]

X1 = keyPoints[:, 1]
X2 = keyPoints2[:, 1]
Y1 = keyPoints[:, 0]
Y2 = keyPoints2[:, 0]

image_process.drawLines(X1, X2, Y1, Y2, deltas, result)