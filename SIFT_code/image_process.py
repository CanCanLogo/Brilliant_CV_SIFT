import cv2
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
# Read image
# def read(filename = "./Vector_Tsoi.jpg"):
def read(filename):
    img = cv2.imread(filename)
    return img
def pread(filename):
    origimg = plt.imread(filename, 0)
    return origimg
# show image
def show(img, title = "result"):
    cv2.imshow(title, img)
# Save result
# def save(img, filename = "vector_p1_s2.jpg"):
def save(img, filename):
    cv2.imwrite(filename, img)

def toGray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def resize(origimg, origimg2, img, img2):
    origimg2 = np.array(Image.fromarray(origimg2).resize((img2.shape[1], img2.shape[0]), Image.BICUBIC))
    result = np.hstack((origimg, origimg2))
    return result

def res(img, img2):
    ScaleRatio = img.shape[0] * 1.0 / img2.shape[0]
    img2 = np.array(Image.fromarray(img2).resize((int(round(ScaleRatio * img2.shape[1])), img.shape[0]), Image.BICUBIC))
    return img2


def draw_gaus(pyramid, rows = 5, cols = 4):
    # 画高斯金字塔
    axes = []
    fig = plt.figure()
    rows = len(pyramid)
    cols = len(pyramid[0])
    for a in range(rows * cols):
        # 第一种方法
        # r = a // 5
        # c = a % 5
        # axes.append(fig.add_subplot(rows, cols, a + 1))
        # subplot_title = ("Subplot" + str(a))
        # axes[-1].set_title(subplot_title)
        # plt.imshow(cv2.cvtColor(pyramid[r][c], cv2.COLOR_BGR2RGB))

        # 第二种方法
        r = a // cols
        c = a % cols
        # fig.add_subplot(rows, cols, a + 1)
        # plt.imshow(cv2.cvtColor(pyramid[r][c], cv2.COLOR_BGR2RGB))

        # 第三种，可以无轴无标题
        plt.subplot(rows, cols, a + 1)
        # plt.imshow(cv2.cvtColor(pyramid[r][c], cv2.COLOR_BGR2RGB))
        plt.imshow(pyramid[r][c], cmap ='gray')
        # plt.title("")
        plt.xticks([]), plt.yticks([])
    fig.tight_layout()
    plt.show()


def Lines(img, info, color=(255, 0, 0), err=700):
    if len(img.shape) == 2:
        result = np.dstack((img, img, img))
    else:
        result = img
    k = 0
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            temp = (info[:, 1] - info[:, 0])
            A = (j - info[:, 0]) * (info[:, 3] - info[:, 2])
            B = (i - info[:, 2]) * (info[:, 1] - info[:, 0])
            temp[temp == 0] = 1e-9
            t = (j - info[:, 0]) / temp
            e = np.abs(A - B)
            temp = e < err
            if (temp * (t >= 0) * (t <= 1)).any():
                result[i, j] = color
                k += 1
    #print(k)

    return result


def drawLines(X1, X2, Y1, Y2, dis, img, num=10):
    info = list(np.dstack((X1, X2, Y1, Y2, dis))[0])
    info = sorted(info, key=lambda x: x[-1])
    info = np.array(info)
    info = info[:min(num, info.shape[0]), :]
    img = Lines(img, info)
    # plt.imsave('./sift/3.jpg', img)

    if len(img.shape) == 2:
        plt.imshow(img.astype(np.uint8), cmap='gray')
    else:
        plt.imshow(img.astype(np.uint8))
    plt.axis('off')
    # plt.plot([info[:,0], info[:,1]], [info[:,2], info[:,3]], 'c')
    # fig = plt.gcf()
    # fig.set_size_inches(int(img.shape[0]/100.0),int(img.shape[1]/100.0))
    plt.savefig('result.jpg')
    plt.show()