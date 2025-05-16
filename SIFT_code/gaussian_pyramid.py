import image_process
import numpy as np
import pickle

# 以下是为了储存dog变量
def save_variable(v, filename):
    f = open(filename, 'wb')
    pickle.dump(v, f)
    f.close()
    return filename

def load_variavle(filename):
    f = open(filename, 'rb')
    r = pickle.load(f)
    f.close()
    return r
'''
filename = save_variable(results, 'results.txt')
results = load_variavle('results.txt')
'''
# 生成卷积核Kernel
def create_kernel(kernel_size, padding, sigma):
    # 定义kernel大小
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    # exp部分
    for x in range(-padding, -padding + kernel_size):
        for y in range(-padding, -padding + kernel_size):
            kernel[y + padding, x + padding] = np.exp(-(x ** 2 + y ** 2) / (2 * (sigma ** 2)))
    # 系数部分
    kernel /= (2 * np.pi * sigma * sigma)
    # 标准化处理，使得所有元素相加为1
    kernel /= kernel.sum()
    return kernel

# Gaussian filter 进行卷积
def convolution(img, kernel_size, sigma):
    # 将图像均变为三个维度
    # img, H, W, C = img_size_to3(img)

    # 灰度处理
    if len(img.shape) == 3:
        H, W, C = img.shape
    elif len(img.shape) == 2:
        H, W = img.shape
        C = 1

    # 对原图像的边缘进行Zero padding填充
    padding = kernel_size // 2
    # conv = np.zeros((H + padding * 2, W + padding * 2, C), dtype=np.float)
    conv = np.zeros((H + padding * 2, W + padding * 2), dtype=np.float32)
    conv[padding: padding + H, padding: padding + W] = img.copy().astype(np.float32)
    # 生成卷积核Kernel
    kernel = create_kernel(kernel_size, padding, sigma)
    # 建立输出图像，与img尺寸相同的零矩阵
    out = np.full_like(img, 0)
    # 卷积
    for y in range(H):
        for x in range(W):
            for c in range(C):
                # 对应点点乘，然后相加
                # out[y, x, c] = np.sum(kernel * conv[y: y + kernel_size, x: x + kernel_size, c])
                out[y, x] = np.sum(kernel * conv[y: y + kernel_size, x: x + kernel_size])
    out = out.squeeze()
    # 限定各像素点取值范围
    out = np.clip(out, 0, 255)
    # 化为int型
    out = out.astype(np.uint8)
    return out

def mean_pooling(img):
    # 降采样，采用平均池化方法

    # ds_data1 = np.mean(img.reshape(-1, 2, img.shape[1]), axis=1)
    # ds_data = np.mean(ds_data1.reshape(-1, 2, ds_data1.shape[2]), axis=1)

    # 原图像大小和pooling大小
    # h, w, c = img.shape
    if len(img.shape) == 3:
        h, w, c = img.shape
    elif len(img.shape) == 2:
        h, w= img.shape
        c = 1
    # 定义产出图片，若为奇数则为n+1
    length = (len(img)+1)//2
    width = (len(img[0])+1)//2
    channels = c

    # 输出的pooling图像大小
    # out = np.zeros((length, width, channels))
    out = np.zeros((length, width))
    # out = np.pad(img, (1,1,0), 'edge')

    '''
    padding处理，为了后面做pooling时避免index out of bound
    '''
    # padding处理原图像

    img_pad = []
    pad_width = 1
    img_padding = np.pad(img.reshape(h, w), pad_width, 'edge').reshape(h + pad_width * 2,
                                                       w + pad_width * 2, 1)
    # print(img.shape)
    # for x in zip(np.dsplit(img, c)):
    #     # print(np.pad(x.reshape(h,w), pad_width,'constant', constant_values=constant_value))
    #     x = np.array(x)
    #     img_pad.append(
    #         np.pad(x.reshape(h, w), pad_width, 'edge').reshape(h + pad_width * 2,
    #                                                                     w + pad_width * 2, 1)
    #     )
    #     img_padding = np.dstack(img_pad)



    # 测试产出图片
    # print(img_padding.shape)
    # cv2.imshow("result", img_padding)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # print(out.shape)
    # print(len(img[0][0]))


    # k = 2
    # data_pool = img.reshape(img.shape[0] // k, k, img.shape[1] // k, k, 3).mean(axis=(1, 3))

    # 开始池化
    # for k in range(channels):
    #     for j in range(0, width, 1):
    #         for i in range(0, length, 1):
    #             pixel1 = img_padding[i*2][j*2][k]
    #             pixel2 = img_padding[i*2+1][j*2][k]
    #             pixel3 = img_padding[i*2][j*2+1][k]
    #             pixel4 = img_padding[i*2+1][j*2+1][k]
    #             # out[i][j][k] = (pixel1 + pixel2 + pixel3 + pixel4)//4
    #             # RuntimeWarning: overflow encountered in scalar add
    #             out[i][j][k] = pixel1 / 4 + pixel2 / 4 + pixel3 / 4 + pixel4 / 4
    for j in range(0, width, 1):
        for i in range(0, length, 1):
            pixel1 = img_padding[i * 2][j * 2]
            pixel2 = img_padding[i * 2 + 1][j * 2]
            pixel3 = img_padding[i * 2][j * 2 + 1]
            pixel4 = img_padding[i * 2 + 1][j * 2 + 1]
            # out[i][j][k] = (pixel1 + pixel2 + pixel3 + pixel4)//4
            # RuntimeWarning: overflow encountered in scalar add
            out[i][j] = pixel1 / 4 + pixel2 / 4 + pixel3 / 4 + pixel4 / 4
    # 若无此段则不能画图
    out = out.astype(np.uint8)
    return out

def gaussian_pyramid(img, Octave = None, S_layers = 5, sigma0 = 1.52):
    # m = len(img)
    # n = len(img[0])
    # octave = int(round(log(min(image_shape)) / log(2) - 1))
    if Octave == None:
        Octave = int(np.log2(min(img.shape[0], img.shape[1]))) - 3  # 计算最大可以计算多少层 O=log2（min(img长，img宽））-3
    n = S_layers - 3
    # list_sigma = [0.8, 1.3, 2, 3]

    k = 2 ** (1.0 / n)
    sigma = [[(k ** s) * sigma0 * (1 << o) for s in range(S_layers)] for o in range(Octave)]
    sigma = np.array(sigma)
    # 每一层 sigma按照 k^1/s * sigama0  排列 下一层的sigma都要比上一层sigma大两倍
    ## linewidth
    # np.set_printoptions(linewidth=5)
    print(sigma)
    # n_std = len(list_sigma)
    pyramid = []
    # 构建五层高斯金字塔
    for i in range(Octave):
        list_octave = []
        for j in range(S_layers):
            # 逐步下采样
            Gau = convolution(img = img, kernel_size = 5, sigma = sigma[i][j])
            list_octave.append(Gau)
        img = mean_pooling(list_octave[2])
        # list_sigma = [i * 2 for i in list_sigma]
        pyramid.append(list_octave)
    return pyramid

def Difference_Of_Gaussian(pyramid):
    row = len(pyramid)
    col = len(pyramid[0])
    dog_pyramid = []
    for i in range(row):
        cols = []
        for j in range(0, col-1, 1):
            pic = pyramid[i][j] - pyramid[i][j+1]
            out = np.clip(pic, 0, 255)
            cols.append(out)
        dog_pyramid.append(cols)
    return dog_pyramid

if __name__ == "__main__":
    '''
    import cv2
    img = cv2.imread("./Vector_Tsoi.jpg")
    '''
    # 读取图片
    # img = image_process.read("./Vector_Tsoi.jpg")
    img = image_process.read("./vector3.jpg")
    img = image_process.toGray(img)
    # 计算金字塔
    pyramid = gaussian_pyramid(img)
    # image_process.draw_gaus(pyramid)
    # # for i in range(len(pyramid)):
    # #     for j in range(len(pyramid[0])):
    # #         pyramid[i][j] = image_process.toGray(pyramid[i][j])
    # #         pyramid[i][j] = np.squeeze(pyramid[i][j])
    # # 保存金字塔计算结果
    # filename = 'pyramid_data2.txt'
    # save_variable(pyramid, filename)
    # image_process.draw_gaus(pyramid)
    #
    # # print(pyramid[i][j].shape)
    # # 差分金字塔
    # dog_pyramid = Difference_Of_Gaussian(pyramid)
    # # 保存dog计算结果
    # filename = 'dog_data2.txt'
    # save_variable(dog_pyramid, filename)
    #
    # # 读取数据测试
    # filename = 'dog_data2.txt'
    # dog = load_variavle(filename)
    # image_process.draw_gaus(dog)

    # print(len(dog_pyramid))
    # print(len(dog_pyramid[0]))
    '''
    # 显示金字塔
    image_process.draw_gaus(dog_pyramid, 5, 4)
    image_process.draw_gaus(pyramid, 5, 5)
    '''

    '''
    # 保存图片
    for i in range(5*4):
        r = i // 4
        c = i % 4
        image_process.save(dog_pyramid[r][c], "./DOG_Pyramid/Octave_"+str(r)+'sigma_'+str(c)+'.jpg')
    '''
    # 用来测试mean_pooling
    # Save result
    # cv2.imshow("result", out)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


