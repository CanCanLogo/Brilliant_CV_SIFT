import numpy as np
import gaussian_pyramid as GAU
import image_process
from PIL import Image
from matplotlib import pyplot as plt

def loadPyramid(filename = 'dog_data.txt', filename2 = 'pyramid_data.txt'):
    # 此函数用于加载两个金字塔
    # 同时将金字塔中的图像转为灰度图

    dog = GAU.load_variavle(filename)
    # image_process.draw_gaus(dog, 5, 4)


    pyramid = GAU.load_variavle(filename2)

    # for i in range(len(dog)):
    #     for j in range(len(dog[0])):
    #         dog[i][j] = image_process.toGray(dog[i][j])
    #         # 去掉维度为1的维度
    #         dog[i][j] = np.squeeze(dog[i][j])
    # # image_process.draw_gaus(dog, 5, 4)
    # for i in range(len(pyramid)):
    #     for j in range(len(pyramid[0])):
    #         pyramid[i][j] = image_process.toGray(pyramid[i][j])
    #         pyramid[i][j] = np.squeeze(pyramid[i][j])
    # image_process.draw_gaus(pyramid, 5, 5)
    return dog, pyramid

def RoughPointsFind(DOG_pyramid, contrastThreshold=0.04):
    # 关键点列表
    KeyPoints = []
    # 层数和每层图数
    Octave = len(DOG_pyramid)
    Slide = len(DOG_pyramid[0])

    # 计算后续所需的阈值，不再把像素值过小的点当作关键点
    threshold = 0.5 * contrastThreshold / ( (Slide-3) * 255) # 这里的Slide-3=n是可用的层数
    for x in range(Octave):
        # 图像尺寸, 得知DOG_pyramid[0][0]的类型是numpy
        # print(DOG_pyramid[x][0].shape)
        m, n = DOG_pyramid[x][0].shape
        for y in range(1, Slide-1):
            # 这里的y的范围是因为第一个和最后一个图像层无法使用
            img_1 = DOG_pyramid[x][y - 1]
            img_2 = DOG_pyramid[x][y]
            img_3 = DOG_pyramid[x][y + 1]
            # 以上三张图是相邻的三张图，其中img_2是我们检测极值点的图
            for i in range(1, m-1, 1):
                for j in range(1, n-1, 1):
                    Up_layer = img_1[i - 1:i + 2, j - 1:j + 2]
                    In_layer = img_2[i - 1:i + 2, j - 1:j + 2]
                    Down_layer = img_3[i - 1:i + 2, j - 1:j + 2]
                    pixel = img_2[i][j]
                    if pixel > threshold:
                            if pixel >= Up_layer.all() and pixel >= In_layer.all() \
                                    and pixel >= Down_layer.all():
                                KeyPoints.append([x, y, i, j])
    return KeyPoints

def CalHessian(cube, img_scale):
    # 求解hessian矩阵，形式如下所示
    # img_scale是一个参数，常为1
    k_2_deriv = img_scale
    k_cross = img_scale * 0.25
    center_pixel_value = cube[1, 1, 1]
    dxx = k_2_deriv * (cube[1, 1, 2] - 2 * center_pixel_value + cube[1, 1, 0])
    dyy = k_2_deriv * (cube[1, 2, 1] - 2 * center_pixel_value + cube[1, 0, 1])
    dss = k_2_deriv * (cube[2, 1, 1] - 2 * center_pixel_value + cube[0, 1, 1])
    dxy = k_cross * (cube[1, 2, 2] - cube[1, 2, 0] - cube[1, 0, 2] + cube[1, 0, 0])
    dxs = k_cross * (cube[2, 1, 2] - cube[2, 1, 0] - cube[0, 1, 2] + cube[0, 1, 0])
    dys = k_cross * (cube[2, 2, 1] - cube[2, 0, 1] - cube[0, 2, 1] + cube[0, 0, 1])
    return np.array([[dxx, dxy, dxs],
                     [dxy, dyy, dys],
                     [dxs, dys, dss]])

def CalGradient(cube, img_scale):
    # 求解三张差分图像叠在一起的三维矩阵的微分值
    # 也即：[delta x, delta y, delta sigma]'
    # img_scale是一个参数，常为1
    k_deriv = img_scale * 0.5
    dx = k_deriv * (cube[1, 1, 2] - cube[1, 1, 0])
    dy = k_deriv * (cube[1, 2, 1] - cube[1, 0, 1])
    ds = k_deriv * (cube[2, 1, 1] - cube[0, 1, 1])
    return np.array([dx, dy, ds])

def FinePointsFind(DOG_pyramid, RoughKeyPoints, contrastThreshold=0.04, edgeThreshold = 10.0, sigma = 1.6):
    '''
    :param DOG_pyramid: 差分金字塔
    :param RoughKeyPoints: 经过粗筛的关键点
    :param contrastThreshold: 用于舍去低对比度的点
    :param edgeThreshold: 用于边缘海森检测，论文建议r = 10，OpenCv也采用r = 10
    :param sigma: 这个是所需的真实sigma，平方减去相机固有sigma后可得sigma0
    :return:
    '''
    # 以上的edgeThreshold用于边缘海森检测，论文建议r = 10，OpenCv也采用r = 10
    # 定义超参数
    FIXPT_SCALE = 1
    # 在Lowe中进行了5次迭代
    STEPS = 5
    # 边界值设置为4，边界外的值不作为极值点
    IMG_BORDER = 4
    # 求导过程中的常量参数
    img_scale = 1.0 / (255 * FIXPT_SCALE)
    # 可用层数为n
    n = len(DOG_pyramid[0]) - 2
    # 最终的关键点列表FineKeyPoint
    FineKeyPoint = []
    for kpoint in zip(RoughKeyPoints):
        kpoint = kpoint[0]
        x, y, i, j = int(kpoint[0]), int(kpoint[1]),int(kpoint[2]),int(kpoint[3])
        # x, y, i, j = point[0] # point是一个元组([,,],)
        img = DOG_pyramid[x][y]
        # 设定循环轮数控制变量
        step = 0
        # 生成一个类似于OpenCV KeyPoint object的列表
        KeyPoint = []
        while step < STEPS:
            step += 1
            # 排除不合法的取值
            # img = np.array(img)
            if y < 1 or y > len(DOG_pyramid[0]) - 2 or j < IMG_BORDER or j >= img.shape[
                1] - IMG_BORDER or i < IMG_BORDER \
                    or i >= img.shape[0] - IMG_BORDER:
                continue
            # 定义其他上下三幅图像
            img = DOG_pyramid[x][y]
            img_up = DOG_pyramid[x][y-1]
            img_down = DOG_pyramid[x][y+1]
            # 将三幅图像合并成一个
            cube = np.stack([img_up[i-1:i+2, j-1:j+2],
                            img[i-1:i+2, j-1:j+2],
                            img_down[i-1:i+2, j-1:j+2]]).astype('float32') / 255.
            # 计算梯度和海森矩阵
            Hessian = CalHessian(cube, img_scale)
            Gradient = CalGradient(cube, img_scale)

            # 计算得到三个方向应有的梯度
            dD = np.matmul(np.linalg.pinv(np.array(Hessian)), np.array(Gradient))
            dy, dx, ds = -1 * dD
            # print(dy)
            '''
            代表相对插值中心的偏移量，当它在任一维度上的偏移量大于0.5时，意味着插值中心已经偏移到它的邻近点上，
            所以必须改变当前关键点的位置。
            同时在新的位置上反复插值直到收敛；也有可能超出所设定的迭代次数或者超出图像边界的范围，
            此时这样的点应该删除，在Lowe中进行了5次迭代。
            另外，过小的点易受噪声的干扰而变得不稳定，所以将小于某个经验值(Lowe论文中使用0.03，rmislam实现时使用0.04/S)的极值点删除。
            同时，在此过程中获取特征点的精确位置(原位置加上拟合的偏移量)以及尺度
            '''
            # 在任一维度上的偏移量均小于0.5，插值中心不会再偏移
            if np.abs(dy) < 0.5 and np.abs(dx) < 0.5 and np.abs(ds) < 0.5:
                break
            j += int(np.round(dy))
            i += int(np.round(dx))
            y += int(np.round(ds))
        # 以下为for循环内的代码
        # 如果循环了五次，说明未更新成功，则非关键点
        if step >= STEPS:
            continue
        if y < 1 or y > len(DOG_pyramid[0]) - 2 or j < IMG_BORDER or j >= img.shape[
            1] - IMG_BORDER or i < IMG_BORDER \
                or i >= img.shape[0] - IMG_BORDER:
            continue
        Updated_Center_Value = img[i][j] + 0.5 * dD.dot(Gradient)
        # 舍去低对比度的点
        if np.abs(Updated_Center_Value) * (len(DOG_pyramid[0]) - 2) < contrastThreshold:
            continue

        # 边缘效应的去除。 利用Hessian矩阵的迹和行列式计算主曲率的比值
        hessian_xy = Hessian[:2, :2]
        # 求解海森矩阵的迹
        trace = np.trace(hessian_xy) # tr = dxx + dyy
        # 求解海森矩阵的行列式 det = np.mat(hessian_xy).det()   error:  'matrix' object has no attribute 'det'
        det = hessian_xy[0][0] * hessian_xy[1][1] - hessian_xy[0][1] * hessian_xy[1][0]

        # 参数r = edgeThreshold
        r = edgeThreshold
        if det <= 0 or trace * trace * r >= (r + 1) * (r + 1) * det:
            continue
        # 1 << o 等效于 2 ** o
        KeyPoint.append((i + dx) * (2 ** x))  # keypoint.point_x
        KeyPoint.append((j + dy) * (2 ** x))  # keypoint.point_y
        KeyPoint.append(x + (y << 8) + (int(np.round((ds + 0.5)) * 255) * (2 ** 16)))  # keypoint.octave
        KeyPoint.append(sigma * np.power(2.0, (y + ds) / n) * (2 ** x) * 2)  # keypoint.size
        # octave_index + 1 because the input image was doubled
        KeyPoint.append(Updated_Center_Value) # keypoint.response

        FineKeyPoint.append([KeyPoint, i, j, y, x])
    return FineKeyPoint

def MainDerection(KeyPoint, G_pyramid, radius_factor=3, num_bins=36, peak_ratio=0.8):
    '''
    :param KeyPoint: 传入的关键点列表
    :param G_pyramid:  高斯金字塔
    :param radius_factor: 半径影响因数
    :param num_bins: 统计的柱子数量
    :param peak_ratio: 拟合参数
    :return: KeyPoint_Oriented有主方向的关键点
    '''
    '''
    keypoint:
    point_x
    point_y
    octave
    size
    response
    orientation
    '''
    KeyPoint_Oriented = []
    # 设置float边界
    float_tolerance = 1e-7
    for iter in enumerate(KeyPoint):
        '''
        iter如下示例：编号，[KeyPoint, i, j, y, x]
        (38023, [[143.84056602666158, 293.1106496731343, 516, 100.64853607919784, 2.000003071514719], 9, 18, 2, 4])
        '''
        # print(iter)
        point = iter[1][0]
        i, j, y, x = int(iter[1][1]), int(iter[1][2]), int(iter[1][3]), int(iter[1][4])

        img = G_pyramid[x][y]
        image_shape = img.shape

        # 以下计算sigma_oct
        sigma_octv = point[-2] * 0.5 / (1 << x)
        # 以下计算半径
        radius = int(round(radius_factor * sigma_octv))
        # 以下计算权重
        weight_factor = -0.5 / (sigma_octv ** 2)
        raw_histogram = np.zeros(num_bins)
        smooth_histogram = np.zeros(num_bins)

        for radi_y in range(-radius, radius+1):
            pt_x = point[0]
            pt_y = point[1]
            region_y = int(round(pt_y / np.float32(2 ** x))) + radi_y
            if region_y > 0 and region_y < image_shape[0] - 1:
                for redi_x in range(-radius, radius + 1):
                    region_x = int(round(pt_x / np.float32(2 ** x))) + redi_x
                    if region_x > 0 and region_x < image_shape[1] - 1:
                        dx = img[region_y, region_x + 1] - img[region_y, region_x - 1]
                        # print(dx)
                        dy = img[region_y - 1, region_x] - img[region_y + 1, region_x]
                        # print(dy)
                        gradient_magnitude = np.sqrt(dx * dx + dy * dy)
                        gradient_orientation = np.rad2deg(np.arctan2(dy, dx))
                        weight = np.exp(weight_factor * (
                                radi_y ** 2 + redi_x ** 2))  # constant in front of exponential can be dropped because we will find peaks later
                        # print(weight)
                        histogram_index = int(round(gradient_orientation * num_bins / 360.))
                        raw_histogram[histogram_index % num_bins] += weight * gradient_magnitude

        for n in range(num_bins):
            # 平滑函数
            smooth_histogram[n] = (6 * raw_histogram[n] + 4 * (
                        raw_histogram[n - 1] + raw_histogram[(n + 1) % num_bins]) + raw_histogram[n - 2] +
                                   raw_histogram[(n + 2) % num_bins]) / 16.
        orientation_max = max(smooth_histogram)
        orientation_peaks = \
        np.where(np.logical_and(smooth_histogram > np.roll(smooth_histogram, 1), smooth_histogram > np.roll(smooth_histogram, -1)))[
            0]
        for peak_index in orientation_peaks:
            peak_value = smooth_histogram[peak_index]
            if peak_value >= peak_ratio * orientation_max:
                # Quadratic peak interpolation
                # The interpolation update is given by equation (6.30) in https://ccrma.stanford.edu/~jos/sasp/Quadratic_Interpolation_Spectral_Peaks.html
                left_value = smooth_histogram[(peak_index - 1) % num_bins]
                right_value = smooth_histogram[(peak_index + 1) % num_bins]
                interpolated_peak_index = (peak_index + 0.5 * (left_value - right_value) / (
                            left_value - 2 * peak_value + right_value)) % num_bins
                orientation = 360. - interpolated_peak_index * 360. / num_bins
                if abs(orientation - 360.) < float_tolerance:
                    orientation = 0
                # new_keypoint = KeyPoint(*keypoint.pt, keypoint.size, orientation, keypoint.response, keypoint.octave)
                new_keypoint = point.append(orientation)
                print(orientation)
                KeyPoint_Oriented.append(point)
    return KeyPoint_Oriented

def CalDescriptors(Keypoints, G_pyramid, Window_Width=4, num_bins=8, Scale_Multiplier=3, descriptor_max_value=0.2):
    '''

    :param Keypoints: 关键点
    :param G_pyramid: 高斯金字塔
    :param window_width: 窗口宽度 将关键点附近的区域划分为d ∗ d(Lowe建议d = 4*4)个子区域
    :param num_bins: 每个子区域作为一个种子点，每个种子点有8个方向
    :param scale_multiplier: 图像区域半径
    :param descriptor_max_value:
    :return:
    '''
    descriptors = []
    float_tolerance = 1e-7
    for keypoint in Keypoints:
        # print(keypoint)
        # [3.651292404473298, 47.651857983499205, 256, 4.3917779267658394, 75.99979638380371, 274.77064818146965]
        '''
        point_x
        point_y
        octave
        size
        response
        orientation
        '''
        octave = keypoint[2] & 255  # 金字塔层数
        layer = (octave >> 8) & 255  # 层内第几个
        scale = 1 / np.float32(1 << octave) if octave >= 0 else np.float32(1 << -octave)  # 缩放倍数
        size = keypoint[3] * scale  # 该特征点所在组的图像尺寸

        pt_x = int(np.round(keypoint[1] * scale))  # 在图像上横坐标
        pt_y = int(np.round(keypoint[0] * scale))  # 纵坐标
        orient = keypoint[5]
        img = G_pyramid[octave][layer]  # 该点所在的金字塔图像
        # gaussian_image = gaussian_images[octave + 1, layer]

        # 高斯金字塔中图像的尺寸
        rows, cols = img.shape

        bins_per_degree = num_bins / 360. # 每一个bin对应角度

        # 计算关键点方向的正弦余弦
        angle = 360. - orient
        cos_angle = np.cos(np.deg2rad(angle))
        sin_angle = np.sin(np.deg2rad(angle))

        # 关键参数计算
        weight_multiplier = -0.5 / ((0.5 * Window_Width) ** 2)

        # 计算Descriptor window size
        hist_width = Scale_Multiplier * 0.5 * size
        radius = int(
            round(hist_width * np.sqrt(2) * (Window_Width + 1) * 0.5))
        # 确保不越界
        radius = int(min(radius, np.sqrt(rows ** 2 + cols ** 2)))

        # 调整cos,sin，用于后面第一步，row_window, col_window需要除以hist_width
        cos_angle = cos_angle / hist_width
        sin_angle = sin_angle / hist_width

        Row_bin_list = []
        Col_bin_list = []
        magnitude_list = []
        orientation_bin_list = []

        # for x in range(-radius, radius + 1):
        #     for y in range(-radius, radius + 1):
        #

        '''
        循环中使用x,y容易混淆行列和横纵
        x是col,y是row
        '''

        for row in range(-radius, radius + 1):
            for col in range(-radius, radius + 1):

                # row, col是相对坐标  计算调整后的坐标
                row_window = col * sin_angle + row * cos_angle
                col_window = col * cos_angle - row * sin_angle

                # 计算调整后的角度
                row_bin = row_window + 0.5 * Window_Width - 0.5
                col_bin = col_window + 0.5 * Window_Width - 0.5

                # 计算绝对坐标
                window_row = int(round(pt_y + row))
                window_col = int(round(pt_x + col))
                if row_bin > -1 and row_bin < Window_Width and col_bin > -1 and col_bin < Window_Width and \
                    window_row > 0 and window_row < rows - 1 and window_col > 0 and window_col < cols - 1:

                    dx = img[window_row, window_col + 1] - img[window_row, window_col - 1]
                    dy = img[window_row - 1, window_col] - img[window_row + 1, window_col]
                    gradient_magnitude = np.sqrt(dx * dx + dy * dy)

                    gradient_orientation = np.rad2deg(np.arctan2(dy, dx)) % 360
                    weight = np.exp(weight_multiplier * (row_window ** 2 + col_window ** 2))

                    # 列表append
                    Row_bin_list.append(row_bin)
                    Col_bin_list.append(col_bin)
                    magnitude_list.append(weight * gradient_magnitude)
                    orientation_bin_list.append((gradient_orientation - angle) * bins_per_degree)
                    pass
        histogram_tensor = np.zeros((Window_Width + 2, Window_Width + 2, num_bins))
        for row_bin, col_bin, magnitude, orientation_bin in zip(Row_bin_list, Col_bin_list, magnitude_list, orientation_bin_list):

            row_bin_floor, col_bin_floor, orientation_bin_floor = np.floor([row_bin, col_bin, orientation_bin]).astype(int)
            row_fraction, col_fraction, orientation_fraction = row_bin - row_bin_floor, col_bin - col_bin_floor, orientation_bin - orientation_bin_floor
            if orientation_bin_floor < 0:
                orientation_bin_floor += num_bins
            if orientation_bin_floor >= num_bins:
                orientation_bin_floor -= num_bins

            c1 = magnitude * row_fraction
            c0 = magnitude * (1 - row_fraction)
            c11 = c1 * col_fraction
            c10 = c1 * (1 - col_fraction)
            c01 = c0 * col_fraction
            c00 = c0 * (1 - col_fraction)
            c111 = c11 * orientation_fraction
            c110 = c11 * (1 - orientation_fraction)
            c101 = c10 * orientation_fraction
            c100 = c10 * (1 - orientation_fraction)
            c011 = c01 * orientation_fraction
            c010 = c01 * (1 - orientation_fraction)
            c001 = c00 * orientation_fraction
            c000 = c00 * (1 - orientation_fraction)

            histogram_tensor[row_bin_floor + 1, col_bin_floor + 1, orientation_bin_floor] += c000
            histogram_tensor[row_bin_floor + 1, col_bin_floor + 1, (orientation_bin_floor + 1) % num_bins] += c001
            histogram_tensor[row_bin_floor + 1, col_bin_floor + 2, orientation_bin_floor] += c010
            histogram_tensor[row_bin_floor + 1, col_bin_floor + 2, (orientation_bin_floor + 1) % num_bins] += c011
            histogram_tensor[row_bin_floor + 2, col_bin_floor + 1, orientation_bin_floor] += c100
            histogram_tensor[row_bin_floor + 2, col_bin_floor + 1, (orientation_bin_floor + 1) % num_bins] += c101
            histogram_tensor[row_bin_floor + 2, col_bin_floor + 2, orientation_bin_floor] += c110
            histogram_tensor[row_bin_floor + 2, col_bin_floor + 2, (orientation_bin_floor + 1) % num_bins] += c111

        descriptor_vector = histogram_tensor[1:-1, 1:-1, :].flatten()  # Remove histogram borders
        # Threshold and normalize descriptor_vector
        threshold = np.linalg.norm(descriptor_vector) * descriptor_max_value
        descriptor_vector[descriptor_vector > threshold] = threshold
        descriptor_vector /= max(np.linalg.norm(descriptor_vector), float_tolerance)
        # Multiply by 512, round, and saturate between 0 and 255 to convert from float32 to unsigned char (OpenCV convention)
        descriptor_vector = np.round(512 * descriptor_vector)
        descriptor_vector[descriptor_vector < 0] = 0
        descriptor_vector[descriptor_vector > 255] = 255
        descriptors.append(descriptor_vector)
    descriptors = np.array(descriptors, dtype='float32')
    return descriptors

if __name__ == "__main__":

    DOG_pyramid, pyramid = loadPyramid('pyramid_data2.txt', 'dog_data2.txt')
    # 粗略计算
    RoughKeyPoints = RoughPointsFind(DOG_pyramid)
    # FineKeyPoints = FinePointsFind(DoG, o, s, i, j, contrastThreshold, edgeThreshold,sigma, n, SIFT_FIXPT_SCALE)

    # 精细计算
    FineKeyPoint = FinePointsFind(DOG_pyramid, RoughKeyPoints)

    # 保存变量
    print(FineKeyPoint)
    filename = 'FineKeyPoint2.txt'
    GAU.save_variable(FineKeyPoint, filename)

    # 用于计算主方向
    filename = 'FineKeyPoint2.txt'
    FineKeyPoint = GAU.load_variavle(filename)
    KeyPoints = MainDerection(FineKeyPoint, pyramid, 0)
    print(KeyPoints)

    # 保存计算完成的关键点
    filename2 = 'KeyPoint2.txt'
    GAU.save_variable(KeyPoints, filename2)

    # 载入关键点
    filename2 = 'KeyPoint2.txt'
    KeyPoints = GAU.load_variavle(filename2)

    # 描述符计算
    descriptors = CalDescriptors(KeyPoints, pyramid)
    print(descriptors)
    filename3 = 'descriptors2.txt'
    GAU.save_variable(descriptors, filename3)

    DOG_pyramid, pyramid = loadPyramid('pyramid_data1.txt', 'dog_data1.txt')
    # 粗略计算
    RoughKeyPoints = RoughPointsFind(DOG_pyramid)
    # FineKeyPoints = FinePointsFind(DoG, o, s, i, j, contrastThreshold, edgeThreshold,sigma, n, SIFT_FIXPT_SCALE)

    # 精细计算
    FineKeyPoint = FinePointsFind(DOG_pyramid, RoughKeyPoints)

    # 保存变量
    print(FineKeyPoint)
    filename = 'FineKeyPoint1.txt'
    GAU.save_variable(FineKeyPoint, filename)

    # 用于计算主方向
    filename = 'FineKeyPoint1.txt'
    FineKeyPoint = GAU.load_variavle(filename)
    KeyPoints = MainDerection(FineKeyPoint, pyramid, 0)
    print(KeyPoints)

    # 保存计算完成的关键点
    filename2 = 'KeyPoint1.txt'
    GAU.save_variable(KeyPoints, filename2)

    # 载入关键点
    filename2 = 'KeyPoint1.txt'
    KeyPoints = GAU.load_variavle(filename2)

    # 描述符计算
    descriptors = CalDescriptors(KeyPoints, pyramid)
    print(descriptors)
    filename3 = 'descriptors1.txt'
    GAU.save_variable(descriptors, filename3)

    image_process.draw_gaus(pyramid)
