import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math
from skimage import color


# # 计算每个通道的平均像素值
def average_color(channel):
    h, w = channel.shape
    sum = 0
    for i in range(h):
        for j in range(w):
            sum = sum + channel[i][j]
    ave = sum / (w * h)
    return ave


# 根据论文公式 进行红 蓝 通道的补偿
def channel_compensate(channel, g):
    h, w = channel.shape
    g_ave = average_color(g)
    now_channel_ave = average_color(channel)
    for i in range(h):
        for j in range(w):
            channel[i][j] = channel[i][j] + (g_ave - now_channel_ave) * (1 - channel[i][j]) * g[i][j]
    return channel


# gray world
def GW(orgImg):
    B, G, R = np.double(orgImg[:, :, 0]), np.double(orgImg[:, :, 1]), np.double(orgImg[:, :, 2])
    B_ave, G_ave, R_ave = np.mean(B), np.mean(G), np.mean(R)
    K = (B_ave + G_ave + R_ave) / 3
    Kb, Kg, Kr = K / B_ave, K / G_ave, K / R_ave
    Ba = (B * Kb)
    Ga = (G * Kg)
    Ra = (R * Kr)
    dst_img = np.uint8(np.zeros_like(orgImg))
    dst_img[:, :, 0] = Ba
    dst_img[:, :, 1] = Ga
    dst_img[:, :, 2] = Ra
    return dst_img


# 总函数 使用该函数进行白平衡
def white_balance(tag, img_path):
    img = cv.imread(img_path, 1)
    h, w, c = img.shape
    b, g, r = cv.split(img)
    b = b / 255
    g = g / 255
    r = r / 255
    r_h, r_w = r.shape
    r_c = channel_compensate(r, g)
    b_c = channel_compensate(b, g)
    img_compen = cv.merge([b_c, g, r_c])
    img_compen = img_compen * 255
    img_compen = img_compen.astype(np.uint8)
    if tag == 1:
        img_compen = GW(img_compen)  # gray world
    return img, img_compen


# gamma校正
def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    # table = np.array([((i / 255.0) ** invGamma) * 255
    #                   for i in np.arange(0, 256)]).astype(np.int64)
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype(np.uint8)

    return cv.LUT(image, table)


# 锐化
def sharpen(img):
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])  #

    output = cv.filter2D(img, -1, kernel)
    return output


def sharpen2(img):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

    output = cv.filter2D(img, -1, kernel)
    return output


# 显示图片,在键盘按键后 图像消失
def img_show(title, img):
    cv.imshow(title, img)
    cv.waitKey(0)
    cv.destroyAllWindows()


def bgr2rgb(img):
    b, g, r = cv.split(img)
    img_rgb = cv.merge([r, g, b])
    return img_rgb


def show(img):
    img = bgr2rgb(img)
    plt.imshow(img)
    plt.show()


def compare_show(img1, img2):
    img1 = bgr2rgb(img1)
    img2 = bgr2rgb(img2)
    plt.subplot(121)
    plt.imshow(img1)
    plt.subplot(122)
    plt.imshow(img2)
    plt.show()


# 计算拉普拉斯权重
def laplace(image_path):
    image = image_path
    # image = cv.imread(image_path, 1)
    # image = cv.cvtColor(image, cv.COLOR_RGB2LAB) # rgb图片转化成lab图
    image_l = np.double(image[:, :, 0]) / 256  # 取亮度值
    # cv.imshow("L", image_l)
    # cv.waitKey(0)
    # print(image_l)
    output_9 = cv.Laplacian(image_l, cv.CV_64F)
    output_9 = cv.convertScaleAbs(output_9)

    # 拉普拉斯算子
    # kernel_1 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    # kernel_2 = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    # kernel_3 = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    # output_9 = cv.filter2D(image_l, -1, kernel_2)
    # output_9 = correlate(image_l, kernel_1, mode='nearest')

    return output_9


# 计算显著性权重
def saliency(image_path):
    image = image_path
    # image = cv.cvtColor(image, cv.COLOR_RGB2LAB)
    image = color.rgb2lab(image)
    output = np.zeros((image.shape[0], image.shape[1]), dtype=int)
    SUML = 0
    SUMB = 0
    SUMA = 0
    size = image.shape[0] * image.shape[1]
    for x in range(image.shape[0]):  # 遍历每一个像素
        for y in range(image.shape[1]):
            l, a, b = image[x, y]
            SUML += l
            SUMA += a
            SUMB += b
    ml = SUML / size
    ma = SUMA / size
    mb = SUMB / size
    for x in range(image.shape[0]):  # 遍历每一个像素
        for y in range(image.shape[1]):
            l, a, b = image[x, y]
            output[x, y] = (l - ml) * (l - ml) + (a - ma) * (a - ma) + (b - mb) * (b - mb)
    # cv.imshow("L", output)
    return output


# 计算饱和权重
def saturation(image_path):
    # image = cv.imread(image_path, 1) # 以彩色图bgr读入
    image = image_path
    # cv.imshow('rgb image', image) # 显示图像
    # if cv.waitKey(0) == 27: # 设置按esc键时，关闭图像
    #     cv.destroyAllWindows()

    '''
    因为不知道输出是二维数组还是灰度图，所以两个都先写着
    '''

    output = np.zeros((image.shape[0], image.shape[1]), dtype=int)  # 初始化一个和图像大小一样的二维数组，每个元素为0

    out_img = np.ones((image.shape[0], image.shape[1]), np.uint8)  # 初始化一个和图像大小一样的灰度图，每个像素为1
    out_img = 255 * out_img  # 将像素改为255

    # cv.imshow('out_img', out_img) # 显示图像
    # if cv.waitKey(0) == 27: # 设置按esc键时，关闭图像
    #     cv.destroyAllWindows()

    for x in range(image.shape[0]):  # 遍历每一个像素
        for y in range(image.shape[1]):
            b, g, r = image[x, y]  # 获取每个像素的各个通道的值
            l = r * 0.299 + g * 0.587 + b * 0.114  # 计算亮度
            w = math.sqrt(1 / 3 * (math.pow((r - l), 2) + math.pow((g - l), 2) + math.pow((b - l), 2)))  # 饱和权重计算
            output[x][y] = w
            out_img[x][y] = w

    # print(output)

    # cv.imshow('out_image', out_img) # 显示计算饱和权后的灰度图
    # if cv.waitKey(0) == 27:
    #     cv.destroyAllWindows()

    return out_img


# 下采样
def down_half(image):
    h = np.array([1, 4, 6, 4, 1]) / 16
    # print(h)
    filt = (h.T).dot(h)
    # print(filt)
    # output = cv.filter2D(image, -1, filt)
    out = cv.filter2D(image, cv.CV_64F, filt)
    output = out[::2, ::2]
    return output


# 高斯金字塔
def gauss_pyramid(image, level):
    pyramid_images = []
    pyramid_images.append(image)
    temp = image.copy()
    for i in range(0, level):
        temp = down_half(temp)
        pyramid_images.append(temp)
        # cv.imshow("gauss" + str(i), temp)
        # cv.waitKey(0)
    return pyramid_images


# 上采样
def up_half(image):
    h = np.array([1, 4, 6, 4, 1]) / 16
    filt = (h.T).dot(h)
    outimage = np.zeros((image.shape[0] * 2, image.shape[1] * 2), dtype=np.float64)
    outimage[::2, ::2] = image[:, :]
    out = cv.filter2D(outimage, cv.CV_64F, filt)
    return out


# 拉普拉斯金字塔
def lap_pyramid(pyramid_images):
    pyramid_lap = []
    k = len(pyramid_images)
    for i in range(0, k - 1):
        big_one = pyramid_images[i]
        small_one = up_half(pyramid_images[i + 1])
        if small_one.shape[0] > big_one.shape[0]:
            small_one = np.delete(small_one, (-1), axis=0)
        if small_one.shape[1] > big_one.shape[1]:
            small_one = np.delete(small_one, (-1), axis=1)
        pyramid_lap.append(big_one - small_one)
        # cv.imshow("lap" + str(i), big_one - small_one)
        # cv.waitKey(0)
    pyramid_lap.append(pyramid_images.pop())
    return pyramid_lap


def split_rgb(image):
    (R, G, B) = cv.split(image)
    return B, G, B


def collapse(img_pyr):
    for i in range(len(img_pyr) - 1, 0, -1):
        lap = up_half(img_pyr[i])
        lap_last = img_pyr[i - 1]
        if lap.shape[0] > lap_last.shape[0]:
            lap = np.delete(lap, (-1), axis=0)
        if lap.shape[1] > lap_last.shape[1]:
            lap = np.delete(lap, (-1), axis=1)
        tmp = lap + lap_last
    output = tmp
    return output


img_path = './img_3.jpg'
img, img_compen = white_balance(tag=1, img_path=img_path)  # 白平衡
# tag == 1 时， 在白平衡后使用gray-world 处理 ， tag == 0 则不处理
img_gamma_adjust = adjust_gamma(img_compen)  # gamma校正
img_sharpen1 = sharpen(img_compen)  # 锐化1
img_sharpen2 = sharpen2(img_compen)  # 锐化2

# compare_show(img, img_sharpen1)                # 图片对比

# img_gamma_adjust = np.array(img_gamma_adjust, dtype=np.float32)
img_lap_1 = laplace(img_gamma_adjust)  # 计算拉普拉斯权重1
img_lap_2 = laplace(img_sharpen1)  # 计算拉普拉斯权重2
# cv.imshow('Laplace Image', img_lap_2)
# cv.waitKey(0)

img_wsal_1 = saliency(img_gamma_adjust)  # 计算显著性权重1
img_wsal_2 = saliency(img_sharpen1)  # 计算显著性权重2
# img_wsal_1 = np.array(img_wsal_1, dtype=np.float32)
# img_wsal_2 = np.array(img_wsal_2, dtype=np.float32)
# cv.imshow('wsal Image', img_wsal_1)
# cv.waitKey(0)

img_wsat_1 = saturation(img_gamma_adjust)  # 计算饱和权重1
img_wsat_2 = saturation(img_sharpen1)  # 计算饱和权重2
# cv.imshow('wsat Image', img_wsat_1)
# cv.waitKey(0)

# WL 拉普拉斯权重
# WS 显著权重
# WSat 饱和权重
# 权重归一化处理
# weight1 = (WL1+WS1+WSat1+0.1)/(WL1+WS1+WSat1+WL2+WS2+WSat2+0.2)
# weight2 = (WL2+WS2+WSat2+0.1)/(WL1+WS1+WSat1+WL2+WS2+WSat2+0.2)
weight1 = (img_lap_1 + img_wsal_1 + img_wsat_1) / (
        img_lap_1 + img_wsal_1 + img_wsat_1 + img_lap_2 + img_wsal_2 + img_wsat_2)
weight2 = (img_lap_2 + img_wsal_2 + img_wsat_2) / (
        img_lap_1 + img_wsal_1 + img_wsat_1 + img_lap_2 + img_wsal_2 + img_wsat_2)

# 多尺度融合
level = 3
Weight1 = gauss_pyramid(weight1, level)
Weight2 = gauss_pyramid(weight2, level)
# print(len(Weight1))
# print(len(Weight2))

(r1, g1, b1) = split_rgb(img_gamma_adjust)  # gamma
r1 = gauss_pyramid(r1, level)
g1 = gauss_pyramid(g1, level)
b1 = gauss_pyramid(b1, level)
r1 = lap_pyramid(r1)
g1 = lap_pyramid(g1)
b1 = lap_pyramid(b1)
(r2, g2, b2) = split_rgb(img_sharpen1)  # 锐化
r2 = gauss_pyramid(r2, level)
g2 = gauss_pyramid(g2, level)
b2 = gauss_pyramid(b2, level)
r2 = lap_pyramid(r2)
g2 = lap_pyramid(g2)
b2 = lap_pyramid(b2)
# print(len(r1))
# print(len(r2))


R = np.array(Weight1) * r1 + np.array(Weight2) * r2
G = np.array(Weight1) * g1 + np.array(Weight2) * g2
B = np.array(Weight1) * b1 + np.array(Weight2) * b2

R = collapse(R)
G = collapse(G)
B = collapse(B)

# 结果
R[R < 0] = 0
R[R > 255] = 255
R = np.array(R, np.uint8)
G[G < 0] = 0
G[G > 255] = 255
G = np.array(G, np.uint8)
B[B < 0] = 0
B[B > 255] = 255
B = np.array(B, np.uint8)

result = cv.merge([B, G, R])  # opencv的颜色通道顺序为BGR
plt.title("result")
plt.imshow(result)
# plt.savefig(title+'.jpg')
plt.show()
