import numpy as np
import cv2 as cv
import func
import math
from sklearn import metrics
# 精度
eps = 1e-10
# 迭代次数
T = 10000


def pca_2d(pics):
    pics = np.array(pics)
    size = pics[0].shape
    # 取得归一化后的所有图片对应位置像素相加之后的均值
    mean = np.average(pics, axis=0)

    # 256*256
    cov_row = np.zeros((size[1], size[1]))
    for i in range(pics.shape[0]):
        # 每张图片减去所有图片的均值
        diff = pics[i] - mean
        # 生成协方差矩阵
        cov_row = cov_row + np.dot(diff.T, diff)
    cov_row = cov_row/cov_row.shape[0]

    # 返回协方差矩阵特征值与特征向量
    # 2DPCA直接掉包实现
    m, row_evec, row_eval = func.jacobi(cov_row, 256, eps, T)
    # row_eval, row_evec = np.linalg.eig(cov_row)
    # 将特征值从大到小排列
    row_evec, row_eval = func.sort_eigen(row_evec, row_eval)
    alpha = 0.95
    k = func.choose_dim(row_eval, alpha)
    print(k)
    # k = 20
    x = row_evec[:, 0:k]
    return x


path = "F:/PCA_Project/Images/all_pics"
channels = func.read_pic_split(path)
number = len(channels[0])
pic_de = []

# blue green red
# error = [0, 0, 0]
# rmse = [0, 0, 0]

for c in range(3):
    # 得到每个通道的变换矩阵
    trans_X = pca_2d(channels[c])
    # 图片降维
    p = np.dot(channels[c][150], trans_X)
    # 图片重构
    res = np.dot(p, trans_X.T)
    pic_de.append(res)
    """
    for i in range(number):
        # c[0]提取了第一张图片的蓝色通道
        # 图像降维
        p = np.dot(channels[c][i], trans_X)
        print(p.shape)
        # 图像重构
        res = np.dot(p, trans_X.T)
        # print(res.shape)
        # res为图像重构后的结果
        # 重构误差
        temp = np.sum(np.square(res - channels[c][i]))
        # 欧式距离
        error[c] += math.sqrt(temp)

        rmse[c] += metrics.mean_squared_error(res, channels[c][i])
        """

"""
print(error)
print(rmse)ds

error_total = sum(error)
print(error_total/number)
print(sum(rmse)/number)"""

# print(rmse_total)
rgb_img = np.dstack((pic_de[0], pic_de[1], pic_de[2])).astype('uint8')
cv.imwrite('F:/PCA_img/plane/2d_pca_best.jpg', rgb_img)

