import numpy as np
import cv2 as cv
import func
import math
from sklearn import metrics
# 精度
eps = 1e-10
# 迭代次数
T = 10000

def pca_2d_2d(pics):
    pics = np.array(pics)
    size = pics[0].shape

    # 取得归一化后的所有图片对应位置像素的均值
    mean = np.average(pics, axis=0)

    cov_row = np.zeros((size[1], size[1]))
    for i in range(pics.shape[0]):
        # 每张图片减去所有图片的均值
        diff = pics[i] - mean
        # 生成协方差矩阵
        cov_row = cov_row + np.dot(diff.T, diff)
    cov_row = cov_row/cov_row.shape[0]

    m, row_evec, row_eval = func.jacobi(cov_row, 256, eps, T)
    # 返回协方差矩阵特征值与特征向量
    # 直接调用现有库计算特征值与特征向量
    # row_eval, row_evec = np.linalg.eig(cov_row)
    # 将特征值从小到大排列，并将索引赋值给sorted_index
    row_evec, row_eval = func.sort_eigen(row_evec, row_eval)
    alpha = 0.95
    k = func.choose_dim(row_eval, alpha)
    # print(k)
    # k = 100
    x = row_evec[:, 0:k]

    # 将图片的宽为新生方阵的m
    cov_col = np.zeros((size[0], size[0]))
    for i in range(pics.shape[0]):
        diff = pics[i] - mean
        cov_col += np.dot(diff, diff.T)
    cov_col = cov_col/cov_col.shape[0]
    # 直接调包计算特征值与特征向量
    col_eval, col_evec = np.linalg.eig(cov_col)
    alpha = 0.95
    row_evec, row_eval = func.sort_eigen(row_evec, row_eval)
    k = func.choose_dim(col_eval, alpha)
    # k = 20
    z = row_evec[:, 0:k]
    print(k)
    return x, z


path = "F:/PCA_Project/Images/all_pics"
channels = func.read_pic_split(path)
pic_de = []
number = len(channels[0])

error = [0, 0, 0]
rmse = [0, 0, 0]


# 对RGB三个通道分别降维
for c in range(3):

    trans_X, Z = pca_2d_2d(channels[c])
    """"
    p = np.dot(Z.T, np.dot(channels[c][150], trans_X))
    res = np.dot(Z, np.dot(p, trans_X.T))
    pic_de.append(res)
    """
    for i in range(number):
        # 图片降维
        p = np.dot(Z.T, np.dot(channels[c][0], trans_X))
        print(p.shape)
        # 图片重构
        res = np.dot(Z, np.dot(p, trans_X.T))
        print(res.shape)
        temp = np.sum(np.dot((res - channels[c][i]), (res - channels[c][i]).T))
        error[c] += math.sqrt(temp)
        rmse[c] += metrics.mean_squared_error(res, channels[c][i])


print(error)
print(rmse)
error_total = sum(error)
print(error_total/number)
print(sum(rmse)/number)

"""
rgb_img = np.dstack((pic_de[0], pic_de[1], pic_de[2])).astype('uint8')
print(rgb_img.shape)
cv.imwrite('F:/PCA_img/plane/2d2d_pca_100.jpg', rgb_img)"""
