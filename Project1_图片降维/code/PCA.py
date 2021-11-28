import numpy as np
import func
import cv2 as cv
from sklearn import metrics

# 精度
eps = 1e-10
# 迭代次数
T = 10000


def pca(pic_matrix):
    # m表示图片的特征数
    m = pic_matrix.shape[0]
    # n表示图片的数量，即样本数
    n = pic_matrix.shape[1]

    # -----------------对数据做中心化操作----------------------#
    # --------------矩阵的每一行特征减去其所在行的均值------------#
    x_avg = np.average(pic_matrix, axis=1).reshape(m, 1)
    pic_matrix_center = (pic_matrix - x_avg)

    # --------------求P'·P的特征值与特征向量-------------------#
    # 不必求协方差矩阵的特征值与特征向量，因为图片的维度过高，难以存储下 Memory Over Flow
    p = pic_matrix_center

    # 可以不用求系数，常数不会影响
    matrix = np.dot(p.T, p)
    mat, vector, value = func.jacobi(matrix, n, eps, T)
    # value, vector = np.linalg.eig(matrix)
    eig_value = abs(np.array(value))
    # -------求奇异值构成的对角阵-------#
    sin_value = np.diag(np.sqrt(eig_value))
    # 特征向量
    vector = np.array(vector)

    # ------将特征向量归一化-------------#
    # np.linalg.norm 求解矩阵二范数
    # 对称矩阵，其特征向量是正交的
    vector = vector / np.linalg.norm(vector, axis=0)
    w = np.dot(p, np.linalg.inv(np.dot(sin_value, vector.T)))
    # 得到变换矩阵W
    w = w.T

    # -----------------将特征值和特征向量按照从大到小排列----------------#
    w, eig_value = func.sort_eigen(w, eig_value)

    # -------------选取特征值中前k大的-------------#
    # 阈值alpha限定主成分对原数据的解释能力，通常设在90%
    # 调用choose_dim确定k的大小，即图片降维后的维度
    alpha = 0.95
    k = func.choose_dim(eig_value, alpha)
    # print(k)
    # k = 20
    trans_matrix = w[0:k, :]
    return trans_matrix


path = "F:/PCA_Project/Images/all_pics"
# ----------------------完成了图片的读入，并以矩阵的方式表示------------------#
# ------------------矩阵的大小为m*n，n为图片的数量，m为特征数-----------------#
pic_Matrix = func.read_pic_vector(path)
trans_Matrix = pca(pic_Matrix)
print(trans_Matrix.shape)
# 得到降维后的图片矩阵
pic_PCA = np.dot(trans_Matrix, pic_Matrix)
print(pic_PCA.shape)
# 通过降维后的图片进行重构
pic_PCA_reconstruct = np.dot(trans_Matrix.T, pic_PCA)
print(pic_PCA_reconstruct.shape)
# pic_0 = pic_PCA_reconstruct[:, 0].reshape(256, 256, 3)
# cv.imwrite('F:/PCA_img/pca_vetor_01.jpg', pic_0)

# 196608*100
"""
number = pic_Matrix.shape[1]
temp = np.square(pic_PCA_reconstruct - pic_Matrix)
error_temp = np.sqrt(np.sum(temp, axis=0))
error_average = np.sum(error_temp)/number
print(error_average)

rmse = 0
for i in range(number):
    rmse += metrics.mean_squared_error(pic_PCA_reconstruct[:, i], pic_Matrix[:, i])
print(rmse/number)"""

pic_0 = pic_PCA_reconstruct[:, 150].reshape(256, 256, 3)
cv.imwrite('F:/PCA_img/pca_vetor_best.jpg', pic_0)
