import numpy as np
import func
import cv2 as cv
from sklearn import metrics
# 精度
eps = 1e-10
# 迭代次数
T = 10000


# pic_matrix的大小为m*n, m表示图片的特征数，n表示样本数
def pca_split(pic_matrix):
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
    print(k)
    # k = 20
    trans_matrix = w[0:k, :]
    return trans_matrix

# ----------------------完成了图片的读入，并以矩阵的方式表示------------------#
# ------------------矩阵的大小为m*n，n为图片的数量，m为特征数-----------------#


path = "F:/PCA_Project/Images/all_pics"

# -------读入图片，并以矩阵的形式存储----------#
# 3个(256*256)*100的矩阵
channels = func.read_pic_vector_split(path)
pic_de = []
number = channels[0].shape[1]
# print(channels[0].shape)
for c in range(3):
    trans_Matrix = pca_split(channels[c])
    # 得到降维后的图片矩阵
    pic_PCA = np.dot(trans_Matrix, channels[c])
    # print(pic_PCA.shape)
    # 图片重构
    pic_PCA_reconstruct = np.dot(trans_Matrix.T, pic_PCA)
    # print(pic_PCA_reconstruct.shape)
    pic_de.append(pic_PCA_reconstruct)

"""
print(channels[0].shape)
print(pic_de[0].shape)

error_avg = [0, 0, 0]
rmse = [0, 0, 0]
for i in range(3):
    temp = np.square(pic_de[i] - channels[i])
    error_temp = np.sqrt(np.sum(temp, axis=0))
    error_avg[i] = np.sum(error_temp)/number
    temp = 0
    for j in range(number):
        temp += metrics.mean_squared_error(pic_de[i][:, j], channels[i][:, j])
    rmse[i] = temp/number

print(sum(error_avg))
print(sum(rmse))


"""
# 通过降维后的图片进行重构
# 将RGB三个通道进行重构，再合并为一张图
pic_0_b = pic_de[0][:, 150].reshape(256, 256)
pic_1_b = pic_de[1][:, 150].reshape(256, 256)
pic_2_b = pic_de[2][:, 150].reshape(256, 256)
rgb_img = np.dstack((pic_0_b, pic_1_b, pic_2_b)).astype('uint8')
cv.imwrite('F:/PCA_img/plane/pca_agri_20.jpg', rgb_img)

