import numpy as np
from math import *
import cv2 as cv
import os


# 将图片的三个通道直接拉为一个向量
def read_pic_vector(path):
    path_dir = os.listdir(path)

    lst = []
    for pic in path_dir:
        img = cv.imread((path + "/" + pic))
        if img.shape == (256, 256, 3):
            img = img.flatten()
            lst.append(img)
    pic_matrix = np.array(lst).T
    return pic_matrix


# 将图片的RGB三个通道分离，每个通道拉为一个向量
def read_pic_vector_split(path):
    # 得到文件夹中所有图片名
    path_dir = os.listdir(path)

    channel = []
    lst_red = []
    lst_blue = []
    lst_green = []
    for pic in path_dir:
        # 256 * 256 * 3
        img = cv.imread((path + "/" + pic))
        if img.shape == (256, 256, 3):
            blue, green, red = cv.split(img)
            # print(img.shape)
            # 将图像拉为行向量
            blue = blue.flatten()
            green = green.flatten()
            red = red.flatten()
            # 找出集合中维度不同的图片
            lst_red.append(red)
            lst_green.append(green)
            lst_blue.append(blue)

    # 得到图片矩阵
    channel.append(np.array(lst_blue).T)
    channel.append(np.array(lst_green).T)
    channel.append(np.array(lst_red).T)
    return channel


# 将图片的三个通道分离，不用拉为向量
def read_pic_split(path):
    # 得到文件夹中所有图片名
    path_dir = os.listdir(path)
    channel = []
    lst_red = []
    lst_blue = []
    lst_green = []

    for pic in path_dir:
        img = cv.imread((path + "/" + pic))
        if img.shape == (256, 256, 3):
            blue, green, red = cv.split(img)
            lst_blue.append(blue)
            lst_green.append(green)
            lst_red.append(red)
    channel.append(lst_blue)
    channel.append(lst_green)
    channel.append(lst_red)
    return channel


# 计算特征向量与特征值
def jacobi(matrix, dim, precision, iteration_max):
    # 将特征向量初始化为单位阵
    eigenvectors = np.eye(dim)
    iter_count = 0
    while 1:
        db_max = matrix[0][1]
        n_row = 0
        n_col = 1
        for i in range(dim):
            for j in range(dim):
                d = fabs(matrix[i][j])
                if i != j and d > db_max:
                    db_max = d
                    n_row = i
                    n_col = j
        if db_max < precision:
            break
        if iter_count > iteration_max:
            break
        iter_count = iter_count + 1
        s_ii = matrix[n_row][n_row]
        s_jj = matrix[n_col][n_col]
        s_ij = matrix[n_row][n_col]
        theta = 0.5*atan2((-2) * s_ij, (s_jj - s_ii))
        sin_theta = sin(theta)
        cos_theta = cos(theta)
        sin_2theta = sin(2*theta)
        cos_2theta = cos(2*theta)
        matrix[n_row][n_row] = s_ii*cos_theta*cos_theta + s_jj * sin_theta * sin_theta + 2*s_ij*cos_theta*sin_theta
        matrix[n_col][n_col] = s_ii*sin_theta*sin_theta + s_jj * cos_theta * cos_theta - 2*s_ij*cos_theta*sin_theta
        matrix[n_row][n_col] = 0.5*(s_jj - s_ii)*sin_2theta + s_ij*cos_2theta
        matrix[n_col][n_row] = matrix[n_row][n_col]
        for i in range(dim):
            if i != n_row and i != n_col:
                db_max = matrix[i][n_row]
                matrix[i][n_row] = matrix[i][n_col] * sin_theta + db_max*cos_theta
                matrix[i][n_col] = matrix[i][n_col] * cos_theta - db_max*sin_theta

        for j in range(dim):
            if j != n_col and j != n_row:
                db_max = matrix[n_row][j]
                matrix[n_row][j] = matrix[n_col][j] * sin_theta + db_max*cos_theta
                matrix[n_col][j] = matrix[n_col][j] * cos_theta - db_max*sin_theta

        # compute eigenvector
        for i in range(dim):
            db_max = eigenvectors[i][n_row]
            eigenvectors[i][n_row] = eigenvectors[i][n_col]*sin_theta + db_max*cos_theta
            eigenvectors[i][n_col] = eigenvectors[i][n_col]*cos_theta - db_max*sin_theta

    # 取matrix的对角元素，得到特征值
    eigenvalues = np.diagonal(matrix)
    for i in range(dim):
        if np.sum(eigenvectors[:, i]) < 0:
            eigenvectors[:, i] *= -1

    return matrix, eigenvectors, eigenvalues


# 根据阈值的大小，选取前k个特征向量
def choose_dim(eigenvalues, alpha):
    threshold = np.sum(eigenvalues)*alpha
    total = 0
    for i in range(len(eigenvalues)):
        total = total + eigenvalues[i]
        if total >= threshold:
            return i+1
    return len(eigenvalues)


# 将特征值从大到小进行排序
# np.argsort返回元素从小到大排列的索引值，添加负号，返回元素从大到小的排列#
def sort_eigen(w, eig_value):
    sort_index = np.argsort(-eig_value)
    temp = eig_value.copy()
    for i in range(eig_value.shape[0]):
        eig_value[i] = temp[sort_index[i]]
    temp = w.copy()
    for i in range(w.shape[0]):
        w[i, :] = temp[sort_index[i], :]
    return w, eig_value