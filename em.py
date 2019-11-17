#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ----------------------------------------
# Author: 520Chris
# Date: 2019-10-24 15:38:13
# LastEditor: 520Chris
# LastEditTime: 2019-11-17 09:26:32
# Description: Implementation of expectation-maximum algorithm
# ----------------------------------------

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal


class EM:
    """EM算法

    Attributes:
        k: 混合的高斯模型的数目
        z: 隐变量的概率分布
        data: 训练数据
        mean: 混合高斯模型的均值
        cov: 混合高斯模型的协方差矩阵
        prior_prob: 各个类别的先验概率
    """

    def __init__(self, k=2):
        self.k = k
        self.z = None
        self.data = None
        self.mean = None
        self.cov = None
        self.prior_prob = None

    def init(self, data):
        m, n, k = *data.shape, self.k
        self.data = data
        self.z = np.zeros((m, k))
        self.mean = np.random.random((k, n))
        self.cov = np.array([np.identity(n) for i in range(k)])
        self.prior_prob = np.ones(k) / k

    def e_step(self):
        m, k = self.data.shape[0], self.k
        for i in range(m):
            total = 0
            for j in range(k):
                self.z[i][j] = (
                    multivariate_normal.pdf(self.data[i, :], self.mean[j], self.cov[j])
                    * self.prior_prob[j]
                )
                total += self.z[i][j]
            self.z[i] /= total

    def m_step(self):
        m, n, k = *self.data.shape, self.k
        for j in range(k):
            c = self.z[:, j].sum()
            self.prior_prob[j] = c / m
            mean = np.zeros(n)
            cov = np.zeros((n, n))
            for i in range(m):
                mean += self.z[i][j] * self.data[i]
                tmp = (self.data[i] - self.mean[j]).reshape(1, -1)  # 将一维数组转化成二维数组
                cov += self.z[i][j] * np.dot(tmp.T, tmp)
            self.mean[j] = mean / c
            self.cov[j] = cov / c

    def fit(self, data):
        self.init(data)
        theta = np.hstack((self.mean.flatten(), self.cov.flatten(), self.prior_prob))
        previous_theta = None
        eps = 1e-4  # 算法停止条件
        i = 0
        while True:
            i += 1
            print("-----Round %s" % i)
            self.e_step()
            self.m_step()
            previous_theta = theta
            theta = np.hstack((self.mean.flatten(), self.cov.flatten(), self.prior_prob))

            if np.linalg.norm(theta - previous_theta) <= eps:
                break


def main():
    # 生成两个高斯分布的数据
    N1 = 200
    N2 = 500
    mean1 = [0, 3]
    mean2 = [20, 10]
    cov1 = [[0.5, 0], [0, 0.8]]
    cov2 = np.identity(2)
    np.random.seed(8)  # 每次产生一样的随机数据
    data1 = np.random.multivariate_normal(mean1, cov1, N1)
    data2 = np.random.multivariate_normal(mean2, cov2, N2)
    data = np.vstack((data1, data2))

    em = EM()
    em.fit(data)

    # 数据可视化
    label = np.argmax(em.z, axis=1)
    pos = data[label == 1]
    neg = data[label == 0]
    plt.subplot(121)
    plt.scatter(data[:, 0], data[:, 1])
    plt.subplot(122)
    plt.scatter(pos[:, 0], pos[:, 1], c="r")
    plt.scatter(neg[:, 0], neg[:, 1], c="y")
    plt.show()

    print("True mean1: %s" % mean1)
    print("Estimated mean1: %s" % em.mean[0])
    print("True mean2: %s" % mean2)
    print("Estimated mean2: %s" % em.mean[1])
    print("True cov1: %s" % cov1)
    print("Estimated cov1: %s" % em.cov[0])
    print("True cov2: %s" % cov2)
    print("Estimated cov2: %s" % em.cov[1])


if __name__ == "__main__":
    main()
