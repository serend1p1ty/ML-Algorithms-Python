"""
@Author: 520Chris
@Date: 2019-10-10 21:07:21
@LastEditor: 520Chris
@LastEditTime: 2019-10-11 16:31:34
@Description: SVM experimentation
"""

from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from scipy import io as spio
from sklearn import svm


def SVM():
    base_path = Path(__file__).parent
    file_path1 = (base_path / "data1.mat").resolve()
    file_path2 = (base_path / "data2.mat").resolve()

    # data1: 线性分类
    data1 = spio.loadmat(file_path1)
    X = data1["X"]
    y = data1["y"]
    y = np.ravel(y)
    plot = plot_data(X, y)
    plot.show()
    model = svm.SVC(C=1.0, kernel="linear").fit(X, y)  # 线性核函数
    plot_decision_boundary(X, y, model)

    # data2: 非线性分类
    data2 = spio.loadmat(file_path2)
    X = data2["X"]
    y = data2["y"]
    y = np.ravel(y)
    plot = plot_data(X, y)
    plot.show()
    model = svm.SVC(gamma=100).fit(X, y)  # gamma为核函数的系数, 值越大拟合的越好
    plot_decision_boundary(X, y, model, class_="notLinear")


def plot_data(X, y):
    plt.figure(figsize=(10, 8))
    pos = np.where(y == 1)
    neg = np.where(y == 0)
    p1, = plt.plot(np.ravel(X[pos, 0]), np.ravel(X[pos, 1]), "ro", markersize=8)
    p2, = plt.plot(np.ravel(X[neg, 0]), np.ravel(X[neg, 1]), "g^", markersize=8)
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.legend([p1, p2], ["y==1", "y==0"])
    return plt


def plot_decision_boundary(X, y, model, class_="linear"):
    plot = plot_data(X, y)

    if class_ == "linear":
        # 线性边界
        w = model.coef_
        b = model.intercept_
        x_val = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 100)
        y_val = -(w[0, 0] * x_val + b) / w[0, 1]
        plot.plot(x_val, y_val, "b-", linewidth=2.0)
        plot.show()
    else:
        # 非线性边界
        x_range = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 100)
        y_range = np.linspace(np.min(X[:, 1]), np.max(X[:, 1]), 100)
        X, Y = np.meshgrid(x_range, y_range)
        predict = np.zeros(X.shape)

        for i in range(X.shape[1]):
            X_part = np.hstack((X[:, i].reshape(-1, 1), Y[:, i].reshape(-1, 1)))
            predict[:, i] = model.predict(X_part)

        plot.contour(X, Y, predict, [0.5])
        plot.show()


if __name__ == "__main__":
    SVM()
