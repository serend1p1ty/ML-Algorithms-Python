#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ----------------------------------------
# Author: 520Chris
# Date: 2019-10-11 16:34:44
# LastEditor: 520Chris
# LastEditTime: 2019-11-17 09:28:35
# Description: Implementation of BP neural network
# ----------------------------------------

from pathlib import Path

import numpy as np
from scipy import io as spio
from scipy import optimize
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class Layer:
    """神经网络的层级结构

    Attributes:
        unit_num: 神经元的数目
        theta: 参数矩阵
        gradient: 梯度矩阵
        output_data: 当前层的输出
    """

    def __init__(self, unit_num=None):
        self.unit_num = unit_num
        self.theta = None
        self.gradient = None
        self.output_data = None

    def initializer(self, pre_units):
        """初始化参数矩阵和梯度矩阵(注：参数矩阵的初始值非常重要)

        Args:
            pre_units: 前一层的神经元的数目
        """
        self.theta = np.random.normal(0, 0.5, (self.unit_num, pre_units + 1))
        self.gradient = np.zeros(self.theta.shape)

    def forward(self, input_data):
        """在当前层执行一次前向传播

        Args:
            input_data: 输入向量

        Returns:
            输出向量
        """
        input_data = np.vstack((input_data, np.ones((1, input_data.shape[1]))))
        self.output_data = sigmoid(np.dot(self.theta, input_data))
        return self.output_data

    def backward(self, delta, output_data_prev):
        """在当前层执行一次后向传播

        Args:
            delta: 当前层的delta向量
            output_data_prev: 上一层的输出向量

        Returns:
            上一层的delta向量
        """
        self.gradient += np.dot(delta, np.vstack((output_data_prev, 1)).T)
        delta_prev = np.dot(self.theta[:, :-1].T, delta) * output_data_prev * (1 - output_data_prev)
        return delta_prev


class BPNN:
    """BP神经网络

    Attributes:
        layers: 存储着所有层的列表
        max_round: 最大迭代次数
        regular_param: 正则化参数
    """

    def __init__(self, max_round=500, regular_param=1):
        self.layers = []
        self.max_round = max_round
        self.regular_param = regular_param

    def add_layer(self, layer):
        self.layers.append(layer)

    def init_layers(self, thetas):
        for layer in self.layers:
            layer.theta = thetas[0 : layer.theta.size].reshape(layer.theta.shape)
            thetas = thetas[layer.theta.size :]

    def forward(self, x):
        """神经网络执行一次前向传播"""
        for layer_id, layer in enumerate(self.layers):
            if not layer_id:
                output = layer.forward(x)
            else:
                output = layer.forward(output)
        return output

    def backward(self, x, y):
        """神经网络执行一次后向传播"""
        layer_num = len(self.layers)
        for layer_id in reversed(range(layer_num)):
            layer = self.layers[layer_id]
            if layer_id == layer_num - 1:
                # 如果是输出层, 初始的delta要特殊计算
                delta = layer.backward(layer.output_data - y, self.layers[layer_id - 1].output_data)
            elif layer_id == 0:
                # 如果是第一个隐藏层, 前一层的输出即样本的各个属性值
                delta = layer.backward(delta, x)
            else:
                delta = layer.backward(delta, self.layers[layer_id - 1].output_data)

    def cal_gradient(self, thetas, X, y):
        """计算梯度"""
        m = X.shape[0]
        self.init_layers(thetas)

        for i in range(m):
            x_i = X[[i]].T
            y_i = y[[i]].T
            self.forward(x_i)
            self.backward(x_i, y_i)

        gradient = np.array([])
        for layer in self.layers:
            layer.gradient /= m
            layer.gradient[:, :-1] += self.regular_param / m * layer.theta[:, :-1]
            gradient = np.append(gradient, layer.gradient)
            layer.gradient = np.zeros(layer.theta.shape)

        return gradient

    def cal_loss(self, thetas, X, y):
        """计算代价函数的值"""
        m = X.shape[0]
        self.init_layers(thetas)
        all_loss = 0
        for i in range(m):
            x_i = X[[i]].T
            y_i = y[[i]].T
            output = self.forward(x_i)
            all_loss += self.loss(output, y_i)
        print("Loss: %s" % (all_loss / m))
        return all_loss / m

    @staticmethod
    def loss(y_predict, y):
        """计算损失"""
        return -(np.log(y_predict[y == 1]).sum() + np.log(1 - y_predict[y == 0]).sum())

    def check_gradient(self, thetas, X, y):
        """通过数值法来验证计算的梯度是否准确"""
        e = 1e-4
        gradient = np.array([])
        for i, _ in enumerate(thetas):
            thetas[i] += e
            loss1 = self.cal_loss(thetas, X, y)
            thetas[i] -= 2 * e
            loss2 = self.cal_loss(thetas, X, y)
            gradient = np.append(gradient, (loss1 - loss2) / (2 * e))
            thetas[i] += e
        return gradient

    def fit(self, X, y):
        initial_thetas = np.array([])

        for i, layer in enumerate(self.layers):
            if not i:
                layer.initializer(X.shape[1])
            else:
                layer.initializer(self.layers[i - 1].unit_num)
            initial_thetas = np.append(initial_thetas, layer.theta.flatten())

        # 用数值法验证计算的准确性
        # num_grad = self.check_gradient(initial_thetas, X[0].reshape(1, -1), y[0].reshape(1, -1))
        # grad = self.cal_gradient(initial_thetas, X[0].reshape(1, -1), y[0].reshape(1, -1))
        # print(np.linalg.norm(num_grad - grad) / np.linalg.norm(num_grad + grad))  # 结果应该小于1e-9

        print("-----Begin training")
        thetas = optimize.fmin_cg(
            self.cal_loss,
            initial_thetas,
            fprime=self.cal_gradient,
            args=(X, y),
            maxiter=self.max_round,
        )

        self.init_layers(thetas)

    def predict(self, X):
        X = X.T
        for layer_id, layer in enumerate(self.layers):
            if not layer_id:
                output = layer.forward(X)
            else:
                output = layer.forward(output)
        return (output == output.max(axis=0)).astype(int).T


def main():
    base_path = Path(__file__).parent
    file_path = (base_path / "digits.mat").resolve()
    data_img = spio.loadmat(file_path)
    X = data_img["X"]
    y_origin = data_img["y"]

    ohe = OneHotEncoder(categories="auto")
    y = ohe.fit_transform(y_origin).toarray()

    bpnn = BPNN()
    bpnn.add_layer(Layer(25))
    bpnn.add_layer(Layer(10))
    bpnn.fit(X, y)

    p = bpnn.predict(X)
    p = ohe.inverse_transform(p)
    accuracy = 100 * accuracy_score(y_origin, p)
    print("-----Accuracy in train-set: %s%%" % accuracy)


if __name__ == "__main__":
    main()
