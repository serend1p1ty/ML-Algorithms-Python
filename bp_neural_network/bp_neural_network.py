'''
@Author: 520Chris
@Date: 2019-09-21 16:12:24
@LastEditor: 520Chris
@LastEditTime: 2019-09-24 16:38:57
@Description: Implementation of BP neural network
'''

import random
import numpy as np
from pathlib import Path
from scipy import io as spio
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def display_data(img_data):
    '''显示图片'''
    sum = 0
    m, n = img_data.shape
    width = np.int32(np.round(np.sqrt(n)))
    height = np.int32(n / width)
    rows_count = np.int32(np.floor(np.sqrt(m)))
    cols_count = np.int32(np.ceil(m / rows_count))
    pad = 1
    display_array = - \
        np.ones((pad + rows_count * (height + pad),
                 pad + cols_count * (width + pad)))
    for i in range(rows_count):
        for j in range(cols_count):
            if sum >= m:
                break
            display_array[pad + i * (height + pad):pad + i * (height + pad) + height,
                          pad + j * (width + pad):pad + j * (width + pad) + width] = \
                img_data[sum, :].reshape(height, width, order="F")
            sum += 1
        if sum >= m:
            break
    plt.imshow(display_array, cmap='gray')
    plt.axis('off')
    plt.show()


class Layer:
    def __init__(self, unit_num=None):
        self.unit_num = unit_num
        self.theta = None
        self.gradient = 0
        self.output_data = None

    def initializer(self, pre_units):
        '''初始化参数矩阵和梯度矩阵

        Args:
            pre_units: 前一层的神经元的数目
        '''
        self.theta = np.random.normal(0, 0.5, (self.unit_num, pre_units + 1))
        self.gradient = np.zeros(self.theta.shape)

    def forward(self, input_data):
        '''在当前层执行一次前向传播

        Args:
            input_data: 输入层的数据

        Returns:
            输出层的结果
        '''
        input_data = np.vstack((input_data, np.ones((1, input_data.shape[1]))))
        self.output_data = sigmoid(np.dot(self.theta, input_data))
        return self.output_data

    def backward(self, delta, output_data_prev):
        '''在当前层执行一次后向传播

        Args:
            delta: 当前层的delta向量
            output_data_prev: 上一层的输出向量

        Returns:
            上一层的delta向量
        '''
        self.gradient += np.dot(delta, np.vstack((output_data_prev, 1)).T)
        delta_prev = np.dot(self.theta[:, :-1].T, delta) * \
            output_data_prev * (1 - output_data_prev)
        return delta_prev


class BPNN:
    def __init__(self, max_round=1000, learning_rate=0.3, regular_param=1):
        self.layers = []
        self.max_round = max_round
        self.learning_rate = learning_rate
        self.regular_param = regular_param

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward(self, x):
        '''神经网络执行一次前向传播'''
        for layer_id, layer in enumerate(self.layers):
            if not layer_id:
                output = layer.forward(x)
            else:
                output = layer.forward(output)
        return output

    def backward(self, x, y):
        '''神经网络执行一次后向传播'''
        layer_num = len(self.layers)
        for layer_id in reversed(range(layer_num)):
            layer = self.layers[layer_id]
            if layer_id == layer_num - 1:
                # 如果是输出层，初始的delta要特殊计算
                delta = layer.backward(layer.output_data - y,
                                       self.layers[layer_id - 1].output_data)
            elif layer_id == 0:
                # 如果是第一个隐藏层，前一层的输出即样本的各个属性值
                delta = layer.backward(delta, x)
            else:
                delta = layer.backward(delta, self.layers[layer_id - 1].output_data)

    def cal_loss(self, y_predict, y):
        '''计算损失'''
        pos = np.where(y == 1)[0]
        neg = np.where(y == 0)[0]
        return -(np.log(y_predict[pos]).sum() + np.log(1 - y_predict[neg]).sum())

    def check_gradient(self, x, y):
        '''通过数值法来验证计算的梯度是否准确'''
        for layer_id, layer in enumerate(self.layers):
            m, n = layer.theta.shape
            e = 1e-4
            for i in range(m):
                for j in range(n):
                    layer.theta[i, j] += e
                    output = self.forward(x)
                    loss1 = self.cal_loss(output, y)
                    layer.theta[i, j] -= 2 * e
                    output = self.forward(x)
                    loss2 = self.cal_loss(output, y)
                    gradient = (loss1 - loss2) / (2 * e)
                    layer.theta[i, j] += e
                    print("Gradient check for %s-th layer [%s][%s] program_grad: %s, real_grad: %s"
                          % (layer_id, i, j, layer.gradient[i][j], gradient))

    def fit(self, X, y):
        m = X.shape[0]

        for i, layer in enumerate(self.layers):
            if not i:
                layer.initializer(X.shape[1])
            else:
                layer.initializer(self.layers[i - 1].unit_num)

        for round in range(self.max_round):
            loss = 0
            for i in range(m):
                x_i = X[[i]].T
                y_i = y[[i]].T
                output = self.forward(x_i)
                loss += self.cal_loss(output, y_i)
                self.backward(x_i, y_i)

                # if not i:
                #     self.check_gradient(x_i, y_i)

            print("Loss at %s-th epoch: %s" % (round, loss))

            for layer in self.layers:
                layer.gradient /= m
                layer.gradient[:, :-1] += self.regular_param / m * layer.theta[:, :-1]
                layer.theta -= self.learning_rate * layer.gradient
                layer.gradient = np.zeros(layer.theta.shape)

    def predict(self, X):
        X = X.T
        for layer_id, layer in enumerate(self.layers):
            if not layer_id:
                output = layer.forward(X)
            else:
                output = layer.forward(output)
        return output.T

if __name__ == "__main__":
    base_path = Path(__file__).parent
    file_path = (base_path / "data_digits.mat").resolve()
    data_img = spio.loadmat(file_path)
    X = data_img['X']
    y = data_img['y']

    # 随机展示100张图片
    random_rows = random.sample(range(X.shape[0]), 100)
    display_data(X[random_rows])

    X_train, X_test, y_train_origin, y_test_origin = train_test_split(X,
                                                                      y,
                                                                      test_size=0.3,
                                                                      random_state=0)

    ohe = OneHotEncoder(categories='auto')
    ohe.fit(y)
    y_train = ohe.transform(y_train_origin).toarray()
    y_test = ohe.transform(y_test_origin).toarray()

    bpnn = BPNN(max_round=1000, learning_rate=1, regular_param=10)
    bpnn.add_layer(Layer(25))
    bpnn.add_layer(Layer(10))
    bpnn.fit(X_train, y_train)

    p = bpnn.predict(X_train)
    p = ohe.inverse_transform(p)
    accuracy = 100 * (p == y_train_origin).sum() / len(p)
    print("-----Accuracy in train-set: %s%%" % accuracy)

    p = bpnn.predict(X_test)
    p = ohe.inverse_transform(p)
    accuracy = 100 * (p == y_test_origin).sum() / len(p)
    print("-----Accuracy in test-set: %s%%" % accuracy)
