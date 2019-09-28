'''
@Author: 520Chris
@Date: 2019-09-26 19:46:27
@LastEditor: 520Chris
@LastEditTime: 2019-09-28 17:20:43
@Description: Implementation of naive bayes classifier
'''

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn import datasets
from collections import Counter
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedKFold


def read_data():
    '''数据预处理'''
    base_path = Path(__file__).parent
    file_path = (base_path / "adult.csv").resolve()
    data = pd.read_csv(file_path, header=None)
    data.columns = [
        "Age", "WorkClass", "fnlwgt", "Education", "EducationNum",
        "MaritalStatus", "Occupation", "Relationship", "Race", "Gender",
        "CapitalGain", "CapitalLoss", "HoursPerWeek", "NativeCountry", "Income"
    ]
    y = data["Income"].values
    X = data.drop("Income", axis=1).values
    return X, y


class NaiveBayesClassifier():
    def cal_prob(self, y):
        '''计算每个值出现的频率'''
        prior_prob = Counter(y)
        for key in prior_prob:
            prior_prob[key] /= len(y)
        return prior_prob

    def fit(self, X, y):
        self.prior_prob = self.cal_prob(y)
        self.likelihood = {}
        for class_id in self.prior_prob:
            for attr in range(X.shape[1]):
                if isinstance(X[0][attr], str):
                    # 离散属性：直接统计频率
                    self.likelihood[(class_id, attr)] = self.cal_prob(X[y == class_id, attr])
                else:
                    # 连续属性：计算高斯分布模型的均值和方差
                    mean = np.mean(X[y == class_id, attr])
                    var = np.var(X[y == class_id, attr])
                    self.likelihood[(class_id, attr)] = (mean, var)

    def cal_likelihood(self, class_id, attr, x):
        # 离散属性
        if isinstance(x, str):
            prob = self.likelihood[(class_id, attr)]
            if x in prob:
                return prob[x]
            else:
                # 如果当前属性的取值没有在训练集出现过，则似然概率为0.5
                return 0.5

        # 连续属性
        mean, var = self.likelihood[(class_id, attr)]
        return 1 / ((2 * np.pi) ** 0.5 * var ** 0.5) \
            * np.exp(-(x - mean) ** 2 / (2 * var))

    def predict(self, X):
        p = []
        for x in X:
            max_posterior = -1
            predict_class = None
            for class_id in self.prior_prob:
                prior_prob = self.prior_prob[class_id]
                like_prob = 1
                for attr in range(X.shape[1]):
                    like_prob *= self.cal_likelihood(class_id, attr, x[attr])

                # 后验概率最大的类别即为预测结果
                if prior_prob * like_prob > max_posterior:
                    max_posterior = prior_prob * like_prob
                    predict_class = class_id
            p.append(predict_class)
        return np.array(p)

if __name__ == "__main__":
    X, y = read_data()
    train_res, test_res = [], []

    # 10次10折交叉验证
    kf = RepeatedKFold(n_splits=10, n_repeats=10, random_state=0)
    for train_index, test_index in kf.split(X):
        X_train = X[train_index]
        X_test = X[test_index]
        y_train = y[train_index]
        y_test = y[test_index]

        naive_bayes = NaiveBayesClassifier()
        naive_bayes.fit(X_train, y_train)

        p = naive_bayes.predict(X_train)
        accuracy = accuracy_score(p, y_train) * 100
        train_res.append(accuracy)
        print("---Accuracy in train-set: %s%%" % accuracy)

        p = naive_bayes.predict(X_test)
        accuracy = accuracy_score(p, y_test) * 100
        test_res.append(accuracy)
        print("---Accuracy in test-set: %s%%" % accuracy)

    mean_train = np.mean(train_res)
    std_train = np.var(train_res) ** 0.5
    mean_test = np.mean(test_res)
    std_test = np.var(test_res) ** 0.5
    print("Train-set: %s%% (std: %s)\nTest-set: %s%% (std: %s)" % (mean_train, std_train,
                                                                   mean_test, std_test))
