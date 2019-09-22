'''
@Author: 520Chris
@Date: 2019-09-18 18:48:38
@LastEditor: 520Chris
@LastEditTime: 2019-09-22 11:31:13
@Description: Implementation of decision tree
'''

from collections import Counter
import numpy as np


class DecisionTree:
    def __init__(self, evaluate_criterion='information_gain'):
        self.attribute = None  # 当前节点要根据什么属性划分集合
        self.threshold = None  # 用来划分集合的属性为连续属性时，划分集合所使用的阈值
        self.result = None     # 当前节点为叶子节点时，预测的结果
        self.children = {}     # 子节点
        if evaluate_criterion == 'information_gain':
            self.evaluate_func = self.max_info_gain
        else:
            raise NotImplementedError("%s hasn't been implemented" % evaluate_criterion)

    @staticmethod
    def entropy(X, y):
        entro = 0
        cnt = Counter(y)
        for key in cnt:
            p = cnt[key] / len(y)
            entro += -p * np.log2(p)
        return entro

    @staticmethod
    def index_by_list(list1, list2):
        return [list1[i] for i in list2]

    def devide_by_attribute(self, X, y, attribute):
        subsets = {}
        threshold = None

        if isinstance(X[0][attribute], str):
            # 离散属性
            for i in range(len(X)):
                if X[i][attribute] in subsets.keys():
                    subsets[X[i][attribute]].append(i)
                else:
                    subsets[X[i][attribute]] = [i]
        else:
            # 连续属性
            column = [row[attribute] for row in X]
            column_copy = column.copy()
            column_copy.sort()

            # 获得所有候选的分割阈值
            candidates = []
            for i in range(len(column_copy) - 1):
                candidates.append((column_copy[i] + column_copy[i + 1]) / 2)

            index = self.index_by_list  # 简化函数名称
            base_entro = self.entropy(X, y)
            max_gain = -1
            subsets = {}

            # 选择最大化信息增益的阈值
            for candidate in candidates:
                split_func = lambda i: column[i] <= candidate
                part1 = [i for i in range(len(column)) if split_func(i)]
                part2 = [i for i in range(len(column)) if not split_func(i)]
                p = len(part1) / len(column)
                entro_sum = p * self.entropy(index(X, part1), index(y, part1)) \
                    + (1 - p) * self.entropy(index(X, part2), index(y, part2))
                if base_entro - entro_sum > max_gain:
                    max_gain = base_entro - entro_sum
                    subsets['smaller'] = part1
                    subsets['larger'] = part2
                    threshold = candidate

        return subsets, threshold

    def max_info_gain(self, X, y, attributes):
        '''选择最大化信息增益的属性'''
        index = self.index_by_list
        base_entro = self.entropy(X, y)
        max_gain = -1
        best_attr = None
        best_threshold = None
        for attrID in attributes:
            subsets, threshold = self.devide_by_attribute(X, y, attrID)
            entro_sum = 0
            for key in subsets:
                p = len(subsets[key]) / len(X)
                entro_sum += p * self.entropy(index(X, subsets[key]), index(y, subsets[key]))
            if base_entro - entro_sum > max_gain:
                max_gain = base_entro - entro_sum
                best_attr = attrID
                best_threshold = threshold

        return best_attr, best_threshold

    def fit(self, X, y):
        self.train_decision_tree(X, y, list(range(0, len(X[0]))))

    def train_decision_tree(self, X, y, attributes):
        # 如果所有样本都属于同一类别C，则将当前节点标记为C类叶节点
        if len(set(y)) == 1:
            self.result = y[0]
            return

        # 判断样本在属性集上取值是否相同
        is_same = 1
        for attrID in attributes:
            column = [row[attrID] for row in X]
            if len(set(column)) != 1:
                is_same = 0

        # 如果属性集为空，或者样本在属性集上取值相同，则将类别
        # 设定为该节点所含样本最多的类别
        if not len(y) or is_same:
            self.result = Counter(y).most_common()[0][0]
            return

        best_attr, best_threshold = self.evaluate_func(X, y, attributes)
        self.attribute = best_attr
        self.threshold = best_threshold
        subsets, _ = self.devide_by_attribute(X, y, best_attr)

        # 在父节点上使用了连续属性并不会禁用子节点继续使用该属性
        subattributes = attributes.copy()
        if not best_threshold:
            subattributes.remove(best_attr)

        # 递归地训练子节点
        for key in subsets:
            subX = [X[i] for i in subsets[key]]
            suby = [y[i] for i in subsets[key]]
            new_child = DecisionTree()
            new_child.train_decision_tree(subX, suby, subattributes)
            self.children[key] = new_child

    def predict(self, x):
        # 如果当前节点为叶节点
        if self.result:
            return self.result

        # 如果当前节点的分割属性为连续属性
        if self.threshold:
            if x[self.attribute] <= self.threshold:
                return self.children['smaller'].predict(x)
            else:
                return self.children['larger'].predict(x)

        return self.children[x[self.attribute]].predict(x)

if __name__ == "__main__":
    decision_tree = DecisionTree()
    dataset = [
        ['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.697, 0.460, '好瓜'],
        ['乌黑', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', 0.774, 0.376, '好瓜'],
        ['乌黑', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.634, 0.264, '好瓜'],
        ['青绿', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', 0.608, 0.318, '好瓜'],
        ['浅白', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.556, 0.215, '好瓜'],
        ['青绿', '稍蜷', '浊响', '清晰', '稍凹', '软粘', 0.403, 0.237, '好瓜'],
        ['乌黑', '稍蜷', '浊响', '稍糊', '稍凹', '软粘', 0.481, 0.149, '好瓜'],
        ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '硬滑', 0.437, 0.211, '好瓜'],
        ['乌黑', '稍蜷', '沉闷', '稍糊', '稍凹', '硬滑', 0.666, 0.091, '坏瓜'],
        ['青绿', '硬挺', '清脆', '清晰', '平坦', '软粘', 0.243, 0.267, '坏瓜'],
        ['浅白', '硬挺', '清脆', '模糊', '平坦', '硬滑', 0.245, 0.057, '坏瓜'],
        ['浅白', '蜷缩', '浊响', '模糊', '平坦', '软粘', 0.343, 0.099, '坏瓜'],
        ['青绿', '稍蜷', '浊响', '稍糊', '凹陷', '硬滑', 0.639, 0.161, '坏瓜'],
        ['浅白', '稍蜷', '沉闷', '稍糊', '凹陷', '硬滑', 0.657, 0.198, '坏瓜'],
        ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '软粘', 0.360, 0.370, '坏瓜'],
        ['浅白', '蜷缩', '浊响', '模糊', '平坦', '硬滑', 0.593, 0.042, '坏瓜'],
        ['青绿', '蜷缩', '沉闷', '稍糊', '稍凹', '硬滑', 0.719, 0.103, '坏瓜']
    ]
    X = [row[:8] for row in dataset]
    y = [row[8] for row in dataset]
    decision_tree.fit(X, y)
    print(decision_tree.predict(['青绿', '稍蜷', '浊响', '清晰', '凹陷', '硬滑', 0.3, 0.264]))
