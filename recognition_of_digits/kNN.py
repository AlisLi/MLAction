import numpy as np
from math import sqrt
from collections import Counter
from .metrics import accuracy_score

# KNN算法

class kNNClassifier:

    """构造函数"""
    def __init__(self, k):
        """初始化kNN分类器"""
        assert k >= 1, "k must be valid"
        self.k = k
        self._X_train = None
        self._y_train = None

    def fit(self, X_train, y_train):
        """根据训练数据集X_trian和y_train训练kNN分类器"""
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must equal to the size of y_train"
        assert self.k <= X_train.shape[0], \
            "the size of X_train must be at least k"

        self._X_train = X_train
        self._y_train = y_train
        return self

    def predict(self, X_predict):
        """给定带预测数据集X_predict，返回表示X_predict的结果向量"""
        assert self._X_train is not None and self._y_train is not None,\
            "must fit before predict!"
        assert X_predict.shape[1] == self._X_train.shape[1],\
            "the feature number of X_predict must be equal to X_train"

        y_predict = [self._predict(x) for x in X_predict]

        return np.array(y_predict)

    def _predict(self, x):
        """给定单个带预测数据x,返回x的预测结果值"""
        assert x.shape[0] == self._X_train.shape[1],\
            "the feature number of x must be equal to X_train"

        distances = [sqrt(np.sum((x_train - x) ** 2)) for x_train in self._X_train]
        nearest = np.argsort(distances)

        # 选取距离所预测的特征向量最近的前k个点的类别
        topK_y = [self._y_train[i] for i in nearest[:self.k]]

        # 计算所对应类别的个数
        # 返回元组列表
        votes = Counter(topK_y)

        # 返回一个元组且取第一个元组的第一个参数（即：类别）
        return votes.most_common(1)[0][0]

    def score(self, X_test, y_test):
        """根据测试数据集 X_test 和 y_test 确定当前模型的准确度(不需要)"""
        y_predict = self.predict(X_test)
        return accuracy_score(y_test, y_predict)

    """输出"""
    def __repr__(self):
        return "kNN(k=%d)" % self.k













