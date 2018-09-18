import numpy as np
import math

"""朴素贝叶斯分类算法"""
class Bayes:

    """朴素贝叶斯分类器的训练函数"""
    # train_data：训练数据
    # train_category：训练数据的分类结果
    # data_categories: 所有的分类的种类列表
    def train_nbm(self, train_data, train_category, data_categories):
        # 训练数据的组数
        num_train_data = len(train_data)
        # 训练数据的特征数量
        num_feature = len(train_data[0])
        # 类别种类的数量
        num_data_categories = len(data_categories)
        # 初始化每个数据集的类别出现的概率
        p_train_category = np.zeros(num_data_categories)
        # 计算每个类别在训练数据集中出现的概率
        for category in data_categories:
            p_train_category[data_categories.index(category)] = train_category.count(category)\
                                                                / float(num_train_data)
        # 初始化类别条件下的各个特征出现的次数 (防止出现0，初始化为单位矩阵)
        num_train_category_feature = np.ones((num_data_categories, num_feature))
        # 初始化类别条件下的各个特征出现的概率
        p_train_category_feature = np.zeros((num_data_categories, num_feature))
        # 初始化每个类别的所有特征在数据集中出现的次数 (防止出现0，初始化为单位矩阵)
        num_every_category = 2 * np.ones(num_data_categories)
        # 按行遍历数据集
        for i in range(num_train_data):
            # 寻找该条数据的类别，将该条数据的特征值(出现为1，不出现为0)加入到p_train_category_feature对应的类别行
            num_train_category_feature[train_category[i]] += train_data[i]
            num_every_category[train_category[i]] += sum(train_data[i])

        # 计算每个类别中所有特征出现的条件概率 （每个类别中的特征出现的次数 / 该类别中的特征出现的总次数）
        # 为了防止下溢 使用log函数
        for i in range(num_train_category_feature):
            p_train_category_feature[i] = math.log(num_train_category_feature[i] / num_every_category[i])

        return p_train_category_feature,p_train_category

    """朴素贝叶斯的分类函数"""
    # need_classify_data: 要进行分类的数据
    # p_train_category_feature: 类别条件下的各个特征出现的概率
    # p_train_category: 数据集的类别出现的概率
    def classify_nbm(self, need_classify_data, p_train_category_feature, p_train_category):
        #初始化分类结果下标列表
        classify_result_index = np.zeros(len(need_classify_data))
        # 循环计算在分类数据提供的特征值下，对应的每个类别的概率
        for i in range(len(need_classify_data)):
            # 初始化一个列表，临时保存该条数据是对应的每个类别的概率
            temp = np.zeros(len(p_train_category))
            for j in range(len(p_train_category)):
                temp[j] = sum(need_classify_data[i] * p_train_category_feature[j]) + math.log(p_train_category[j])
            # 挑选概率最大的类别作为分类结果
            classify_result_index[i] = p_train_category.index(max(temp))

        return classify_result_index