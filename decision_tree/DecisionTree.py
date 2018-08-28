from math import log
import operator

"""普通的决策树"""
class DecisionTree:

    """计算给定数据集的香农熵"""
    def _calc_shanno_ent(self,data_set):
        # 计算数据集的组数
        num_entries = len(data_set)
        # 初始化类别字典
        label_counts = {}
        for featVec in data_set:
            #获取当前组的类别
            current_label = featVec[-1]
            if current_label not in label_counts.keys():
                # 不存在该类别则初始化到字典中
                label_counts[current_label] = 0
            # 该类别数量+1
            label_counts[current_label] += 1
        #初始化香侬熵
        shanno_ent = 0.0
        # 计算香农熵
        for key in label_counts:
            # 计算该类别的概率
            prob = label_counts[key] / num_entries
            # 香农熵公式
            shanno_ent -= prob * log(prob,2)

        return shanno_ent

    """按照给定的特征划分数据集"""
    # data_set: 待划分的数据集
    # axis：划分数据集的特征
    # value：需要返回的特征值
    def _split_data_set(self, data_set, axis, value):
        # 为了不将原数据集修改，初始化一个数据集
        ret_data_set = []
        for featVec in data_set:
            # 如果特征值为value，则将数据放入临时列表
            if featVec[axis] == value:
                # 获取给定特征值前半段数据
                reduced_featVec = featVec[:axis]
                # 获取给定特征值后半段数据
                reduced_featVec.extend(featVec[axis + 1 :])
                # 将重新组合的一组数据加入临时列表
                ret_data_set.append(reduced_featVec)

        return ret_data_set

    """选择最好的数据集划分方式"""
    def _choose_best_feature_to_split(self,data_set):
        # 获取特征值数量
        number_features = len(data_set[0]) - 1
        # 基础熵
        base_entropy = self._calc_shanno_ent(data_set)
        # 初始化最好的信息增益，最好的特征
        best_info_gain = 0.0
        best_feature = -1
        # 循环遍历每一个特征
        for i in range(number_features):
            # 获得该特征中的所有的值
            feat_list = [example[i] for example in data_set]
            # 将特征值的重复值去掉
            unique_values = set(feat_list)
            # 初始化新的熵
            new_entropy = 0.0
            for value in unique_values:
                # 得到按该特征值分类后的数据集
                sub_data_set = self._split_data_set(data_set, i, value)
                # 计算特征值对应的出现概率
                prob = len(sub_data_set) / float(len(data_set))
                # 计算新的熵值
                new_entropy += prob * self._calc_shanno_ent(sub_data_set)
            info_gain = base_entropy - new_entropy
            # new_entropy越高，混合数据越多，划分效果越差，所以info_gain越小，越好
            if(info_gain > best_info_gain):
                best_info_gain = info_gain
                best_feature = i

        return best_feature

    """当出现所有特征已经处理，但类标签依然不是唯一的时，采用多数表决来定义该叶子节点"""
    def _majority_cnt(self, class_list):
        # 初始化数据字典
        class_count = {}
        for vote in class_list:
            if vote not in class_count.keys():
                class_count[vote] = 0
            class_count[vote] += 1
        # 将字典按value降序排列
        # 返回列表（[特征，计数]）
        # key：获取字典中的值（第二列）
        # reverse：true（降序），false（升序）
        sorted_class_count = sorted(class_count.items(),key=operator.itemgetter(1),reverse=True)

        # 返回计数最大的特征
        return sorted_class_count[0][0]

    """递归创建决策树"""
    def create_tree(self, data_set, labels):
        print("1")
        # 获取类别列表
        class_list = [example[-1] for example in data_set]
        # 数据集中类别完全相同则停止继续划分,返回类别
        if class_list.count(class_list[0]) == len(class_list):
            return class_list[0]
        # 遍历完所有的特征值，返回出现次数最多的类别
        if len(data_set[0]) == 1:
            return self._majority_cnt(class_list)
        # 获取最好的划分特征下标
        best_feat = self._choose_best_feature_to_split(data_set)
        # 获取最好的特征标签
        best_feat_label = labels[best_feat]
        # 初始化决策树
        my_tree = {best_feat_label:{}}
        # 删除已划分特征标签
        del(labels[best_feat])
        # 获取该特征的所有特征值
        feat_values = [example[best_feat] for example in data_set]
        # 去掉相同的特征值
        uniqueVals = set(feat_values)
        # 循环递归创建决策树
        for value in uniqueVals:
            # 复制特征标签
            sub_labels = labels[:]
            my_tree[best_feat_label][value] = self.create_tree\
                (self._split_data_set(data_set,best_feat,value),sub_labels)

        return my_tree


    """创建测试数据集"""
    def create_data_set(self):
        data_set = [
            [1, 1, 'yes'],
            [1, 1, 'yes'],
            [1, 0, 'no'],
            [0, 1, 'no'],
            [0, 1, 'no'],
        ]

        return data_set



def main():
    dt = DecisionTree()
    data_set = dt.create_data_set()
    labels = ['no surfacing', 'flippers']
    my_tree = dt.create_tree(data_set, labels)
    result = dt._split_data_set(data_set,0,1)
    shanno_ent = dt._calc_shanno_ent(data_set)
    lab_most = dt._majority_cnt([example[-1] for example in data_set])


    print(result)
    print(shanno_ent)
    print(lab_most)
    print(my_tree)

if __name__ == '__main__':
    main()

