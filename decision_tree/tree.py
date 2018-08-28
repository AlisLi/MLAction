

"""使用决策树的分类函数"""
# input_tree: 决策树
# feat_labels: 测试数据集的标签列表
# test_vec: 所要分类的数据（测试数据：只能是一条数据）
def classify(input_tree, feat_labels, test_vec):
    first_str = list(input_tree.keys())[0]
    second_dict = input_tree[first_str]
    # 获取特征为first_str在feat_labels中的下标
    feat_index = feat_labels.index(first_str)
    for key in second_dict.keys():
        # 根据下标获得对应的特征值，和决策树中的特征值作比较
        if test_vec[feat_index] == key:
            # 找到，则继续递归向子节点寻找，直到找到分类
            if type(second_dict[key]) == dict:
                # 子节点为字典，递归
                class_label = classify(second_dict[key], feat_labels, test_vec)
            else:
                #子节点不是字典，分类结束，返回结果
                class_label = second_dict[key]

    return class_label