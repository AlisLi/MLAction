import numpy as np
from decision_tree import DecisionTree as dt
from decision_tree import treePlotter as tp


class Lenses:

    # 数据预处理
    def main(self):
        # 读取文件内容
        fr = open('F:\\machine_learning_workspace\\MLAction\\date_set\\lenses\\lenses.txt')
        # 将数据保存到列表中
        lenses = [inst.strip().split(' ')[1:] for inst in fr.readlines()]
        # 初始化特征标签
        lenses_labels = ['age', 'prescript', 'astigmatic', 'tearRate']
        # 构造决策树
        d_tree = dt.DecisionTree()
        lenses_tree = d_tree.create_tree(lenses, lenses_labels)
        print(lenses)
        print(lenses_tree)
        tp.create_plot(lenses_tree)

def main():
    l = Lenses()
    l.main()

if __name__ == '__main__':
    main()
