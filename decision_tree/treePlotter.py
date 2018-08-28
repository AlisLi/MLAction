import matplotlib.pyplot as plt
from decision_tree import tree

#解决中文显示问题
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False

"""使用文本注解绘制树节点"""

# 定义文本框和箭头格式
decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4",fc="0.8")
arrow_args = dict(arrowstyle="<-")

"""绘制带箭头的注解"""
def plot_node(nodeTxt, centerPt, parentPt, nodeType):
    axl.annotate(nodeTxt, xy=parentPt,\
                            xycoords='axes fraction',\
                            xytext=centerPt,textcoords='axes fraction',\
                            va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)

"""绘制示例图"""
# def create_plot():
#     global axl
#     fig = plt.figure(1, facecolor='white')
#     fig.clf()
#     axl = plt.subplot(111,frameon=False)
#     plot_node('决策节点',(0.5,0.1),(0.1,0.5),decisionNode)
#     plot_node('叶节点',(0.8,0.1),(0.3,0.8),leafNode)
#     plt.show()

"""获取叶结点的数目"""
def get_num_leafs(my_tree):
    # 初始化叶节点数目
    num_leafs = 0
    # 第一个子节点
    first_str = list(my_tree.keys())[0]
    second_dict = my_tree[first_str]
    for key in second_dict.keys():
        # 判断子节点类型
        if type(second_dict[key]) == dict:
            # 类型为字典，继续递归
            num_leafs += get_num_leafs(second_dict[key])
        else:
            # 否则为叶节点,也节点数量+1
            num_leafs += 1
    return num_leafs

"""计算树的层数"""
def get_tree_depth(my_tree):
    # 初始化最大深度
    max_depth = 0
    # 第一个子节点
    first_str = list(my_tree.keys())[0]
    second_dict = my_tree[first_str]
    for key in second_dict.keys():
        # 判断子节点类型
        """python3 需要这么判断"""
        if type(second_dict[key]) == dict:
            # 类型为字典，继续递归
            this_depth = 1 + get_tree_depth(second_dict[key])
        else:
            # 否则为叶节点,节点深度为1
            this_depth = 1
    if this_depth > max_depth:
        max_depth = this_depth

    return max_depth

"""存储树的信息"""
def retrieve_tree(i):
    list_of_tree = [
        {'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
        {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no',1: 'yes'}},1: 'no'}}}}
    ]

    return list_of_tree[i]

"""在父子节点间填充文本信息"""
def plot_mid_text(cntr_pt, parent_pt, txt_string):
    x_mid = (parent_pt[0] - cntr_pt[0]) / 2.0 + cntr_pt[0]
    y_mid = (parent_pt[1] - cntr_pt[1]) / 2.0 + cntr_pt[1]
    axl.text(x_mid, y_mid, txt_string)

"""绘制树"""
def plot_tree(my_tree, parent_pt, node_txt):
    global x_off,y_off,total_w,total_d

    # 获取叶子节点数目
    num_leafs = get_num_leafs(my_tree)
    # 获取树的深度
    depth = get_tree_depth(my_tree)
    # python3中，字典获取第一个key时需要转化为list
    first_str = list(my_tree.keys())[0]
    # 计算节点的起始坐标
    cntr_pt = (plot_tree.x_off + (1.0 + float(num_leafs)) / 2.0/plot_tree.total_w, plot_tree.y_off)
    # 在父子节点间加注释
    plot_mid_text(cntr_pt, parent_pt, node_txt)
    # 绘制节点
    plot_node(first_str, cntr_pt, parent_pt, decisionNode)
    # 获取节点的子节点
    second_dict = my_tree[first_str]
    # 减少y偏移量
    plot_tree.y_off = plot_tree.y_off - 1.0/plot_tree.total_d
    # 循环遍历子节点
    for key in second_dict.keys():
        # 判断子节点是否还有子节点
        if type(second_dict[key]) == dict:
            # 子节点仍有子节点,递归绘制
            plot_tree(second_dict[key], cntr_pt, str(key))
        else:
            # 子节点没有子节点,直接绘制
            plot_tree.x_off = plot_tree.x_off + 1.0/plot_tree.total_w
            plot_node(second_dict[key], (plot_tree.x_off, plot_tree.y_off), cntr_pt, leafNode)
            plot_mid_text((plot_tree.x_off, plot_tree.y_off), cntr_pt, str(key))
    # 返回上一节点的高度
    plot_tree.y_off = plot_tree.y_off + 1.0/plot_tree.total_d

"""绘制示例图"""
def create_plot(in_tree):
    global axl
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    axl = plt.subplot(111,frameon=False, **axprops)
    # 初始化total_w，total_d，x_off，y_off
    plot_tree.total_w = float(get_num_leafs(in_tree))
    plot_tree.total_d = float(get_tree_depth(in_tree))
    plot_tree.x_off = -0.5/plot_tree.total_w
    plot_tree.y_off = 1.0

    plot_tree(in_tree, (0.5, 1.0), '')
    plt.show()



def main():
    # create_plot()
    my_tree = retrieve_tree(0)
    labels = ['no surfacing', 'flippers']
    # print(get_num_leafs(my_tree))
    # print(get_tree_depth(my_tree))
    # create_plot(my_tree)
    print(tree.classify(my_tree, labels, [1, 1]))

if __name__ == '__main__':
    main()

