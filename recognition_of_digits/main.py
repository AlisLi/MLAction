import numpy as np
from recognition_of_digits import kNN,metrics,preprocessing
from recognition_of_digits import data_preprocessing as dp

class RecognitionOfDigits:

    def __init__(self,train_file,test_file):
        self.train_file = open(train_file)
        self.test_file = open(test_file)
        self.train_X = None
        self.train_Y = None
        self.deal_train_X = None
        self.test_X = None
        self.test_Y = None
        self.deal_test_X = None

    def _data_processing(self,filename):
        # 按行读取文件
        lines = filename.readlines()
        x = np.zeros((len(lines),64),dtype=np.int)
        y = np.zeros(len(lines),dtype=np.int)
        i = 0
        for line in lines:
            temp = line.split(",")
            for j in range(64):
                x[i][j] = temp[j]
            y[i] = temp[-1]
            i += 1

        return x,y

    def predict(self):
        # 获得 训练数据集和测试数据集
        self.train_X,self.train_Y = self._data_processing(self.train_file)
        self.test_X,self.test_Y = self._data_processing(self.test_file)
        np.set_printoptions(threshold=np.inf)
        print(self.train_Y)
        np.set_printoptions(threshold=np.inf)
        print(self.train_X[:5])
        ss = preprocessing.StandardScaler()
        preprocessing.StandardScaler.fit(ss,self.train_X)
        preprocessing.StandardScaler.fit(ss,self.test_X)
        self.deal_train_X = preprocessing.StandardScaler.transform(ss,self.train_X)
        self.deal_test_X = preprocessing.StandardScaler.transform(ss,self.test_X)

        kcf = kNN.kNNClassifier(5)
        kNN.kNNClassifier.fit(kcf,self.deal_train_X,self.train_Y)

        predict_y = kNN.kNNClassifier.predict(kcf,self.deal_test_X)
        score = kNN.kNNClassifier.score(kcf,self.deal_test_X,self.test_Y)

        return predict_y,score



