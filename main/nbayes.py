# -*- coding: utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding('gbk')
import numpy as np

class NBayes(object):
    def __init__(self):
        self.vacabulary = []
        self.idf = 0
        self.tf = 0
        self.tdm = 0  # p(x|y)
        self.Pcates = {}  # p(y)
        self.labels = []  # 每个文本的分类
        self.doclength = 0  # 训练集文本数
        self.vacablen = 0  # 词典词长
        self.testset = 0  # 测试集

    def train_set(self, trainset, classVec):
        self.cate_prob(classVec)
        self.doclength = len(trainset)
        tempset = set()
        [tempset.add(word) for doc in trainset for word in doc]
        self.vacabulary = list(tempset)
        self.vacablen = len(self.vacabulary)
        self.calc_wordfreq(trainset)
        self.build_tdm()

    def cate_prob(self, classVec):  # 计算p(y)
        self.labels = classVec
        labeltemps = set(self.labels)
        for label in labeltemps:
            self.Pcates[label] = float(self.labels.count(label)) / len(self.labels)

    # 生成普通的词频向量
    def calc_wordfreq(self, trainset):
        self.idf=np.zeros([1,self.vacablen])
        self.tf=np.zeros([self.doclength,self.vacablen])
        for indx in range(self.doclength):
            for word in trainset[indx]:
                self.tf[indx][self.vacabulary.index(word)]+=1
            for sigleword in set(trainset[indx]):
                self.idf[0,self.vacabulary.index(sigleword)]+=1

    def build_tdm(self):
        self.tdm=np.zeros([len(self.Pcates),self.vacablen])
        sumlist=np.zeros([len(self.Pcates),1])
        for indx in range(self.doclength):
            self.tdm[self.labels[indx]]+= self.tf[indx]

            sumlist[self.labels[indx]]=np.sum(self.tdm[self.labels[indx]])
        self.tdm=self.tdm/sumlist

    def map2vocab(self, testdata):
        self.testset=np.zeros([1,self.vacablen])
        for word in testdata:
            # 应该加 try
            self.testset[0,self.vacabulary.index(word)]+=1

    def predict(self, testset):
        if np.shape(testset)[1]!=self.vacablen:
            print "输入错误"
            exit(0)
        predvalue=0
        predclass=""
        for tdm_vector,keyclass in zip(self.tdm,self.Pcates):
            temp=np.sum(testset*tdm_vector*self.Pcates[keyclass])
            print testset,tdm_vector,self.Pcates[keyclass]
            print temp
            print "*"*10
            if temp>predvalue:
                predvalue=temp
                predclass=keyclass
        return predclass


    def cal_tfidf(self, trainset):
        self.idf=np.zeros([1,self.vacablen])
        self.tf=np.zeros([self.doclength,self.vacablen])
        for indx in range(self.doclength):
            for word in trainset[indx]:
                self.tf[indx,self.vacabulary.index(word)]+=1
            self.tf[indx]=self.tf[indx]/float(len(trainset[indx]))
            for sigleword in set(trainset[indx]):
                self.idf[0,self.vacabulary.index(sigleword)]+=1
        self.idf=np.log(float(self.doclength)/self.idf)
        self.tf=np.multiply(self.tf,self.idf)


if __name__ == '__main__':
    trainset=[['你','吃饭','了','吗'],
              ['你','喝水','了','吗'],
              ['你', '睡醒', '了', '吗']]

    nb=NBayes()
    nb.train_set(trainset,[0,1,1])
    nb.cal_tfidf(trainset)
    print nb.idf
    print nb.tf
    print ' '.join(nb.vacabulary)
    nb.map2vocab(trainset[0])
    print nb.predict(nb.testset)
    # nb.map2vocab(trainset[1])
    # print nb.predict(nb.testset)
    # nb.map2vocab(trainset[2])
    # print nb.predict(nb.testset)






























