# -*- coding: utf-8 -*-
import numpy as np

class KWExtracter(object):

    def __init__(self):
        self.vacabulary = []
        self.idf = 0
        self.tf = 0
        self.tfidf=0
        self.tdm = 0  # p(x|y)
        self.Pcates = {}  # p(y)
        self.labels = []  # 每个文本的分类
        self.doclength = 0  # 训练集文本数
        self.vacablen = 0  # 词典词长
        self.testset = 0  # 测试集

    def train_set(self, trainset):
        self.doclength = len(trainset)
        tempset = set()
        [tempset.add(word) for doc in trainset for word in doc]
        self.vacabulary = list(tempset)
        self.vacablen = len(self.vacabulary)
        self.calc_wordfreq(trainset)

    # 生成普通的词频向量
    def calc_wordfreq(self, trainset):
        self.idf=np.zeros([1,self.vacablen])
        self.tf=np.zeros([self.doclength,self.vacablen])
        for indx in range(self.doclength):
            for word in trainset[indx]:
                self.tf[indx][self.vacabulary.index(word)]+=1
            for sigleword in set(trainset[indx]):
                self.idf[0,self.vacabulary.index(sigleword)]+=1

    def map2vocab(self, testdata):
        self.testset=np.zeros([1,self.vacablen])
        for word in testdata:
            try:
                self.testset[0,self.vacabulary.index(word)]+=1
            except:
                print "Warning: unlisted word 未登录词：",word

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
        self.tfidf=np.multiply(self.tf,self.idf)

    def get_key_word(self,trainset,top):
        # if trainset!=None:
        print "*"*10,"\nKey word extracting...."
        self.train_set(trainset)
        self.cal_tfidf(trainset)
        sorted_indx = np.argsort(-self.tfidf)
        keyword_trainset = []
        for i, indxs in enumerate(sorted_indx):
            keyword_doc = []
            for indx in indxs:
                if self.tf[i][indx] != 0 and len(keyword_doc) < top:
                    keyword_doc.append(self.vacabulary[indx])
            keyword_trainset.append(keyword_doc)
        print "Done! \n"
        return keyword_trainset


if __name__ == '__main__':

    trainset=[['你','吃饭','了','吗'],
              ['你','喝水','了','吗'],
              ['你', '睡醒', '了', '吗'],
              ['我','玩']]
    extracter = KWExtracter()
    # extracter.train_set(trainset)
    # extracter.cal_tfidf(trainset)
    keyword_trainset=extracter.get_key_word(trainset,3)
    for doc in keyword_trainset:
        print " ".join(doc)