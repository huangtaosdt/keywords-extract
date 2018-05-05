# -*- coding: utf-8 -*-
import numpy as np


class KWExtracter(object):

    def __init__(self):
        self.vacabulary = []
        self.idf = 0
        self.tf = 0
        self.tfidf = 0
        self.tdm = 0  # p(x|y)
        self.Pcates = {}  # p(y)
        self.labels = []  # 每个文本的分类
        self.doclength = 0  # 训练集文本数
        self.vacablen = 0  # 词典词长
        self.testset = 0  # 测试集

    def train_set(self, trainset):
        self.doclength = len(trainset)
        tempset = set()
        stopwords=[]
        with open("stop_words") as fr:
            for word in fr:
                stopwords.append(word.strip())
        # for i in stopwords[:10]:
        #     print i

        [tempset.add(word) for doc in trainset for word in doc]
        # tempset=[word for word in tempset if word not in stopwords]
        self.vacabulary = list(tempset)
        self.vacablen = len(self.vacabulary)
        self.calc_wordfreq(trainset)

    # 生成普通的词频向量
    def calc_wordfreq(self, trainset):
        self.idf = np.zeros([1, self.vacablen])
        self.tf = np.zeros([self.doclength, self.vacablen])
        for indx in range(self.doclength):
            for word in trainset[indx]:
                try:
                    self.tf[indx][self.vacabulary.index(word)] += 1
                except:
                    pass
            for sigleword in set(trainset[indx]):
                try:
                    self.idf[0, self.vacabulary.index(sigleword)] += 1
                except:
                    pass

    def map2vocab(self, testdata):
        self.testset = np.zeros([1, self.vacablen])
        for word in testdata:
            try:
                self.testset[0, self.vacabulary.index(word)] += 1
            except:
                print "Warning: unlisted word 未登录词：", word

    def cal_tfidf(self, trainset):
        self.idf = np.zeros([1, self.vacablen])
        self.tf = np.zeros([self.doclength, self.vacablen])
        for indx in range(self.doclength):
            for word in trainset[indx]:
                try:
                    self.tf[indx, self.vacabulary.index(word)] += 1
                except:
                    pass
            self.tf[indx] = self.tf[indx] / float(len(trainset[indx]))
            for sigleword in set(trainset[indx]):
                try:
                    self.idf[0, self.vacabulary.index(sigleword)] += 1
                except:
                    pass
        self.idf = np.log(float(self.doclength) / self.idf)
        self.tfidf = np.multiply(self.tf, self.idf)

    def get_key_word(self, trainset, top):
        # if trainset!=None:
        print "*" * 10, "\nKey word extracting...."
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


def load_dataset(trainset):
    import jieba

    docs = []
    for doc in trainset:
        doc = jieba.cut(doc)
        docs.append(list(doc))

    return docs


def load_stopwords(file1, file2):
    stop_words = []
    fr1 = open(file1);
    fr2 = open(file2)
    raw = fr1.readlines();
    raw.extend(fr2.readlines())
    with open("stop_words", "w") as fout:
        for word in raw:
            if word.strip() not in stop_words:
                stop_words.append(word.strip())
                fout.write(word.strip() + '\n')
    return stop_words


if __name__ == '__main__':

    trainset = [['你', '吃饭', '了', '吗'],
                ['你', '喝水', '了', '吗'],
                ['你', '睡醒', '了', '吗'],
                ['我', '玩']]

    res=load_dataset(["生物课上，教授正在讲解精子构造。 当教授讲到精子的主要成分是蛋白质 和葡萄糖时，一个女生突然站起来提问：“那为什么一点都不甜呢？”全班顿时寂静，然而教授镇静地说：“因为感受甜味的味蕾在舌尖，不是在舌根。好好的一节生物课。。。被两个老司机彻底毁了",
                  "有个朋友在外住旅馆。门缝里塞进来一张小卡片。他捡起来打算扔进垃圾桶，突然愣住了……“怎么会……”他不可置信的看着卡片，即便印刷粗糙，但他绝不会认错，穿那套衣服还是他下的单——绝不会认错！他坐在床沿，沉默许久，终于下定决心掏出了手机。又突然想起什么，换了酒店的座机。“1……3……7……4……”每一个按键都那么沉重，他想不好该如何开口。电话终究还是接通了。话筒里传来一个慵懒的女声：“你好……”千言万语梗塞在咽喉，他涨红了脸，怒吼到:“你们他妈太没有职业道德了吧?盗我女装照片印卡片！我前天刚拍的啊！”"])
    for doc in res:
        print " ".join(doc)
    trainset=["生物课上，教授正在讲解精子构造。 当教授讲到精子的主要成分是蛋白质 和葡萄糖时，一个女生突然站起来提问：“那为什么一点都不甜呢？”全班顿时寂静，然而教授镇静地说：“因为感受甜味的味蕾在舌尖，不是在舌根。好好的一节生物课。。。被两个老司机彻底毁了",
                  "有个朋友在外住旅馆。门缝里塞进来一张小卡片。他捡起来打算扔进垃圾桶，突然愣住了……“怎么会……”他不可置信的看着卡片，即便印刷粗糙，但他绝不会认错，穿那套衣服还是他下的单——绝不会认错！他坐在床沿，沉默许久，终于下定决心掏出了手机。又突然想起什么，换了酒店的座机。“1……3……7……4……”每一个按键都那么沉重，他想不好该如何开口。电话终究还是接通了。话筒里传来一个慵懒的女声：“你好……”千言万语梗塞在咽喉，他涨红了脸，怒吼到:“你们他妈太没有职业道德了吧?盗我女装照片印卡片！我前天刚拍的啊！”"]
    extracter = KWExtracter()
    # extracter.train_set(trainset)
    # extracter.cal_tfidf(trainset)
    keyword_trainset = extracter.get_key_word(trainset, 6)
    for doc in keyword_trainset:
        # print doc
        # print " ".join(doc)
        for word in doc:
            print word