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
        pass


    def build_tdm(self):
        pass

    def map2vocab(self, testdata):
        pass

    def predict(self, testset):
        pass

    def cal_tfidf(self, trainset):
        pass
