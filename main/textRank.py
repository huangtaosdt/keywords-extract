#!/usr/bin/python
#coding:utf-8


'''
用TextRank算法抽取关键词
@author zhangjialin
微信公众号：FinTech修行僧
edit_time:2017.11.19
'''

import jieba
import chardet
import jieba.posseg as pseg
import time
import math
import copy

class textRank_keyword:

    def __init__(self):
        self.stopwords=[] #存放外部停用词
        self.doc=[] #存放分词后，剔除停用词后文档的关键词
        self.word_in={} #存放单词的连入节点，形式为{word:(word2,word3,word4,...)},这里用set集合是为了去除重复单词
        self.word_out={} #存放单词的连出节点,形式为{word:(word2,word3,word4,...])}，这里用set集合是为了去除重复单词
        self.word_weight={} #存放单词的权重，形式为{word1:1,word2:2,word3:5,...}

    #读取文档数据和外部停用词
    def ReadFile(self,filepath,stopwordspath):
        #读取用于提取关键词和短语的文档
        with open(filepath,'r') as f :
            for line in f:
                #print line
                #line = line.encode('utf-8')
                #code = chardet.detect(line)['encoding']
                #line = line.strip('\n').strip().decode( code,'ignore' ).encode('utf-8','ignore')
                line = line.strip('\n').strip()
                #print line
                #return
                #开启4个并行分词线程
                jieba.enable_parallel(4)
                #将分词后的结果暂时放入self.doc中
                #print "\t".join(jieba.cut_for_search(line,HMM=True)).encode('utf-8')
                self.doc+=jieba.cut_for_search(line,HMM=True) #进行分词前先去除首尾的空格，并采用HMM
            
            #print '\t'.join(self.doc).encode('utf-8')
        tmp_doc = copy.deepcopy(self.doc)
        self.doc = []
        for item in tmp_doc:
            self.doc.append(item.encode('utf-8'))
        #print '\t'.join(self.doc)
        #读取停用词词典
        with open(stopwordspath,'r') as f:
            for line in f:
                #code = chardet.detect(line)['encoding']
                #line = line.strip('\n').strip().decode( code,'ignore').encode('utf-8','ignore')
                line = line.strip('\n').strip()
                #print line
                #if(line == "会"):
                 #   print "dddddddddddd"
                #line = line.decode('utf-8').encode('utf-8')
                #line = line.encode('utf-8')
                #print line 
                #return
                #print '\t'.join(line.encode('utf-8')
                #self.stopwords.append(line.strip('\n').strip().replace('\r\n',''))#去除换行符和空格
                self.stopwords.append(line)#去除换行符和空格
            #print '\t'.join(self.stopwords)

    #对分词后的文档，先剔除停用词，再进行词性标注 ，只保留名词、动词、形容词
    def del_stopwords_and_pos(self):
        #剔除停用词
        #print(self.stopwords)
        #print '\t'.join(self.stopwords)
        l1=len(self.doc)
        s=[]
        for word in self.doc:
            #print word
            #if (word=="会"):
             #   print "eeeeeeeeeeeeeee" + word
            if word not in self.stopwords:
                #print word.encode('utf-8')
                #print word
                #利用列表的删除方法remove，删除指定值
                #self.doc.remove(word)
                s.append(word)
        self.doc=s
        l2=len(self.doc)
        print(l2,l1)
        #词性标注
        #先将self.doc的元素转为字符串的形式，便于词性标注
        self_doc_str=' '.join(self.doc)
        #词性标注,词性标注后返回的是一个二元组的迭代器
        pos_words=pseg.cut(self_doc_str)
        #print type(pos_words)
        #for key,value in pos_words:
            #print key.encode('utf-8'),value.encode('utf-8')
            #print '################'
        #用一个列表推导式，筛选出名词、动词、形容词
        #先清空self.doc,便于存放我们指定词性的关键词
        self.doc=[]
        #查询结巴词性标注类别后，我们只保留：形容词a ，名形词an，成语i，习用语I,名词n，人名nr，地名ns，机构团体nt，动词v，动名词vn
        self.doc=[word for word,pos in pos_words if pos  in [u'a',u'an','i','I','n','nr','ns','nt','v','vn']]
        #print len(self.doc)

    #初始化self.word_in , self.word_out ,self.word_weight便于构建关键词图
    def init_wordset(self):
        for word in self.doc:
            self.word_in[word]=set() #set()是为了去重
            self.word_out[word]=set()
            self.word_weight[word]=1 #单词初始权重赋值为1


    #构建关键词图,窗口window 大小为 k,默认为10
    def keywords_graph(self,window_k=10):
        #先调用初始化函数，初始化需要用的集合
        self.init_wordset()
        #单词word窗口内的k个单词,都计入其连出节点集合,同时窗口内的每个单词之间都存在无向无权的边，如果一个单词重复出现，则便可以获得其连入节点的集合
        #从头开始遍历单词,窗口下界为len(self.doc)-window_k-1
        window_lower=len(self.doc)-window_k
        for i in range(0,window_lower):
            # 将窗口内的单词添加进单词self.doc[i]的连出节点集合内
            for word in self.doc[i+1:i+window_k]:
                self.word_out[self.doc[i]].add(word)
            #将当前单词self.doc[i]添加进窗口内所有单词的连入节点集合
            for word in self.doc[i+1:i+window_k]:
                self.word_in[word].add(self.doc[i])

    #对关键词图进行迭代计算，设置最大迭代次数iter_num和迭代退出的判断条件exit_limit,以及阻尼系数d,默认为0.85
    def iter_calculate(self,iter_num=10000,exit_limit=0.001,d=0.85):
        i=0
        #循环迭代
        while i<iter_num:
            #获取迭代前各单词的权重值，便于后面比较迭代前后权重值差异的变化,
            # 不能直接赋值，如before_word_weight=self.word_weight，因为两者是会指向同一个字典，修改任何一个，另一个也会改变
            before_word_weight={}
            for word in self.word_weight:
                before_word_weight[word]=self.word_weight[word]

            #遍历单词集合,利用TextRank算法对各个单词权重进行更新
            for word in self.doc:
                #先判断单词Word的连入节点集合是否为空，若为空，则这部分节点初始值不变
                if self.word_in[word] == None:
                    continue
                # 累计单词Word的所有连入节点对其投票的权重和
                ws=0
                #遍历单词word的连入节点集合self.word_in
                for word_in in self.word_in[word]:
                    #判断连入节点的连出节点集合的元素个数是否为0
                    if len(self.word_out[word_in]) !=0:
                        ws+=self.word_weight[word_in]*1/len(self.word_out[word_in])
                    else: #连入节点的连出节点集合的元素个数若为0
                        ws+=self.word_weight[word_in]
                #更新单词word的权重
                self.word_weight[word]=(1-d)+d*ws
                #print(ws)

            #比较两次迭代前后的差异
            #这里，我们计算迭代前后单词权重对应的均方误差值，即mse=∑(before-after)^2
            #均方误差mse
            mse=0
            for word in self.doc:
                mse+=math.pow((before_word_weight[word]-self.word_weight[word]),2) #求前后权重值差的平方
                #可以输出迭代前后单词权重的值，比较一下
                #print(before_word_weight[word],self.word_weight[word])
            mse=math.sqrt(mse)
            #print(mse)
            #判断均方误差mse是否满足退出条件
            if mse<exit_limit:
                #print("mse:%d"%mse)
                break
            i+=1

    #抽取出前n个PR权重最大的关键词
    def extract_n_keyword(self,n):
        #对单词权重集合self.word_weight按值value排序,获取前n 个权重最大的关键词
        #存放按字典value排序的列表
        word_lst=sorted(self.word_weight,key=self.word_weight.__getitem__,reverse=True) #返回按value值倒排的key列表
        print("前%d个PR值最大的关键词为："%n)
        #print(word_lst[:n])
        print("\t".join(word_lst[:n]).encode('utf-8'))
        print('================================================')

        for word in word_lst[:n]:
            #code = chardet.detect(word)['encoding']
            #print word.decode('gbk').encode('utf-8')
            #word = word.decode( code,'ignore'   ).encode('utf-8','ignore')
            #print(word,u'：PR值为 ', self.word_weight[word])
            print word.encode("utf-8") + ' PR值为:' + str(self.word_weight[word]) + '\n'
            print('--------------------------------------')
            tmp_in = ''
            tmp_out = ''
            for item in self.word_in[word]:
                tmp_in = tmp_in + item.encode('utf-8') + '\t'
            print word.encode("utf-8") + ' 的连入单词集合为:' + tmp_in + '\n'
            for item in self.word_in[word]:
                tmp_in = tmp_in + item.encode('utf-8') + '\t'
            print word.encode("utf-8") + ' 的连出单词集合为:' + tmp_out + '\n'
            #print(word,'的连入单词集合为：', ' '.join(list(self.word_in[word])).encode('utf-8'))
            #print(word,'的连出单词集合为', ' '.join(list(self.word_out[word])).encode('utf-8'))


if __name__ =="__main__":
    start=time.time()
    textRank=textRank_keyword()
    textRank.ReadFile('data/2.txt','stopwords.txt')
    textRank.del_stopwords_and_pos()
    textRank.keywords_graph()
    textRank.iter_calculate()
    textRank.extract_n_keyword(20)
    end=time.time()
    print('程序共耗时 %d s'%(end-start))
