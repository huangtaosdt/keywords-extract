# -*- coding: UTF-8 -*-
from sklearn.cross_validation import train_test_split
from gensim.models.word2vec import Word2Vec
import numpy as np
import pandas as pd
import jieba
from sklearn.externals import joblib
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import sys, os

reload(sys)
sys.setdefaultencoding('utf8')
"""
| 载入数据、分词、切分训练、测试  （持久化）
    load_file_and_preprocessing
    get_data
| 构建词、句向量
    get_train_vecs
    build_sentence_vector
| 训练、预测
    svm_train
    get_predict_vecs
    svm_predict
"""


# ----------------------
# 载入数据、分词、切分训练、测试集
# def load_file_and_preprocessing():
#     neg = pd.read_excel('data/neg.xls', header=None, index=None)
#     pos = pd.read_excel('data/pos.xls', header=None, index=None)
#
#     cw = lambda x: list(jieba.cut(x))
#     pos['words'] = pos[0].apply(cw)
#     neg['words'] = neg[0].apply(cw)
#     y = np.concatenate((np.ones(len(pos)), np.zeros(len(neg))))
#     x_train, x_test, y_train, y_test = train_test_split(np.concatenate((pos['words'], neg['words'])), y, test_size=0.2)
#
#     save_path = os.getcwd() + '\svm_data'
#     if not os.path.exists(save_path):
#         os.mkdir(save_path)
#     np.save('svm_data/y_train.npy', y_train)
#     np.save('svm_data/y_test.npy', y_test)
#     return x_train, x_test

def load_file_and_preprocessing():
    # data=pd.read_table('data/res_query_labeled_all',header=None)
    data=pd.read_table('data/res_norm',header=None)
    data=data[~data[0].isnull()]
    cw=lambda x:list(jieba.cut(x))
    data['words']=data[0].apply(cw)
    y=list(data[3])
    x_train,x_test,y_train,y_test=train_test_split(data['words'],y,test_size=0.2)
    save_path = os.getcwd() + '\svm_data'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    np.save('svm_data/y_train.npy', y_train)
    np.save('svm_data/y_test.npy', y_test)
    return x_train, x_test


# 对每个句子的所有词向量取均值，来生成一个句子的vector
def build_sentence_vector(text, size, imdb_w2v):
    vec = np.zeros(size).reshape((1, size))
    # vec=np.zeros((1,size))
    count = 0
    for word in text:
        try:
            vec += imdb_w2v[word].reshape((1, size))
            count += 1
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec

def load_corpus():
    corpus=pd.read_table("data/tmp_nanzhi_ht_sum",header=None)
    print corpus.head()
# 生成词向量
def get_train_vecs(x_train, x_test):
    n_dim = 300
    imdb_w2v = Word2Vec(size=n_dim, min_count=10)
    imdb_w2v.build_vocab(x_train)

    imdb_w2v.train(x_train,total_examples=len(x_train),epochs=imdb_w2v.iter)
    train_vecs = np.concatenate([build_sentence_vector(z, n_dim, imdb_w2v) for z in x_train])
    np.save("svm_data/train_vecs.npy", train_vecs)

    imdb_w2v.train(x_test,total_examples=len(x_test),epochs=imdb_w2v.iter)
    test_vecs = np.concatenate([build_sentence_vector(z, n_dim, imdb_w2v) for z in x_test])
    np.save("svm_data/test_vecs.npy", test_vecs)

    imdb_w2v.save('svm_data/w2v_model.pkl')

def get_data():
    train_vecs=np.load('svm_data/train_vecs.npy')
    test_vecs=np.load('svm_data/test_vecs.npy')
    y_train=np.load('svm_data/y_train.npy')
    y_test=np.load('svm_data/y_test.npy')
    return train_vecs,y_train,test_vecs,y_test

# ---------------
# 训练svm模型
def svm_train(train_vecs,y_train,test_vecs,y_test):
    clf=SVC(kernel='rbf',verbose=True)
    clf.fit(train_vecs,y_train)
    joblib.dump(clf,'svm_data/svm_model.pkl')
    print clf.score(test_vecs,y_test)

# 构建待预测句子的向量
def get_predict_vecs(words):
    n_dim=300
    imdb_w2v=Word2Vec.load('svm_data/w2v_model.pkl')
    train_vecs=build_sentence_vector(words,n_dim,imdb_w2v)
    return train_vecs

def svm_predict(string):
    words=jieba.lcut(string)
    words_vecs=get_predict_vecs(words)
    clf=joblib.load('svm_data/svm_model.pkl')

    result=clf.predict(words_vecs)
    if int(result)==1:
        print string,' positive'
    else:
        print string,' negative'

def rf_train(train_vecs,y_train,test_vecs,y_test):
    rf=RandomForestClassifier(max_depth=8,random_state=0)
    rf.fit(train_vecs,y_train)
    joblib.dump(rf,'svm_data/rf_model.pkl')
    print rf.score(test_vecs,y_test)

def rf_predict(string):
    words=jieba.lcut(string)
    words_vecs=get_predict_vecs(words)
    rf=joblib.load('svm_data/rf_model.pkl')

    result=rf.predict(words_vecs)
    rr=rf.predict_proba(words_vecs)
    print string
    # print result.decode('utf-8')
    print result[0]
    print rr
    print ' '.join(list(rf.classes_)).decode('utf-8')
    # if int(float(result))==1:
    #     print string,' positive'
    # else:
    #     print string,' negative'


if __name__ == '__main__':
    # x_train, x_test = load_file_and_preprocessing()
    # # 训练词、句向量，持久化
    # get_train_vecs(x_train,x_test)
    # # 加载数据
    # train_vecs, y_train, test_vecs, y_test=get_data()
    # # 训练svm
    # # svm_train(train_vecs,y_train,test_vecs,y_test)
    # # string = '电池充完了电连手机都打不开.简直烂的要命.真是金玉其外,败絮其中!连5号电池都不如'
    # string="白癜风怎样引起的"
    # rf_train(train_vecs,y_train,test_vecs,y_test)
    # rf_predict(string)
    load_corpus()

    # svm_predict(string)

    # model=get_train_vecs(x_train,x_test)
    # print model["电池".decode('utf-8')]

    # print x_train.head(10)
    '''
    0.796019900498
    电池充完了电连手机都打不开.简直烂的要命.真是金玉其外,败絮其中!连5号电池都不如  negative
    ------
    randomforest
    0.799336650083
    电池充完了电连手机都打不开.简直烂的要命.真是金玉其外,败絮其中!连5号电池都不如  negative
    '''
