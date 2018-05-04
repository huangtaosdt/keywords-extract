# -*- coding: UTF-8 -*-
import jieba,time
import sys
reload(sys)
sys.setdefaultencoding('utf8')

start = time.clock()


fout=open("data/corpus","w")
words_file=open("data/sum_nanzhi_ht",'r')
a=1
def get_words_dict():
    print a
    words_list = {}
    for row in words_file:

        row=row.split('\t')
        value=row[0].replace("<","")
        value=value.replace(">","")
        tmp={}.fromkeys(row[1].split(" "),value)
        words_list.update(tmp)
    return words_list

del words_file

with open("data/tmp_nanzhi_ht_sum") as query:
    words_dict=get_words_dict()
    for line in query:
        row=line.split("\t")
        # row[0] = ' '.join(list(jieba.cut(row[0])))
        # row[0]=' '.join(list(jieba.cut(row[0])))
        # with open("data/sum_nanzhi_ht") as words:
        # for line in words:
        #     norm_key = line.split('\t')[0]
        #     norm_value = line.split('\t')[1].split(" ")
        #     for value in norm_value:
        #         if value.strip() in row[0]:
        #             row[0]=row[0].replace(value,norm_key)
        tmp=row[0].split(" ")
        for indx,word in enumerate(tmp):
            if words_dict.has_key(word):
                tmp[indx]=word
        print "raw line:",row[0]
        print "new line:"," ".join(tmp)
        fout.write('\t'.join(row).rstrip()+'\n')


end = time.clock()
print 'runtime:',end-start
