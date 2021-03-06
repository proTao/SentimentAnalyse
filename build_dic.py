#-*- coding: UTF-8 -*-
import jieba
import logging
from gensim.models import word2vec
from gensim.corpora import Dictionary
import re 
import os


dic_name = "new_dic"
word2vec_model_name = "new_word2vec.model"

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class MySentences(object):
    def __init__(self,toolong=300):
        self.iterlist = []
        self.toolong=toolong
        # 将被移除的正则模式
        self.remove_pattern=[]
        self.remove_pattern.append(re.compile(u"www\..*\.com"))
        self.remove_pattern.append(re.compile(u"\(.{0,10}\)"))
        self.remove_pattern.append(re.compile(u"\[.{0,10}\]"))
        self.remove_pattern.append(re.compile(u"（.{0,10}）"))
        self.remove_pattern.append(re.compile(u"「.{0,10}」"))
        self.remove_pattern.append(re.compile(u"&nbsp;"))
        self.remove_pattern.append(re.compile(u"[●　 -“”]"))

        # 被移除的集合
        self.remove_log=set()
        
        # 将被用来分割句子的正则模式
        self.split_pattern = []
        self.split_pattern.append(re.compile(u"[?!。？！…「」\(\)（）:：]"))
        self.split_pattern.append(re.compile(u"\.　"))
        self.split_pattern.append(re.compile(u"． "))
        self.split_pattern.append(re.compile(u"\.\.\."))

        self.codingerr_log = set()
        self.long_sentences_log = set()

    def append(self, dirname):
        # 添加数组存放文件夹
        self.iterlist.append(dirname)

    def __iter__(self):
        # 遍历iterlist中的文件夹
        for dirname in self.iterlist:
            print("in "+dirname)
            # 遍历所有文件
            for filename in os.listdir(dirname):
                filepath = os.path.join(dirname,filename)
                if os.path.isfile(filepath):
                    lines = []
                    with open(filepath) as f:
                        lines = f.readlines()
                    try:       
                        lines = map(lambda line: line.decode("gbk"), lines)
                    except Exception as e:
                        self.codingerr_log.add(filepath)
                        continue

                    # 收集日志
                    for index in range(5):
                        removecontent=map(lambda line: self.remove_pattern[index].findall(line), lines)
                        for i in removecontent:
                            if(len(i)):
                                for j in i:
                                    if len(j)>5:
                                        if j[1:-1] not in self.remove_log:
                                            self.remove_log.add(j[1:-1])

                    # 剔除指定的正则模式
                    for pattern in self.remove_pattern:
                        lines = map(lambda line: pattern.sub("", line), lines)

                    # 替换指定的正则模式为换行符
                    for pattern in self.split_pattern:
                        lines = map(lambda line: pattern.sub("\n", line), lines)
                    
                    # 用splitlines()切分句子
                    lines = map(lambda line: line.splitlines(),lines)
                    flatten = lambda x: [y for l in x for y in flatten(l)] if type(x) is list else [x]
                    lines = flatten(lines)
                    

                    
                    # 剔除空行
                    lines = filter(lambda line:len(line.strip())>0, lines)

                    # 将lines列表中每个元素（句子字符串），转换为词列表
                    lines=map(lambda line: list(jieba.cut(line)),lines)

                    for line in lines:
                        # 剔除无效字符（不知道哪里来的）
                        line = filter(lambda word: word!=u"\x00", line)
                        if(len(line)>self.toolong):
                            self.long_sentences_log.add(filepath)
                        if(len(line)>1):
                            yield line

def countSentenceLength(dictionary,dataset,minlen=4,maxlen=300):
    length = {}
    long_sentence = []
    short_sentence = []
    for line in dataset:
        l=len(line)
        if l in length.keys():
            length[l] = length[l]+1
        else:
            length[l] = 1
            if l>=maxlen:
                long_sentence.append(line)
            if l<=minlen:
                short_sentence.append(line)

    for i in range(len(long_sentence)):
        long_sentence[i] = map(lambda w: dictionary[w],long_sentence[i])
    long_sentence = map(lambda sen: "".join(sen), long_sentence)

    for i in range(len(short_sentence)):
        short_sentence[i] = map(lambda w: dictionary[w],short_sentence[i])
    short_sentence = map(lambda sen: "".join(sen), short_sentence)
    return length,long_sentence,short_sentence


def parseSentences2Nums(sentences, word2index):
    '''
    sentences:已经拆分成词列表的句子列表
    dictionary: gensim.corpora.Dictionary对象通过id2token返回的dict
    word2index: dictionary的token2id生成的字典对象
    '''
    dataset = []
    for sentence in sentences:
        num_list = []
        for word in sentence:
            if word in word2index:
                num_list.append(word2index[word])
            else:
                num_list.append(0)
        dataset.append(num_list)
    return dataset

def showTop10(model,s):
    l = model.most_similar(s)
    for i in l:
        print(i[0])



# initialize
sentences = MySentences()
sentences.append("/home/tao/tao/nlp_project/yuliao/SogouC/C000008")
sentences.append("/home/tao/tao/nlp_project/yuliao/SogouC/C000010")
sentences.append("/home/tao/tao/nlp_project/yuliao/SogouC/C000013")
sentences.append("/home/tao/tao/nlp_project/yuliao/SogouC/C000014")
sentences.append("/home/tao/tao/nlp_project/yuliao/SogouC/C000016")
sentences.append("/home/tao/tao/nlp_project/yuliao/SogouC/C000020")
sentences.append("/home/tao/tao/nlp_project/yuliao/SogouC/C000022")
sentences.append("/home/tao/tao/nlp_project/yuliao/SogouC/C000023")
sentences.append("/home/tao/tao/nlp_project/yuliao/SogouC/C000024")
sentences.append("/home/tao/tao/nlp_project/yuliao/Ctrip/neg")
sentences.append("/home/tao/tao/nlp_project/yuliao/Ctrip/pos")
sentences.append("/home/tao/tao/nlp_project/yuliao/Dangdang/neg")
sentences.append("/home/tao/tao/nlp_project/yuliao/Dangdang/pos")
sentences.append("/home/tao/tao/nlp_project/yuliao/Jingdong/neg")
sentences.append("/home/tao/tao/nlp_project/yuliao/Jingdong/pos")
sentences.append("/home/tao/tao/nlp_project/yuliao/ChnSentiCorp/neg")
sentences.append("/home/tao/tao/nlp_project/yuliao/ChnSentiCorp/pos")
sentences.append("/home/tao/tao/nlp_project/yuliao/novel")

# 构建词典
logging.info("******************")
logging.info("   开始创建词典   ")
logging.info("******************")

dic = Dictionary(sentences)
# dic = Dictionary.load("dictionary")
# model = word2vec.Word2Vec.load("word2vec.model")

# 过滤掉字典中文档频率小于3，低于100%的词
# no_below这个参数是频率数
# no_above这个参数是频率所占总文档数比例
# keep_n这个参数是保留前多少个词，当这个数大于当前总词汇时参数无效
dic.filter_extremes(no_below=3,no_above=1,keep_n=len(dic)+1)
dic.compactify()
logging.info("**********************************")
logging.info("    词典构建完成，保存至"+dic_name+"  ")
logging.info("**********************************")

dic.save(new_dic)

# 训练词向量模型并保存
logging.info("*******************")
logging.info("开始训练词向量模型")
logging.info("*******************")
model = word2vec.Word2Vec(sentences, min_count=2, size=150)
model.save(word2vec_model_name)
logging.info("***************************************")
logging.info("词向量模型构建完成，保存至" + word2vec_model_name)
logging.info("***************************************")


dataset = parseSentences2Nums(sentences, word2index)
length={}
length, longsen, shortsen = countSentenceLength(dic, dataset)

