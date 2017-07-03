#-*- coding: UTF-8 -*-
import jieba
import logging
from gensim.models import word2vec
from gensim.corpora import Dictionary
import re 
import os

# ------------准备神经网络--------------
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Embedding
from keras.layers import LSTM
import numpy as np
import pandas as pd
from keras.preprocessing import sequence
from sklearn.cross_validation import train_test_split

# 保存读取模型
import h5py 
from keras.models import model_from_json  


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

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


def loadfile():
    neg=pd.read_excel('neg.xls',header=None,index=None)
    pos=pd.read_excel('pos.xls',header=None,index=None)
    cw = lambda x: list(jieba.cut(x))
    pos['words'] = pos[0].apply(cw)
    neg['words'] = neg[0].apply(cw)
    combined=np.concatenate((pos['words'], neg['words']))
    y = np.concatenate((np.ones(len(pos),dtype=int), np.zeros(len(neg),dtype=int)))
    return combined,y

def predict_one(s): #单个句子的预测函数
    s=list(jieba.cut(s))
    s=[s]
    s=parseSentences2Nums(s,word2index)
    s=sequence.pad_sequences(s, maxlen=maxlen)
    return network.predict_classes(s, verbose=0)[0][0]

def displayOneSample(i):
    print("".join(map(lambda w:index2word[w],x_train[i])))
    print(y_train[i])


# dic = Dictionary(sentences)
dic = Dictionary.load("small_dictionary")
model = word2vec.Word2Vec.load("word2vec.model")



# dic中所有的词的索引加1，把0的位置空出来
word2index = {v: k+1 for k, v in dic.items()}
index2word = {k+1:v for k, v in dic.items()}
index2word[0] = "*"
word2vec = {word: model[word] for word in word2index.keys()}




# 词向量维度 
wv_dim = model.layer1_size
# 词表数目
vocab_scale = len(dic)+1
batch_size = 32
n_epoch = 5
maxlen=140

embedding_weights = np.zeros((vocab_scale, wv_dim))#索引为0的词语，词向量全为0
for index, word in dic.items():#从索引为1的词语开始，对每个词语对应其词向量
    embedding_weights[index+1, :] = model.wv[word]

combined,y=loadfile()
dataset = parseSentences2Nums(combined, word2index)  
x_train, x_test, y_train, y_test = train_test_split(dataset, y, test_size=0.2)
x_train = sequence.pad_sequences(x_train, maxlen=maxlen) 
x_test = sequence.pad_sequences(x_test, maxlen=maxlen) 

network = Sequential()  # or Graph or whatever
network.add(Embedding(output_dim=wv_dim,
                    input_dim=vocab_scale,
                    mask_zero=True,
                    weights=[embedding_weights],
                    input_length=max_len))  # Adding Input Length
network.add(LSTM(output_dim=50, activation='sigmoid', inner_activation='hard_sigmoid'))
network.add(Dropout(0.5))
network.add(Dense(1))
network.add(Activation('sigmoid'))





json_string = model.to_json()  
open('my_model_architecture.json','w').write(json_string)  
model.save_weights('my_model_weights.h5')  
#读取model  
model = model_from_json(open('my_model_architecture.json').read())  
model.load_weights('my_model_weights.h5')  