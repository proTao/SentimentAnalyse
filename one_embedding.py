# -*- coding:utf-8 -*-

'''
one embedding测试
在GTX960上，36s一轮
经过30轮迭代，训练集准确率为95.95%，测试集准确率为89.55%
Dropout不能用太多，否则信息损失太严重
'''



dir = "/home/tao/tao/nlp_project/SogouC.reduced/Reduced"
sub_dir_list = os.listdir(dir)
for d in dir_list:
    num=10
    file_list = os.listdir(os.path.join(dir,d,str(num)))
    for file_pwd in file_list:
        with open(file_pwd) as f:




maxlen = 200 #截断字数
min_count = 20 #出现次数少于该值的字扔掉。这是最简单的降维方法

content = ''.join(all_[0])
abc = pd.Series(list(content)).value_counts()
abc = abc[abc >= min_count]
abc[:] = range(1, len(abc)+1)
abc[''] = 0 #添加空字符串用来补全
word_set = set(abc.index)

def doc2num(s, maxlen): 
    s = [i for i in s if i in word_set]
    s = s[:maxlen] + ['']*max(0, maxlen-len(s))
    return list(abc[s])

all_['doc2num'] = all_[0].apply(lambda s: doc2num(s, maxlen))

#手动打乱数据
idx = range(len(all_))
np.random.shuffle(idx)
all_ = all_.loc[idx]

#按keras的输入要求来生成数据
x = np.array(list(all_['doc2num']))
y = np.array(list(all_['label']))
y = y.reshape((-1,1)) #调整标签形状


from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Embedding
from keras.layers import LSTM

#建立模型
model = Sequential()
model.add(Embedding(len(abc), 256, input_length=maxlen))
model.add(LSTM(128)) 
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

batch_size = 128
train_num = 15000

model.fit(x[:train_num], y[:train_num], batch_size = batch_size, nb_epoch=30)

model.evaluate(x[train_num:], y[train_num:], batch_size = batch_size)

def predict_one(s): #单个句子的预测函数
    s = np.array(doc2num(s, maxlen))
    s = s.reshape((1, s.shape[0]))
    return model.predict_classes(s, verbose=0)[0][0]