import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
import logging
import time
import codecs
import re
import random
import string
from tqdm import tqdm
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, StratifiedKFold
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
import tensorflow as tf
import keras
from keras.models import Model
from keras.layers import Input, Dense, Lambda
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from keras.utils import to_categorical
tf.logging.set_verbosity(tf.logging.ERROR)

import re
from urllib.parse import unquote
output_path = "../output/"
output_filename = "bert_wwm_output.pkl"

# included as of Unicode 8.0
ranges = [
    {"from": ord(u"\u3300"), "to": ord(u"\u33ff")},  # compatibility ideographs
    {"from": ord(u"\ufe30"), "to": ord(u"\ufe4f")},  # compatibility ideographs
    {"from": ord(u"\uf900"), "to": ord(u"\ufaff")},  # compatibility ideographs
    {"from": ord(u"\U0002F800"), "to": ord(u"\U0002fa1f")},  # compatibility ideographs
    {'from': ord(u'\u3040'), 'to': ord(u'\u309f')},  # Japanese Hiragana
    {"from": ord(u"\u30a0"), "to": ord(u"\u30ff")},  # Japanese Katakana
    {"from": ord(u"\u2e80"), "to": ord(u"\u2eff")},  # cjk radicals supplement
    {"from": ord(u"\u4e00"), "to": ord(u"\u9fff")},
    {"from": ord(u"\u3400"), "to": ord(u"\u4dbf")},
    {"from": ord(u"\U00020000"), "to": ord(u"\U0002a6df")},
    {"from": ord(u"\U0002a700"), "to": ord(u"\U0002b73f")},
    {"from": ord(u"\U0002b740"), "to": ord(u"\U0002b81f")},
    {"from": ord(u"\U0002b820"), "to": ord(u"\U0002ceaf")}
]


def is_cjk(char):
    return any([range["from"] <= ord(char) <= range["to"] for range in ranges])


def remove_url(x, url_prefix):
    if 'www.okex' in x:
        return x.replace('www.okex','okex')
    prefix_list = [0] * len(x)
    turn = 0
    for i in range(len(x)):
        if i < len(x) - len(url_prefix) and x[i:i + len(url_prefix)] == url_prefix:
            turn = 1
        if is_cjk(x[i]):
            turn = 0
        prefix_list[i] = turn
    s = ''
    for i in range(len(x)):
        if prefix_list[i] == 0:
            s += x[i]
    return s


def text_clean(x):
    '''
    文本数据清洗:

    '''
    x = unquote(x, 'utf-8')
    # 原文不要做大规模修改，不要统一大小写，线上测试数据大小写敏感
    # 这里主要清楚一下无关紧要的东西
    # x = x.lower()
    x = re.sub(r'\?{2,}', '', x) # 在做entity任务时不要去掉？，有entity包含？， 标注数据问题 ????宜湃网 ????信和大金融 等等
    x = re.sub(r'\.{2,}', '', x)
    x = re.sub(r'\-{2,}', '--', x)
    x = re.sub(r'\s{2,}', ' ', x)
    x = re.sub(r"\{IMG\:\d{1,}\}", '', x)
    x = remove_url(x, 'http')
    x = remove_url(x, 'www')
    x = re.sub('<[^>]*>', '', x)
    marks = ['&nbsp;', '&quot;', '&gt;', '&ldquo;', '&rdquo;', '&middot;', '|']
    for i in marks:
        x = x.replace(i, '')
    return x

train_data = '../input/Train_Data.csv'
test_data = '../input/Test_Data.csv'
submit_temp = '../input/Submit_Example.csv'

df_train = pd.read_csv(train_data)
df_test = pd.read_csv(test_data)
df_submit = pd.read_csv(submit_temp)

df_train['title'] = df_train['title'].fillna('').astype(str)
df_train['text'] = df_train['text'].fillna('').astype(str)
df_test['title'] = df_test['title'].fillna('').astype(str)
df_test['text'] = df_test['text'].fillna('').astype(str)

df_train['title_clean'] = df_train['title'].apply(text_clean)
df_train['text_clean'] = df_train['text'].apply(text_clean)
df_test['title_clean'] = df_test['title'].apply(text_clean)
df_test['text_clean'] = df_test['text'].apply(text_clean)

df_train['title_text'] = df_train['title'] + df_train['text']
df_test['title_text'] = df_test['title'] + df_test['text']

df_train['title_text_clean'] = df_train['title_clean'] + df_train['text_clean']
df_test['title_text_clean'] = df_test['title_clean'] + df_test['text_clean']

df_train = df_train[~df_train['entity'].isnull()]
# 预测数据的时候这里要注意一下, 剔除了部分数据
df_test = df_test[~df_test['entity'].isnull()]

df_train['key_entity'] = df_train['key_entity'].fillna('').astype(str)

log_path = '../log/'
logger = logging.getLogger('bert_wwm_logger')
logger.setLevel(logging.DEBUG)

experiment_desc = 'text_entity_binary'

timestamp = time.strftime("%Y.%m.%d_%H.%M.%S", time.localtime())
fh = logging.FileHandler(log_path+'bert_log_'+experiment_desc+'_'+ timestamp+'.txt')
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

formatter = logging.Formatter('[%(asctime)s][%(levelname)s] ## %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)

logger.addHandler(fh)
logger.addHandler(ch)


learning_rate = 5e-5
min_learning_rate = 1e-5

roberta_path = '../input/bert_wwm/'
data_path = '../input/'
outdir = '../output/'

config_path = roberta_path + "bert_config.json"
checkpoint_path = roberta_path
dict_path = roberta_path + "vocab.txt"

BATCH_SIZE = 6
MAXLEN=300
DROPOUT_RATE=0.3
INIT_LEARNING_RATE=5e-5
THRESHOLD=0.5

class DataGenerator():
    def __init__(self, data, batch_size=BATCH_SIZE):
        self.data = data
        self.batch_size = batch_size
        self.steps = len(self.data[0]) // self.batch_size
        self.tokenizer = tokenizer
        if len(self.data[0]) % self.batch_size != 0:
            self.steps += 1
        
    def __len__(self):
        return self.steps
    
    def __iter__(self):
        while True:
            X1, X2, y, index = self.data
            idxs = list(range(len(self.data[0])))
            np.random.shuffle(idxs)
            T, T_, Y, INDEX = [], [], [], []
            for _, i in enumerate(idxs):
                txt1 = X1[i]
                txt2 = X2[i]
                t, t_ = tokenizer.encode(first=txt1, second=txt2, max_len=MAXLEN)
                T.append(t)
                T_.append(t_)
                Y.append(y[i])
                INDEX.append(index[i])
                if len(T) == self.batch_size or i == idxs[-1]:
                    T = np.array(T)
                    T_ = np.array(T_)
                    Y = np.array(Y)
                    INDEX = np.array(INDEX)
                    yield [T, T_], Y, INDEX
                    T, T_, Y, INDEX = [], [], [], []
    
import keras.backend as K
from keras.callbacks import Callback

def get_model():
    bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=None)
    # set bert model fix or not
    for layer in bert_model.layers:
        layer.trainable = True

    x1_in = Input(shape=(None,))
    x2_in = Input(shape=(None,))

    x = bert_model([x1_in, x2_in])
    x = Lambda(lambda x: x[:, 0])(x) # 取出[CLS]对应的向量用来做分类
    x = keras.layers.Dropout(rate=DROPOUT_RATE)(x)
    p = Dense(1, activation='sigmoid')(x)

    model = Model([x1_in, x2_in], p)
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(INIT_LEARNING_RATE), # 用足够小的学习率
        metrics=['accuracy']
    )
    model.summary()
    return model

# online evaluating score is defined by weighted F1 score
# but currently, we split the training data into text_entity combinations which boost training number for text, we need make some modifications
def weight_f1_score(val_y, prob, THRESHOLD, val_id):
    val_dict = {}
    for i in range(len(val_y)):
        if val_id[i] not in val_dict:
            val_dict[val_id[i]] = {'y':val_y[i], 's':0, 'e':{'tp':0, 'fp':0, 'fn':0}} 
        p_ = 0
        if prob[i] > THRESHOLD: 
            val_dict[val_id[i]]['s'] = 1
            if val_y[i] == 1:
                val_dict[val_id[i]]['e']['tp'] += 1
            else:
                val_dict[val_id[i]]['e']['fp'] += 1
        else:
            if val_y[i] == 1:
                val_dict[val_id[i]]['e']['fn'] += 1
    tps = 0
    fps = 0
    fns = 0
    for k, v in val_dict.items():
        if v['y'] == 1:
            if v['s'] == 1:
                tps += 1
            else:
                fns += 1
        else:
            if v['s'] == 1:
                fps += 1
    ps = tps/(tps+fps)
    rs = tps/(tps+fns)
    fs1 = 2*ps*rs/(ps+rs)
    tpe = 0
    fpe = 0
    fne = 0
    for k, v in val_dict.items():
        tpe += v['e']['tp']
        fpe += v['e']['fp']
        fne += v['e']['fn']
    pe = tpe/(tpe+fpe)
    re = tpe/(tpe+fne)
    fe1 = 2*pe*re/(pe+re)
    f1 = 0.4*fs1 + 0.6*fe1
    return f1,fs1,fe1


class Evaluate(Callback):
    def __init__(self, val_data, val_index, val_id):
        self.score = []
        self.best = 0.
        self.stop_epoch = 0
        self.patience=2
        self.early_stopping = 0
        self.val_data = val_data
        self.val_index = val_index
        self.val_id = val_id
        self.lr = 0
        self.passed = 0
        self.best_weights = None
        self.restore_best_weights=True
    
    def on_batch_begin(self, batch, logs=None):
        '''warmup on first epoch, turn lr to minimal from second epoch
        '''
        if self.passed < self.params['steps']:
            self.lr = (self.passed + 1.) / self.params['steps'] * learning_rate
            K.set_value(self.model.optimizer.lr, self.lr)
            self.passed += 1
        elif self.params['steps'] <= self.passed < self.params['steps'] * 2:
            self.lr = (2 - (self.passed + 1.) / self.params['steps']) * (learning_rate - min_learning_rate)
            self.lr += min_learning_rate
            K.set_value(self.model.optimizer.lr, self.lr)
            self.passed += 1
    
    def on_epoch_end(self, epoch, logs=None):
        f1, fs1, fe1 = self.evaluate()
        if f1 > self.best:
            self.best = f1
            self.early_stopping = 0
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
#                 model.save_weights('/output/bert_finetune.w')
        else:
            self.early_stopping += 1
            if self.early_stopping >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                if self.restore_best_weights:
                    self.model.set_weights(self.best_weights)
        logger.info('lr: %.6f, epoch: %d, f_1: %.4f, fs_1: %.4f, fe_1: %.4f, best f_1: %.4f\n' % \
                   (self.lr, epoch, f1, fs1, fe1, self.best))
    
    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))
            logger.info('Epoch %05d: early stopping' % (self.stopped_epoch + 1))
        
    def evaluate(self):
        prob = []
        val_x1, val_x2, val_y = self.val_data
        for i in tqdm(range(len(val_x1))):
            txt1 = val_x1[i]
            txt2 = val_x2[i]
            T1, T1_ = tokenizer.encode(first=txt1, second=txt2, max_len=MAXLEN)
            T1, T1_ = np.array([T1]), np.array([T1_])
            _prob = model.predict([T1, T1_])
            oof_train[self.val_index[i]] = _prob[0]
            prob.append(_prob[0])
        f1, fs1, fe1 = weight_f1_score(val_y, prob, THRESHOLD, self.val_id)
        return f1, fs1, fe1
            
def predict(data):
    prob = []
    val_x1, val_x2 = data
    for i in tqdm(range(len(val_x1))):
        txt1 = val_x1[i]
        txt2 = val_x2[i]
        T1, T1_ = tokenizer.encode(first=txt1, second=txt2, max_len=MAXLEN)
        T1, T1_ = np.array([T1]), np.array([T1_])
        _prob = model.predict([T1, T1_])
        prob.append(_prob[0])
    return prob

# 词汇表
token_dict = {}
with codecs.open(dict_path, "r", encoding="utf-8") as f:
    for line in f:
        token = line.strip()
        token_dict[token] = len(token_dict)

class OurTokenizer(Tokenizer):
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]')  # space类用未经训练的[unused1]表示
            else:
                R.append('[UNK]')  # 剩余的字符是[UNK]
        return R
    
tokenizer = OurTokenizer(token_dict)

# training data prepare
train_title_texts, train_entities, labels, train_ids = [], [], [], []
train_ids2count = {}
train_idx_data = []
for row in df_train.iloc[:].itertuples():
    idx = row.id
    train_ids2count[idx] = len(train_ids2count)
    title_text = row.title_text
    entities = [e.strip() for e in row.entity.split(';')]
    key_entities = [e.strip() for e in row.key_entity.split(';')]
    for e in entities:
        if e == '':continue 
        if e in key_entities:
            labels.append(1)
            train_idx_data.append([idx, title_text, e, '1'])
        else:
            labels.append(0)
            train_idx_data.append([idx, title_text, e, '0'])
        train_title_texts.append(title_text)
        train_entities.append(e)
        train_ids.append(train_ids2count[idx])
# testing data prepare
test_title_texts, test_entities, test_ids = [], [], []
test_idx_data = []
for row in df_test.iloc[:].itertuples():
    idx = row.id
    title_text = row.title_text
    entities = [e.strip() for e in row.entity.split(';')]
    for e in entities:
        if e == '':continue 
        test_title_texts.append(title_text)
        test_entities.append(e)
        test_ids.append(idx)
        test_idx_data.append([idx, title_text, e])
        

train_title_texts = np.array(train_title_texts)
train_entities = np.array(train_entities)
labels = np.array(labels)
train_ids = np.array(train_ids)
test_title_texts = np.array(test_title_texts)
test_entities = np.array(test_entities)
test_ids = np.array(test_ids)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2019)

oof_train = np.zeros((len(train_title_texts), 1), dtype=np.float32)
oof_test = np.zeros((len(test_title_texts), 1), dtype=np.float32)

for fold, (train_index, valid_index) in enumerate(skf.split(train_title_texts, labels)):
    logger.info('=========    fold {}    ========='.format(fold))
    x1 = train_title_texts[train_index]
    x2 = train_entities[train_index]
    y = labels[train_index]
    ids = train_ids[train_index]
    
    val_x1 = train_title_texts[valid_index]
    val_x2 = train_entities[valid_index]
    val_y = labels[valid_index]
    val_ids = train_ids[valid_index]
    
    train_D = DataGenerator([x1, x2, y, ids])
    evaluator = Evaluate([val_x1, val_x2, val_y], valid_index, val_ids)
    
    model = get_model()
    model.fit_generator(train_D.__iter__(),
                        steps_per_epoch=len(train_D),
                        epochs=10,
                        callbacks=[evaluator]
                       )
    oof_test += predict([test_title_texts, test_entities])
    K.clear_session()

oof_test /= 5

with open(output_path + output_filename, 'wb') as fp:
    pickle.dump(oof_test, fp)

with open(outdir+'train_cv_prob.txt', 'w') as fw:
    assert len(train_idx_data) == len(oof_train)
    for i in range(len(oof_train)):
        fw.write('\t'.join(train_idx_data[i]) + '\t' + str(oof_train[i]) + '\n')

with open(outdir+'test_prob.txt', 'w') as fw:
    assert len(test_idx_data) == len(oof_test)
    for i in range(len(oof_test)):
        fw.write('\t'.join(test_idx_data[i]) + '\t' + str(oof_test[i]) + '\n')           

cv_f1, cv_fs1, cv_fe1 = weight_f1_score(labels, oof_train, THRESHOLD, train_ids)
print('cv_f1: ', cv_f1)
print('cv_fs1: ',cv_fs1)
print('cv_fe1: ',cv_fe1)
logger.info('cv_f1: {}'.format(cv_f1))
logger.info('cv_fs1: {}'.format(cv_fs1))
logger.info('cv_fe1: {}'.format(cv_fe1))

test_pred_dict = {}
test_labels = [1 if x > THRESHOLD else 0 for x in oof_test]
for idx, data in enumerate(test_idx_data):
    sid = data[0]
    e = data[2]
    if sid not in test_pred_dict:
        test_pred_dict[sid] = {'neg': '', 'enti':[]}
    if test_labels[idx] == 1:
        test_pred_dict[sid]['neg'] = '1'
        test_pred_dict[sid]['enti'].append(e)
    
def merge_max_length(x_list):
    x_count = [0] * len(x_list)
    for i in range(len(x_list)):
        for x in x_list:
            if x_list[i] in x:
                x_count[i] += 1
    result = []
    for i in range(len(x_count)):
        if x_count[i] == 1:
            result.append(x_list[i])
    return result
            
with open(data_path+'Submit_Example.csv', 'r') as f, \
    open(outdir+'bert_wwm.csv' ,'w') as fw:
    fw.write('id,negative,key_entity\n')
    for line in f.readlines():
        if line.strip().endswith('label'): continue
        idx = line.strip().split(',')[0]
        if idx in test_pred_dict:
            neg = test_pred_dict[idx]['neg']
            enti = merge_max_length(test_pred_dict[idx]['enti'])
            if neg == '1':
                fw.write(idx + ',' + '1' + ',' + ';'.join(enti) + '\n')
            else:
                fw.write(idx + ',' + '0' + ',' + '' +'\n')
        else:
            fw.write(idx + ',' + '0' + ',' + '' +'\n')