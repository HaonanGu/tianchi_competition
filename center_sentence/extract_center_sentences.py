# coding=utf-8

import sys
from imp import reload
reload(sys)
# sys.setdefaultencoding('utf-8')

import tensorflow as tf
import numpy as np
import jieba
import os
import copy
import gensim
import re
from preprocess import *

#############################
#天池竞赛重大合同关键句子抽取代码归总
#############################

tf.reset_default_graph()
num_units = 512
num_rnn_layer = 3
input_size = 50 #词向量维度
category_num = 2 #分类数目
batch_size = 30

keep_prob = tf.placeholder(tf.float32, [])
x = tf.placeholder(tf.float32, [None,  None, input_size]) #[None,time_step,input_size]
y_label = tf.placeholder(tf.float32, [None, category_num])

def load_w2v_model(path):
    word_vectors = gensim.models.KeyedVectors.load_word2vec_format(path, binary=False)
    return word_vectors

def Multi_LSTM(num_units):
    def cell(num_units):
        cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=num_units,reuse=tf.AUTO_REUSE)
        return tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)

    cells = tf.nn.rnn_cell.MultiRNNCell([cell(num_units) for _ in range(num_rnn_layer)])
    h0 = cells.zero_state(batch_size, dtype=tf.float32)

    output, hs = tf.nn.dynamic_rnn(cells, inputs=x, initial_state=h0)
    output = hs[-1].h
    w = tf.Variable(tf.truncated_normal([num_units, category_num], stddev=0.1), dtype=tf.float32)
    b = tf.Variable(tf.constant(0.1, shape=[category_num]), dtype=tf.float32)
    y_fc = tf.matmul(output, w) + b
    return y_fc

def use_model(test_txt_path,test_txt_center_sentences_path):
    if not os.path.isdir(test_txt_center_sentences_path):
        os.makedirs(test_txt_center_sentences_path)
    list_txt = os.listdir(test_txt_path)
    try:
        with open('/home/118_16/code/center_sentence/adyl.txt', 'r') as f:
            tmptxt = f.read()
            delete_list = tmptxt.split('\n')
    except:
        delete_list=[]

    y_fc=Multi_LSTM(num_units=num_units)
    y=tf.nn.softmax(y_fc)

    print('------------提取公告文档中心句----------')
    print('loading w2v...')
    w2v_model= load_w2v_model(path='/home/118_16/code/center_sentence/wordembedding/zhwiki_2017_03.sg_50d.word2vec')

    saver = tf.train.Saver(max_to_keep=50)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, tf.train.latest_checkpoint('/home/118_16/code/center_sentence/model_file_center_sentence/'))

        w2v_vocab = w2v_model.vocab.keys()
        w2v_vocab_dict=dict(zip(w2v_vocab, w2v_vocab))

        print('extracting, wait a moment...')
        for i in range(0,len(list_txt)):
            path = os.path.join(test_txt_path,list_txt[i])
            path_save = os.path.join(test_txt_center_sentences_path, list_txt[i])
            try:
                if os.path.isfile(path):
                    with open(path,'r') as f:
                        text=f.read().strip()
                        delete_list=delete_list+re.findall(r'本公司董事会.*重大遗漏承担责任|本公司董事会.*重大遗漏|本公司及董事会.*承担个别及连带责任', text)+\
                                    re.findall(r'本公司及董事会全体成员.*承担个别及连带责任|公司及董事会全体成员.*承担个别及连带责任', text)+\
                                    re.findall(r'本公司及董事会全体成员.*重大遗漏承担责任|本公司及董事会全体成员.*重大遗漏', text)
                        for n in delete_list: text=text.replace(n,'')
                        try:
                            text = normalize_money(copy.deepcopy(text))
                        except:
                            pass
                        text=text.replace('。\n','。<enter>').replace('\n','<enter>').replace('。','。<enter>')
                        sentence_list=text.split('<enter>')
                        while sentence_list.count(u''):
                            sentence_list.remove(u'')
                        save_sentence=[]
                        for txt in sentence_list:
                            if txt=='' or str_similarity(u'本公司及董事会全体成员保证公告内容的真实、准确和完整，没有虚假记载、误导性陈述或者重大遗漏'.encode('utf-8').decode('utf-8'),txt,0.7):
                                continue
                            whole_sentences_list = ' '.join(jieba.cut(txt, cut_all=False)).split()
                            s = [e for e in whole_sentences_list if e in w2v_vocab_dict]
                            if s==[] :
                                continue
                            y_pred=sess.run(y, feed_dict={x: np.array([list(w2v_model[s])] * 30), keep_prob: 1.0})[0,:]
                            y_pred = np.array(y_pred).flatten()
                            for m in range(len(y_pred)):
                                if y_pred[m]>0.5:
                                    y_pred[m] =1
                                else:
                                    y_pred[m] = 0
                            if list(y_pred) == [0, 1]:
                                continue
                            else:
                                save_sentence.append(txt)

                    with open(path_save,'w') as f:
                        for j in save_sentence:
                            f.write(j+'\n')
                        # print (list_txt[i],' saved')
            except:
                with open(path, 'r') as f:
                    text = f.read().strip()
                    delete_list = delete_list + re.findall(r'本公司董事会.*重大遗漏承担责任|本公司董事会.*重大遗漏|本公司及董事会.*承担个别及连带责任', text) + \
                                  re.findall(r'本公司及董事会全体成员.*承担个别及连带责任|公司及董事会全体成员.*承担个别及连带责任', text) + \
                                  re.findall(r'本公司及董事会全体成员.*重大遗漏承担责任|本公司及董事会全体成员.*重大遗漏', text)
                    for n in delete_list: text = text.replace(n, '')
                with open(path_save, 'w') as f:
                    f.write(text)

def main(input_path,output_path):
    use_model(test_txt_path=input_path, test_txt_center_sentences_path=output_path)
    print('sucessfully, center sentences in path: ',output_path)

if __name__ == "__main__":
    main(input_path='test_txt/',output_path='test_txt_save/')