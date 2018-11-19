# coding=utf-8

import sys
from imp import reload
reload(sys)
sys.path.append('/home/118_16/code/html_txt/')

import tensorflow as tf
import numpy as np
import jieba
import os
import copy
import gensim
from tqdm import tqdm
import re
from preprocess import *
from html2txt import chans_one_html

#############################
#天池竞赛重大合同关键句子抽取代码归总
#############################

class Center_sentence:
    '''
    You should use cs=Center_sentence() to restore the multiLSTM model firstly.

    Then, use the mothod: cs.use_model_savetextfiles(inputpath,outputpath) to extract <xxx.txt>s which
    are saved in inputpath. The center_sentence of <xxx.txt>s are saved in path: outputpath.

    Or,you can use the method:cs.use_model_text(text), it's input is an unicode string('\n' divides different lines) ,
    and it's output is the center text.It can be used as an API.
    '''
    def __init__(self):
        print('------------提取公告文档中心句----------')
        self.num_units = 512
        self.num_rnn_layer = 3
        self.input_size = 50  # 词向量维度
        self.category_num = 2  # 分类数目
        self.batch_size = 30
        self.keep_prob = tf.placeholder(tf.float32, [])
        self.x = tf.placeholder(tf.float32, [None,  None, self.input_size]) #[None,time_step,input_size]
        self.y_label = tf.placeholder(tf.float32, [None, self.category_num])
        print('loading w2v...')
        self.w2v_model= self.load_w2v_model(path='/home/118_16/code/center_sentence/wordembedding/zhwiki_2017_03.sg_50d.word2vec')
        self.y_fc=self.multi_LSTM(num_units=self.num_units)
        self.y = tf.nn.softmax(self.y_fc)
        print('restoring model...')
        self.sess=self.restore_model()

    def restore_model(self):
        saver = tf.train.Saver(max_to_keep=50)
        sess=tf.Session()
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, tf.train.latest_checkpoint('/home/118_16/code/center_sentence/model_file_center_sentence/'))
        return sess

    def load_w2v_model(self,path):
        word_vectors = gensim.models.KeyedVectors.load_word2vec_format(path, binary=False)
        return word_vectors

    def multi_LSTM(self,num_units):
        def cell(num_units):
            cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=num_units,reuse=tf.AUTO_REUSE)
            return tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)

        cells = tf.nn.rnn_cell.MultiRNNCell([cell(num_units) for _ in range(self.num_rnn_layer)])
        h0 = cells.zero_state(self.batch_size, dtype=tf.float32)

        output, hs = tf.nn.dynamic_rnn(cells, inputs=self.x, initial_state=h0)
        output = hs[-1].h
        w = tf.Variable(tf.truncated_normal([num_units, self.category_num], stddev=0.1), dtype=tf.float32)
        b = tf.Variable(tf.constant(0.1, shape=[self.category_num]), dtype=tf.float32)
        y_fc = tf.matmul(output, w) + b
        return y_fc

    def use_model_savetextfiles(self,test_txt_path,test_txt_center_sentences_path):
        if not os.path.isdir(test_txt_center_sentences_path):
            os.makedirs(test_txt_center_sentences_path)
        list_txt = os.listdir(test_txt_path)
        try:
            with open('/home/118_16/code/center_sentence/adyl.txt', 'r') as f:
                tmptxt = f.read()
                delete_list = tmptxt.split('\n')
        except:
            delete_list=[]

        w2v_model= self.w2v_model
        sess=self.sess

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
                            y_pred=sess.run(self.y, feed_dict={self.x: np.array([list(w2v_model[s])] * 30), self.keep_prob: 1.0})[0,:]
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

    def use_model_savetext_from_html(self,html_path,test_txt_center_sentences_path):
        if not os.path.isdir(test_txt_center_sentences_path):
            os.makedirs(test_txt_center_sentences_path)
        list_html = os.listdir(html_path)
        try:
            with open('/home/118_16/code/center_sentence/adyl.txt', 'r') as f:
                tmptxt = f.read()
                delete_list = tmptxt.split('\n')
        except:
            delete_list=[]

        w2v_model= self.w2v_model
        sess=self.sess

        w2v_vocab = w2v_model.vocab.keys()
        w2v_vocab_dict=dict(zip(w2v_vocab, w2v_vocab))
        
        pbar = tqdm(total = len(list_html))
        print('extracting, wait a moment...')
        for i in range(0,len(list_html)):
            pbar.update(1)
            pbar.set_description(list_html[i])
            html_path_each=os.path.join(html_path, list_html[i])
            text = '\n'.join(chans_one_html(html_path_each))
            path_save = os.path.join(test_txt_center_sentences_path, list_html[i].replace('.html','.txt'))
            try:
                text=text.strip()
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
                    y_pred=sess.run(self.y, feed_dict={self.x: np.array([list(w2v_model[s])] * 30), self.keep_prob: 1.0})[0,:]
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
                    #print (list_html[i].replace('.html','.txt'),' saved',' num:',i)
            except:
                text = text.strip()
                delete_list = delete_list + re.findall(r'本公司董事会.*重大遗漏承担责任|本公司董事会.*重大遗漏|本公司及董事会.*承担个别及连带责任', text) + \
                              re.findall(r'本公司及董事会全体成员.*承担个别及连带责任|公司及董事会全体成员.*承担个别及连带责任', text) + \
                              re.findall(r'本公司及董事会全体成员.*重大遗漏承担责任|本公司及董事会全体成员.*重大遗漏', text)
                for n in delete_list: text = text.replace(n, '')
                with open(path_save, 'w') as f:
                    f.write(text)
                    print('center_sentence err ',list_html[i].replace('.html', '.txt'), ' saved',' num:',i)
        pbar.close()

    def use_model_text(self,input_text):
        try:
            with open('/home/118_16/code/center_sentence/adyl.txt', 'r') as f:
                tmptxt = f.read()
                delete_list = tmptxt.split('\n')
        except:
            delete_list=[]

        w2v_model= self.w2v_model
        sess=self.sess

        w2v_vocab = w2v_model.vocab.keys()
        w2v_vocab_dict=dict(zip(w2v_vocab, w2v_vocab))

        try:
            delete_list=delete_list+re.findall(r'本公司董事会.*重大遗漏承担责任|本公司董事会.*重大遗漏|本公司及董事会.*承担个别及连带责任', input_text)+\
                        re.findall(r'本公司及董事会全体成员.*承担个别及连带责任|公司及董事会全体成员.*承担个别及连带责任', input_text)+\
                        re.findall(r'本公司及董事会全体成员.*重大遗漏承担责任|本公司及董事会全体成员.*重大遗漏', input_text)
            for n in delete_list: input_text=input_text.replace(n,'')
            try:
                input_text = normalize_money(copy.deepcopy(input_text))
            except:
                pass
            input_text=input_text.replace('。\n','。<enter>').replace('\n','<enter>').replace('。','。<enter>')
            sentence_list=input_text.split('<enter>')
            while sentence_list.count(u''):
                sentence_list.remove(u'')
            center_sentence=[]
            for txt in sentence_list:
                if txt=='' or str_similarity(u'本公司及董事会全体成员保证公告内容的真实、准确和完整，没有虚假记载、误导性陈述或者重大遗漏'.encode('utf-8').decode('utf-8'),txt,0.7):
                    continue
                whole_sentences_list = ' '.join(jieba.cut(txt, cut_all=False)).split()
                s = [e for e in whole_sentences_list if e in w2v_vocab_dict]
                if s==[] :
                    continue
                y_pred=sess.run(self.y, feed_dict={self.x: np.array([list(w2v_model[s])] * 30), self.keep_prob: 1.0})[0,:]
                y_pred = np.array(y_pred).flatten()
                for m in range(len(y_pred)):
                    if y_pred[m]>0.5:
                        y_pred[m] =1
                    else:
                        y_pred[m] = 0
                if list(y_pred) == [0, 1]:
                    continue
                else:
                    center_sentence.append(txt)
            center_text='\n'.join(center_sentence)
        except:
            delete_list = delete_list + re.findall(r'本公司董事会.*重大遗漏承担责任|本公司董事会.*重大遗漏|本公司及董事会.*承担个别及连带责任', input_text) + \
                          re.findall(r'本公司及董事会全体成员.*承担个别及连带责任|公司及董事会全体成员.*承担个别及连带责任', input_text) + \
                          re.findall(r'本公司及董事会全体成员.*重大遗漏承担责任|本公司及董事会全体成员.*重大遗漏', input_text)
            for n in delete_list: input_text = input_text.replace(n, '')
            center_text=input_text

        return center_text

def html_center(input_path,output_path):
    cs=Center_sentence()

    cs.use_model_savetext_from_html( html_path=input_path,
                                 test_txt_center_sentences_path=output_path)

    # cs.use_model_savetextfiles(test_txt_path=input_path, test_txt_center_sentences_path=output_path)
    #
    # text = u'北京科锐配电自动化股份有限公司关于项目中标的提示性公告。\n' \
    #        u'本公司及董事会全体成员保证公告内容真实、准确和完整，并对公告中的虚假记载、误导性陈述或者重大遗漏承担责任。'
    # center_text=cs.use_model_text(text)
    # print(center_text)
    #
    print('----------\nsucessfully, center sentences in path: \n',output_path,'\n-------------')

if __name__ == "__main__":
    main(input_path='/home/118_16/data/hetong_testb_html/',output_path='/home/118_16/data/hetong_testb_center_txt/')
