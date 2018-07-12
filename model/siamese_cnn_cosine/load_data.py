# coding = utf-8
import os
import pandas as pd
import numpy as np
import re
import random
from env.torchtext import data
from datetime import datetime
import traceback  
import env.torchtext.datasets as datasets
import pickle
from gensim.models import Word2Vec
import jieba

jieba.load_userdict('data/special_word.txt')





def preprocess(data_path):
    '''
    convert Chinese sentences to word lists
    '''
    df = pd.read_csv(data_path, sep='\t', names=['id', 'q1', 'q2', 'label'])
    df['q1_list'] = df['q1'].apply(lambda x: [str(i.encode('utf-8')) for i in jieba.cut(x, cut_all=True, HMM=False) if len(i)])
    df['q2_list'] = df['q2'].apply(lambda x: [str(i.encode('utf-8')) for i in jieba.cut(x, cut_all=True, HMM=False) if len(i)])
    return df

def train_word2vec_model(df):
    '''
    basic w2v model trained by sentences
    '''
    corpus = []
    for i, r in df.iterrows():
        corpus.append(r['q1_list'])
        corpus.append(r['q2_list'])
    word2vec_model = Word2Vec(corpus, size=300, window=3, min_count=1, sg=0)
    return word2vec_model


def load_glove_as_dict(filepath):
    word_vec = {}
    with open(filepath) as fr:
        for line in fr:
            line = line.split()
            word = line[0]
            vec = line[1:]
            word_vec[word] = vec
    return word_vec

def gen_iter(path, text_field, label_field, args):
    '''
        Load TabularDataset from path,
        then convert it into a iterator
        return TabularDataset and iterator
    '''
    tmp_data = data.TabularDataset(
                            path=path,
                            format='tsv',
                            skip_header=False,
                            fields=[
                                    ('question1', text_field),
                                    ('question2', text_field),
                                    ('label', label_field)
                                    ])

    tmp_iter = data.BucketIterator(
                    tmp_data,
                    batch_size=args.batch_size,
                    sort_key=lambda x: len(x.question1) + len(x.question2),
                    device=-1, # 0 for GPU, -1 for CPU
                    repeat=False)
    return tmp_data, tmp_iter


def gen_iter_test(path, text_field, label_field, args):
    '''
        Load TabularDataset from path,
        then convert it into a iterator
        return TabularDataset and iterator
    '''
    tmp_data = data.TabularDataset(
                            path=path,
                            format='tsv',
                            skip_header=False,
                            fields=[
                                    ('pid', label_field),
                                    ('question1', text_field),
                                    ('question2', text_field)
                                    ])

    # tmp_iter = data.BucketIterator(
    #                 tmp_data,
    #                 batch_size=args.batch_size,
    #                 sort_key=lambda x: len(x.question1) + len(x.question2),
    #                 device=-1, # 0 for GPU, -1 for CPU
    #                 repeat=False)
    tmp_iter = data.Iterator(
                    dataset=tmp_data,
                    batch_size=args.batch_size,
                    device=-1, # 0 for GPU, -1 for CPU
                    shuffle=False,
                    repeat=False)
    return tmp_data, tmp_iter



def preprocess(data_path):
    '''
    convert Chinese sentences to word lists
    '''
    df = pd.read_csv(data_path, sep='\t', names=['id', 'q1', 'q2', 'label'])
    df['q1_list'] = df['q1'].apply(lambda x: [str(i.encode('utf-8')) for i in jieba.cut(x, cut_all=True, HMM=False) if len(i)])
    df['q2_list'] = df['q2'].apply(lambda x: [str(i.encode('utf-8')) for i in jieba.cut(x, cut_all=True, HMM=False) if len(i)])
    return df


def preprocess_test(data_path):
    '''
    convert Chinese sentences to word lists
    '''
    df = pd.read_csv(data_path, sep='\t', names=['id', 'q1', 'q2'])
    #print df.head()
    df['q1_list'] = df['q1'].apply(lambda x: [str(i.encode('utf-8')) for i in jieba.cut(x, cut_all=True, HMM=False) if len(i)])
    df['q2_list'] = df['q2'].apply(lambda x: [str(i.encode('utf-8')) for i in jieba.cut(x, cut_all=True, HMM=False) if len(i)])
    return df

def load_data(args):
    '''
        1. train the w2v_model
        2. split the data as 9:1(train:dev)
        3. load the data
        load as pairs
    '''
    df = preprocess(args.data_path)
    #print ('Positive in set: %s' %str(len(df[df['label'] == 1])))
    #print ('Negative in set: %s' %str(len(df[df['label'] == 0])))
    word2vec_model = train_word2vec_model(df)
    word2vec_model.save(args.w2v_model_path)
    df = df[['q1_list', 'q2_list', 'label']]
    df['q1_list'] = df['q1_list'].apply(lambda x: ' '.join(x))
    df['q2_list'] = df['q2_list'].apply(lambda x: ' '.join(x))
    train_df = df.head(int(len(df)*0.9))
    train_df_rev = pd.DataFrame()
    train_df_rev['q1_list'] = train_df['q2_list']
    train_df_rev['q2_list'] = train_df['q1_list']
    train_df_rev['label'] = train_df['label']
    train_df = pd.concat([train_df, train_df_rev])
    dev_df   = df.tail(int(len(df)*0.1))
    # print 'Positive in dev set', len(dev_df[dev_df['label'] == 1])
    # print 'Negative in dev set', len(dev_df[dev_df['label'] == 0])
    train_df.to_csv(args.train_path, index=False, encoding='utf-8', sep='\t', header=None)
    # add qoura
    # with open('data/qoura_train.tsv' , 'r') as fqoura, open('data/combine.csv', 'w') as fcomb:
    #     for line in fqoura:
    #         label, q1, q2, pid = line.strip().split('\t')
    #         fcomb.write('%s\t%s\t%s\n' %(q1, q2, label))
    # with open(args.train_path , 'r') as ftrain, open('data/combine.csv', 'a') as fcomb:
    #     for line in ftrain:
    #         fcomb.write(line)

    dev_df.to_csv(args.dev_path, index=False, encoding='utf-8', sep='\t', header=None)

    
    text_field    = data.Field(sequential=True, use_vocab=True, batch_first=True, eos_token='<EOS>', init_token='<BOS>', pad_token='<PAD>')
    label_field   = data.Field(sequential=False, use_vocab=False)
    
    # train_data, train_iter = gen_iter(args.train_path, text_field, label_field, args)
    # dev_data, dev_iter     = gen_iter(args.dev_path, text_field, label_field, args)
    
    # text_field.build_vocab(train_data, dev_data)

    # return text_field, label_field, \
    #     train_data, train_iter,\
    #     dev_data, dev_iter

    df_test = preprocess_test(args.test_path)
    df_test['q1_list'] = df_test['q1_list'].apply(lambda x: ' '.join(x))
    df_test['q2_list'] = df_test['q2_list'].apply(lambda x: ' '.join(x))
    df_test = df_test[['id', 'q1_list', 'q2_list']]
    df_test.to_csv(args.to_test_path, index=False, encoding='utf-8', sep='\t', header=None)
    
    # train_data, train_iter = gen_iter('data/combine.csv', text_field, label_field, args)
    train_data, train_iter = gen_iter(args.train_path, text_field, label_field, args)
    dev_data, dev_iter     = gen_iter(args.dev_path, text_field, label_field, args)
    test_data, test_iter   = gen_iter_test(args.to_test_path, text_field, label_field, args)
    text_field.build_vocab(train_data, dev_data)

    return text_field, label_field, \
        train_data, train_iter,\
        dev_data, dev_iter,\
        test_data, test_iter
          
