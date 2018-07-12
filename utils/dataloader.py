# coding: utf-8
import sys
import torch
import time
import os
import numpy as np
from numpy import linalg as LA
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from gensim import corpora

class BatchWrapper:
    def __init__(self, df, iter_columns, dictionary, batch_size, fixlen=20, numerical_columns=None):
        self.df = df
        self.iter_columns = iter_columns
        self.dictionary = dictionary
        self.batch_size = batch_size
        self.unknown_idx = len(dictionary)
        self.batch_num = self.df.shape[0] // batch_size
        self.fixlen = fixlen
        self.numerical_columns = numerical_columns

    def shuffle(self):
        self.df = self.df.sample(frac=1).reset_index(drop=True)

    def pad(self, text):
        text = self.dictionary.doc2idx(text.split(), self.unknown_idx)[0: self.fixlen]
        text = text + [self.unknown_idx + 1] * (self.fixlen - len(text))
        return text

    def __iter__(self):
        yield_batch = []
        for idx, row in self.df.iterrows():
            for name in self.numerical_columns:
                row[name] = self.pad(row[name])
            yield_batch.append(row[self.iter_columns].values)
            if len(yield_batch) == self.batch_size:
                yield np.array(yield_batch).T.tolist()
                yield_batch = []
        if yield_batch:
            yield np.array(yield_batch).T.tolist()

def load_batched_data_from(data_df, valid_df=None, dictionary=None,
                           focus_columns=["id", "text1", "text2","label"],
                           numerical_columns=["text1", "text2"],
                           fixlen=20, batch_size=32):
    if valid_df is not None:
        train_iter = BatchWrapper(data_df, focus_columns, dictionary,
                                                        batch_size=batch_size, fixlen=fixlen,
                                                        numerical_columns=numerical_columns)
        valid_iter = BatchWrapper(valid_df, focus_columns, dictionary,
                                                             batch_size=batch_size, fixlen=fixlen,
                                                             numerical_columns=numerical_columns)
        return train_iter, valid_iter
    else:
        data_iter = BatchWrapper(data_df, focus_columns, dictionary,
                                                             batch_size=batch_size, fixlen=fixlen,
                                                             numerical_columns=numerical_columns)
        return data_iter

def load_embedding_matrix(vocab_size, pretrain_path, id2token, dim=300, norm=True):
    try:
        word_vec = Word2Vec.load(pretrain_path)
    except:
        word_vec = KeyedVectors.load_word2vec_format(pretrain_path, binary=False)
    dim = word_vec.vector_size
    word_vec = word_vec.wv
    word_vec_list = []
    oov = 0
    for word_idx in range(vocab_size):
        try:
            word = id2token[word_idx]
            vector = np.array(word_vec[word], dtype=float).reshape(1,dim)
        except Exception as e:
            oov += 1
            vector = np.random.rand(1, dim)
        if norm:
            vector = vector / LA.norm(vector, 2)
        word_vec_list.append(torch.from_numpy(vector))
    wordvec_matrix = torch.cat(word_vec_list)
    print("Load embedding finished.")
    print("Total words count: {}, oov count: {}.".format(wordvec_matrix.size()[0], oov))
    return wordvec_matrix if device == -1 else wordvec_matrix.to(device)

def build_vocab(dataframes, text_columns, save_path, tokenizer=None,
                no_below=5, no_above=0.5, keep_n=100000):
    sentences = []
    if isinstance(dataframes, list):
        for df in dataframes:
            for col in text_columns:
                sentences.extend(df[col].tolist())
    else:
        for col in text_columns:
            sentences.extend(dataframes[col].tolist())
    if tokenizer is not None:
        sentences = list(map(lambda x: tokenizer(x), sentences))
    else:
        sentences = list(map(lambda x: x.split(), sentences))
    dictionary = corpora.Dictionary(sentences)
    dictionary.filter_extremes(no_below=no_below, no_above=no_above)
    dictionary.compactify()
    print("Saving dictionary into {}!".format(save_path))
    dictionary.save(save_path)
    # dictionary.save_as_text("vocab.txt")
    return dictionary
