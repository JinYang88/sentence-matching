# coding: utf-8
import sys
import torch
import time
import pickle
import pandas as pd
from gensim import corpora, models, similarities
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

def test_on(model, data_dl, output_path=None, checkpoint_path=None):
    print("loading model from {}!".format(checkpoint_path))
    model.load_state_dict(torch.load(checkpoint_path,  map_location=lambda storage, loc: storage))
    print("Sucessfully loaded!")

    model.eval()
    pred_list = []
    ids = []
    prob_list = []
    threshold = 0.5

    for batch_data in data_dl:
        ids.extend(batch_data[0])
        y_pred = model(batch_data).sigmoid()
        prob_list.extend(y_pred.data.numpy().reshape(-1))
        y_pred = list(map(lambda x: 1 if x >= threshold else 0, y_pred))
        pred_list.extend(y_pred)
    
    if output_path:
        with open(output_path, 'w') as fw:
            for idx, item in enumerate(pred_list):
                fw.write('{}\t{}\n'.format(ids[idx], item))
    return ids, prob_list, pred_list