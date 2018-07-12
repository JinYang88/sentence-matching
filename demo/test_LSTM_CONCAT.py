import sys
sys.path.append(".")
sys.path.append("./model")
import pandas as pd
from utils import dataloader
from utils import evaluator
from utils import train_model
from gensim import corpora
from LSTM_CONCAT.LSTM_CONCAT import LSTM_CONCAT
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

test_path = "data/quora/test.tsv"
dictionary_path = "data/quora/dictionary.bin"
checkpoint_path = "demo/checkpoints/LSTM_CONCAT.pth"
output_path = "data/quora/predict.txt"

device = "cpu"
epochs = 10
batch_size = 32
fixlen = 20
early_stop_num = 15

print("Using device {}!".format(device))
print("Loading data!")
test_df = pd.read_csv(test_path, sep="\t", nrows=10000)

print("Loading dictionary!")
dictionary = corpora.Dictionary.load(dictionary_path)

print("Building iterator!")
test_iter = dataloader.load_batched_data_from(test_df,
                                              dictionary=dictionary,
                                              fixlen=fixlen,
                                              batch_size=batch_size)
print("Building model!")
MODEL = LSTM_CONCAT(len(dictionary)+2, device)

print("Copy parameters and evaluate!")
ids, prob_list, pred_list = evaluator.test_on(MODEL, test_iter, output_path=output_path, checkpoint_path=checkpoint_path)

print("Finish testing!")
print("----------------")
print("Acc: [{}]!".format(accuracy_score(test_df["label"], pred_list)))
print("F1: [{}]!".format(f1_score(test_df["label"], pred_list)))
print("Recall: [{}]!".format(recall_score(test_df["label"], pred_list)))
print("Precision: [{}]!".format(precision_score(test_df["label"], pred_list)))
print("----------------")
