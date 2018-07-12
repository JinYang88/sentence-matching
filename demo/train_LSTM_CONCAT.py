import sys
sys.path.append(".")
sys.path.append("./model")
import pandas as pd
from utils import dataloader
from utils import evaluator
from utils import train_model
from LSTM_CONCAT.LSTM_CONCAT import LSTM_CONCAT


train_path = "data/quora/train.tsv"
valid_path = "data/quora/dev.tsv"
embedding_path = "data/quora/wordvec.txt"
dictionary_path = "data/quora/dictionary.bin"
# pretrain_path = "data/quora/wordvec.txt"
pretrain_path = None
checkpoint_path = "demo/checkpoints/LSTM_CONCAT.pth"
device = "cpu"
epochs = 10
batch_size = 32
fixlen = 20
early_stop_num = 15

print("Using device {}!".format(device))
print("Loading data!")
train_df = pd.read_csv(train_path, sep="\t", nrows=10000)
valid_df = pd.read_csv(valid_path, sep="\t", nrows=10000)

print("Building vocab!")
dictionary = dataloader.build_vocab(dataframes=train_df,
                       text_columns=["text1", "text2"],
                       save_path=dictionary_path)

print("Building iterator!")
train_iter, valid_iter = dataloader.load_batched_data_from(train_df,
                                                           valid_df,
                                                           dictionary,
                                                           fixlen=fixlen,
                                                           batch_size=batch_size)

if pretrain_path:
    print("Loading pretrain embedding!")
    word_matrix = dataloader.load_embedding_matrix(len(dictionary)+2, pretrain_path,
                                               dictionary, dim=300, norm=True)
else:
    print("Learn embeddings during training!")
    word_matrix=None

print("Building model!")
MODEL = LSTM_CONCAT(len(dictionary)+2, device, word_matrix)

print("Training!")
train_model.train(MODEL, train_iter, valid_iter,
                  checkpoint_path, device=device, epochs=epochs,
                  print_every=5, early_stop_num=15)