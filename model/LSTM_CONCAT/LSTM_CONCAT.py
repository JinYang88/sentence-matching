import torch.nn as nn
import torch
import torch.nn.functional as F

class LSTM_CONCAT(torch.nn.Module) :
    def __init__(self,
                 vocab_size,
                 device,
                 word_matrix=None,
                 embedding_dim=300,
                 hidden_dim=300
                 ):
        super(LSTM_CONCAT, self).__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.word_embedding = nn.Embedding(vocab_size, embedding_dim)

        if word_matrix is not None:
            word_matrix = word_matrix.to(self.device)
            self.word_embedding.weight.data.copy_(word_matrix)
            self.word_embedding.weight.requires_grad = False
        
        self.lstm = nn.LSTM(embedding_dim, hidden_dim//2, batch_first=True, bidirectional=True)
        self.linear1 = nn.Linear(embedding_dim * 2, 200)
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(200)
        self.linear2 = nn.Linear(200, 200)
        self.batchnorm2 = nn.BatchNorm1d(200)
        self.linear3 = nn.Linear(200, 200)
        self.batchnorm3 = nn.BatchNorm1d(200)
        self.linear4 = nn.Linear(200, 200)
        self.batchnorm4 = nn.BatchNorm1d(200)
        self.linear5 = nn.Linear(200, 1)
        
    def forward(self, batched_data) :
        text1 = torch.tensor(batched_data[1]).to(self.device)
        text2 = torch.tensor(batched_data[2]).to(self.device)
        self.batch_size = text1.size()[0]
        text1_word_embedding = self.word_embedding(text1)
        text2_word_embedding = self.word_embedding(text2)
        text1_seq_embedding = self.lstm_embedding(self.lstm, text1_word_embedding)
        text2_seq_embedding = self.lstm_embedding(self.lstm, text2_word_embedding)
        feature_vec = torch.cat((text1_seq_embedding,text2_seq_embedding), dim=1)
        
        merged = self.linear1(feature_vec)
        merged = F.relu(merged)
        merged = self.dropout(merged)
        merged = self.batchnorm1(merged)

        merged = self.linear2(merged)
        merged = F.relu(merged)
        merged = self.dropout(merged)
        merged = self.batchnorm2(merged)

        merged = self.linear3(merged)
        merged = F.relu(merged)
        merged = self.dropout(merged)
        merged = self.batchnorm3(merged)

        merged = self.linear4(merged)
        merged = F.relu(merged)
        merged = self.dropout(merged)
        merged = self.batchnorm4(merged)

        merged = self.linear5(merged)

        return merged
    
    def lstm_embedding(self, lstm, word_embedding):
        lstm_out,(lstm_h, lstm_c) = lstm(word_embedding)
        seq_embedding = torch.cat((lstm_h[0], lstm_h[1]), dim=1)
        return seq_embedding

    def init_hidden(self, batch_size, device) :
        layer_num = 2 if self.bidirectional else 1
        if device == -1:
            return (Variable(torch.randn(layer_num, batch_size, self.hidden_dim//layer_num)),Variable(torch.randn(layer_num, batch_size, self.hidden_dim//layer_num)))  
        else:
            return (Variable(torch.randn(layer_num, batch_size, self.hidden_dim//layer_num).cuda()),Variable(torch.randn(layer_num, batch_size, self.hidden_dim//layer_num).cuda()))  

