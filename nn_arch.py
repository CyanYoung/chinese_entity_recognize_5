import torch
import torch.nn as nn


win_len = 7


class Cnn(nn.Module):
    def __init__(self, embed_mat, class_num):
        super(Cnn, self).__init__()
        vocab_num, embed_len = embed_mat.size()
        self.embed = nn.Embedding(vocab_num, embed_len, _weight=embed_mat)
        self.conv = nn.Conv1d(embed_len, 128, kernel_size=win_len, padding=0)
        self.gate = nn.Conv1d(embed_len, 128, kernel_size=win_len, padding=0)
        self.la = nn.Sequential(nn.Linear(128, 200),
                                nn.ReLU())
        self.dl = nn.Sequential(nn.Dropout(0.2),
                                nn.Linear(200, class_num))

    def forward(self, x):
        x = self.embed(x)
        x = x.permute(0, 2, 1)
        g = torch.sigmoid(self.gate(x))
        x = self.conv(x)
        x = x * g
        x = x.permute(0, 2, 1)
        x = self.la(x)
        return self.dl(x)


class Rnn(nn.Module):
    def __init__(self, embed_mat, class_num):
        super(Rnn, self).__init__()
        vocab_num, embed_len = embed_mat.size()
        self.embed = nn.Embedding(vocab_num, embed_len, _weight=embed_mat)
        self.ra = nn.LSTM(embed_len, 200, batch_first=True, bidirectional=True)
        self.dl = nn.Sequential(nn.Dropout(0.2),
                                nn.Linear(400, class_num))

    def forward(self, x):
        x = self.embed(x)
        h, hc_n = self.ra(x)
        return self.dl(h)
