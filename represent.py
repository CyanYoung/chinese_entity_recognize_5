import json
import pickle as pk

import numpy as np

from gensim.corpora import Dictionary

from util import sent2label


embed_len = 200
min_freq = 1
max_vocab = 5000
win_len = 7
seq_len = 50

pad_ind, oov_ind = 0, 1

path_word_vec = 'feat/word_vec.pkl'
path_word_ind = 'feat/word_ind.pkl'
path_embed = 'feat/embed.pkl'
path_label_ind = 'feat/label_ind.pkl'


def tran_dict(word_inds, off):
    off_word_inds = dict()
    for word, ind in word_inds.items():
        off_word_inds[word] = ind + off
    return off_word_inds


def embed(text_words, path_word_ind, path_word_vec, path_embed):
    model = Dictionary(text_words)
    model.filter_extremes(no_below=min_freq, no_above=1.0, keep_n=max_vocab)
    word_inds = model.token2id
    word_inds = tran_dict(word_inds, off=2)
    with open(path_word_ind, 'wb') as f:
        pk.dump(word_inds, f)
    with open(path_word_vec, 'rb') as f:
        word_vecs = pk.load(f)
    vocab = word_vecs.vocab
    vocab_num = min(max_vocab + 2, len(word_inds) + 2)
    embed_mat = np.zeros((vocab_num, embed_len))
    for word, ind in word_inds.items():
        if word in vocab:
            if ind < max_vocab:
                embed_mat[ind] = word_vecs[word]
    with open(path_embed, 'wb') as f:
        pk.dump(embed_mat, f)


def sent2ind(words, word_inds, seq_len, keep_oov):
    seq = list()
    for word in words:
        if word in word_inds:
            seq.append(word_inds[word])
        elif keep_oov:
            seq.append(oov_ind)
    return pad(seq, seq_len)


def pad(seq, seq_len):
    if len(seq) < seq_len:
        return [pad_ind] * (seq_len - len(seq)) + seq
    else:
        return seq[-seq_len:]


def label2ind(sents, path_label_ind):
    labels = list()
    for pairs in sents.values():
        labels.extend(sent2label(pairs))
    labels = sorted(list(set(labels)))
    label_inds = dict()
    label_inds['N'] = 0
    for i in range(len(labels)):
        label_inds[labels[i]] = i + 1
    with open(path_label_ind, 'wb') as f:
        pk.dump(label_inds, f)


def add_buf(seqs):
    buf = [0] * int((win_len - 1) / 2)
    buf_seqs = list()
    for seq in seqs:
        buf_seqs.append(buf + seq + buf)
    return buf_seqs


def align_sent(text_words, path_sent, extra):
    with open(path_word_ind, 'rb') as f:
        word_inds = pk.load(f)
    pad_seqs = list()
    for words in text_words:
        pad_seq = sent2ind(words, word_inds, seq_len, keep_oov=True)
        pad_seqs.append(pad_seq)
    if extra:
        pad_seqs = add_buf(pad_seqs)
    pad_seqs = np.array(pad_seqs)
    with open(path_sent, 'wb') as f:
        pk.dump(pad_seqs, f)


def align_label(sents, path_label):
    with open(path_label_ind, 'rb') as f:
        label_inds = pk.load(f)
    ind_mat = list()
    for pairs in sents.values():
        inds = list()
        for pair in pairs:
            inds.append(label_inds[pair['label']])
        ind_mat.append(pad(inds, seq_len))
    ind_mat = np.array(ind_mat)
    with open(path_label, 'wb') as f:
        pk.dump(ind_mat, f)


def vectorize(path_data, path_cnn_sent, path_rnn_sent, path_label, mode):
    with open(path_data, 'r') as f:
        sents = json.load(f)
    texts = sents.keys()
    text_words = [list(text) for text in texts]
    if mode == 'train':
        embed(text_words, path_word_ind, path_word_vec, path_embed)
        label2ind(sents, path_label_ind)
    align_sent(text_words, path_cnn_sent, extra=True)
    align_sent(text_words, path_rnn_sent, extra=False)
    align_label(sents, path_label)


if __name__ == '__main__':
    path_data = 'data/train.json'
    path_cnn_sent = 'feat/cnn_sent_train.pkl'
    path_rnn_sent = 'feat/rnn_sent_train.pkl'
    path_label = 'feat/label_train.pkl'
    vectorize(path_data, path_cnn_sent, path_rnn_sent, path_label, 'train')
    path_data = 'data/dev.json'
    path_cnn_sent = 'feat/cnn_sent_dev.pkl'
    path_rnn_sent = 'feat/rnn_sent_dev.pkl'
    path_label = 'feat/label_dev.pkl'
    vectorize(path_data, path_cnn_sent, path_rnn_sent, path_label, 'dev')
