import pickle as pk

import numpy as np

import torch
import torch.nn.functional as F

from represent import sent2ind, add_buf

from util import map_item


def ind2label(label_inds):
    ind_labels = dict()
    for label, ind in label_inds.items():
        ind_labels[ind] = label
    return ind_labels


device = torch.device('cpu')

seq_len = 50

path_word_ind = 'feat/word_ind.pkl'
path_label_ind = 'feat/label_ind.pkl'
with open(path_word_ind, 'rb') as f:
    word_inds = pk.load(f)
with open(path_label_ind, 'rb') as f:
    label_inds = pk.load(f)

ind_labels = ind2label(label_inds)

paths = {'cnn': 'model/cnn.pkl',
         'rnn': 'model/rnn.pkl'}

models = {'cnn': torch.load(map_item('cnn', paths), map_location=device),
          'rnn': torch.load(map_item('rnn', paths), map_location=device)}


def predict(text, name):
    text = text.strip()
    pad_seq = sent2ind(text, word_inds, seq_len, keep_oov=True)
    if name == 'cnn':
        pad_seq = add_buf([pad_seq])[0]
    sent = torch.LongTensor([pad_seq]).to(device)
    model = map_item(name, models)
    with torch.no_grad():
        model.eval()
        probs = F.softmax(model(sent), dim=-1)
    probs = probs.numpy()[0]
    inds = np.argmax(probs, axis=1)
    preds = [ind_labels[ind] for ind in inds[-len(text):]]
    pairs = list()
    for word, pred in zip(text, preds):
        pairs.append((word, pred))
    return pairs


if __name__ == '__main__':
    while True:
        text = input('text: ')
        print('cnn: %s' % predict(text, 'cnn'))
        print('rnn: %s' % predict(text, 'rnn'))
