import json
import pickle as pk

from sklearn.metrics import f1_score, accuracy_score

from recognize import predict


seq_len = 50

path_sent = 'data/test.json'
path_label_ind = 'feat/label_ind.pkl'
with open(path_sent, 'r') as f:
    sents = json.load(f)
with open(path_label_ind, 'rb') as f:
    label_inds = pk.load(f)

slots = list(label_inds.keys())
slots.remove('N')
slots.remove('O')


def flat(labels):
    flat_labels = list()
    for label in labels:
        flat_labels.extend(label)
    return flat_labels


def test(name, sents):
    label_mat, pred_mat = list(), list()
    for text, quaples in sents.items():
        labels = list()
        for quaple in quaples:
            labels.append(quaple['label'])
        bound = seq_len if len(text) > seq_len else len(text)
        label_mat.append(labels[:bound])
        pairs = predict(text, name)
        preds = [pred for word, pred in pairs]
        pred_mat.append(preds)
    labels, preds = flat(label_mat), flat(pred_mat)
    f1 = f1_score(labels, preds, average='weighted', labels=slots)
    print('\n%s f1: %.2f - acc: %.2f' % (name, f1, accuracy_score(labels, preds)))


if __name__ == '__main__':
    test('trm', sents)
