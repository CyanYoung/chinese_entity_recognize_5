import json

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from recognize import label_inds, ind_labels, predict

from util import map_item


seq_len = 50

path_sent = 'data/test.json'
with open(path_sent, 'r') as f:
    sents = json.load(f)

class_num = len(label_inds)

label_set = list(ind_labels.keys())
label_set.remove(label_inds['N'])
label_set.remove(label_inds['O'])

paths = {'trm': 'metric/trm.csv'}


def test(name, sents):
    flat_labels, flat_preds = [0], [0]
    for text, pairs in sents.items():
        labels = list()
        for pair in pairs:
            labels.append(label_inds[pair['label']])
        bound = seq_len if len(text) > seq_len else len(text)
        flat_labels.extend(labels[:bound])
        flat_preds.extend(predict(text, name))
    precs = precision_score(flat_labels, flat_preds, average=None)
    recs = recall_score(flat_labels, flat_preds, average=None)
    with open(map_item(name, paths), 'w') as f:
        f.write('label,prec,rec' + '\n')
        for i in range(1, class_num):
            f.write('%s,%.2f,%.2f\n' % (ind_labels[i], precs[i], recs[i]))
    f1 = f1_score(flat_labels, flat_preds, average='weighted', labels=label_set)
    print('\n%s f1: %.2f - acc: %.2f' % (name, f1, accuracy_score(flat_labels, flat_preds)))


if __name__ == '__main__':
    test('trm', sents)
