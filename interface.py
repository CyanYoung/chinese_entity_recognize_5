import json

from recognize import ind_labels, predict

from util import load_pair, get_logger


path_zh_en = 'dict/zh_en.csv'
zh_en = load_pair(path_zh_en)

path_log_dir = 'log'
logger = get_logger('recognize', path_log_dir)


def insert(entity, label, entitys, labels):
    entitys.append(entity)
    labels.append(zh_en[label])


def make_dict(entitys, labels):
    slot_dict = dict()
    for label, entity in zip(labels, entitys):
        if label not in slot_dict:
            slot_dict[label] = list()
        slot_dict[label].append(entity)
    return slot_dict


def merge(text, preds):
    entitys, labels = list(), list()
    entity, label = [''] * 2
    for word, pred in zip(text, preds):
        pred = ind_labels[pred]
        if pred[:2] == 'B-':
            if entity:
                insert(entity, label, entitys, labels)
            entity = word
            label = pred[2:]
        elif pred[:2] == 'I-' and entity:
            entity = entity + word
        elif entity:
            insert(entity, label, entitys, labels)
            entity = ''
    if entity:
        insert(entity, label, entitys, labels)
    return make_dict(entitys, labels)


def response(text, name):
    data = dict()
    preds = predict(text, name)
    slot_dict = merge(text, preds)
    data['content'] = text
    data['slot'] = slot_dict
    data_str = json.dumps(data, ensure_ascii=False)
    logger.info(data_str)
    return data_str


if __name__ == '__main__':
    while True:
        text = input('text: ')
        print(response(text, 'trm'))
