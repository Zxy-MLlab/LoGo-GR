import re
import codecs
import json

import random

def read_origin_data():
    data_path = ['data/drug_var/0/data_graph_1', 'data/drug_var/0/data_graph_2',
                          'data/drug_var/1/data_graph_1', 'data/drug_var/1/data_graph_2',
                          'data/drug_var/2/data_graph_1', 'data/drug_var/2/data_graph_2',
                          'data/drug_var/3/data_graph_1', 'data/drug_var/3/data_graph_2',
                          'data/drug_var/4/data_graph_1', 'data/drug_var/4/data_graph_2']

    relation_set = {'resistance or non-response': 0, 'sensitivity': 1, 'response': 2, 'resistance': 3, 'None': 4, }

    data = []

    ins = {}

    for path in data_path:
        with codecs.open(path, 'rU', 'utf-8') as f_read:
            for inst in json.load(f_read):

                sents = []

                sentences = inst['sentences']

                for sen in sentences:

                    sentence = []
                    for node in sen['nodes']:
                        word = node['label']
                        sentence.append(word)

                    sents.append(sentence)

                ins['sents'] = sents

                entities = inst['entities']

                ens = {}

                for i, entity in enumerate(entities):
                    entity_id = entity['id']
                    entity_name = entity['mention']
                    entity_type = entity['type']
                    try:
                        entity_pos = [entity['indices'][0], entity['indices'][len(entity['indices'])-1]+1]
                    except:
                        entity_pos = [8, 8+1]
                        entities[1]['indices'] = [entities[1]['indices'][0]-1]

                    e = {
                        'id': entity_id,
                        'name': entity_name,
                        'type': entity_type,
                        'pos': entity_pos,
                    }

                    if entity_id not in ens.keys():
                        ens[entity_id] = [e]
                    else:
                        ens[entity_id].append(e)

                    if i == 0:
                        h = entity_id
                    else:
                        t = entity_id

                ins['entities'] = ens

                r = relation_set[inst['relationLabel'].strip()]

                label = dict(
                    h=h,
                    t=t,
                    r=r
                )
                ins['labels'] = [label]

                title = inst['article']

                title_id = inst['article']

                ins['title_id'] = title_id

                ins['title'] = title

                data.append(ins)

                ins = {}

    return data

def process(data):
    output_data = []
    for ins in data:

        output_ins = {
            'title_id': ins['title_id'],
            'title': ins['title'],
            'sents': ins['sents'],
        }

        sen_lens = [len(sen) for sen in ins['sents']]

        entities = []

        for entity_id in ins['entities'].keys():
            entity = ins['entities'][entity_id]

            for i, e in enumerate(entity):
                entity_pos = e['pos'][0]

                sen_len = 0

                sen_id = 0

                for s_i, opt in enumerate(sen_lens):
                    sen_len += opt

                    if entity_pos < sen_len:
                        sen_id = s_i
                        break

                entity[i]['sent_id'] = sen_id

            entities.append(entity)

        output_ins['vertexSet'] = entities

        labels = ins['labels']

        for label in labels:
            label['intrain'] = False
            label['indev_train'] = False

        output_ins['labels'] = labels

        output_data.append(output_ins)

    return output_data

def write_data(output_data):
    random.shuffle(output_data)

    train_data = output_data[ : int(len(output_data) * 0.9)]

    test_data = output_data[int(len(output_data) * 0.9) : ]

    json.dump(train_data, open('prepro_data/train.json', "w"))

    json.dump(test_data, open('prepro_data/test.json', "w"))

    return

def main():
    data = read_origin_data()

    output_data = process(data)

    write_data(output_data)

    return

if __name__ == '__main__':
    main()