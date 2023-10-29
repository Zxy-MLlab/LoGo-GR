import re
from stanfordcorenlp import StanfordCoreNLP
import json
import os
import tqdm


origin_path = 'data/CDR.Corpus.v010516'
data_sets = ['CDR_TrainingSet.PubTator.txt', 'CDR_DevelopmentSet.PubTator.txt', 'CDR_TestSet.PubTator.txt']

out_path = 'prepro_data'

def read_ori(data_set, ):
    dat_set_path = origin_path + '/' + data_set

    data = []

    title_index = 0
    doc_index = 0

    ins = {}
    entities = {}
    labels = []

    with open(dat_set_path, 'r', encoding='utf-8') as f_read:
        for d in f_read:

            if d == '\n':
                title_index = 0
                doc_index = 0


                ins['entities'] = entities
                ins['labels'] = labels

                data.append(ins)

                ins = {}
                entities = {}
                labels = []

                continue

            else:
                title_index += 1
                doc_index += 1

            if title_index == 1:
                d = d.strip().split('|t|')
                title_id = d[0]
                title = d[1]

                ins['title_id'] = title_id
                ins['title'] = title
            elif doc_index == 2:
                d = d.strip().split('|a|')
                doc = d[1]

                sents = [ins['title']] + doc.split('. ')

                for i, sent in enumerate(sents):
                    if i > 0 and i < len(sents)-1:
                        sents[i] = sent + '.'

                ins['sents'] = sents
            else:
                d = d.strip().split('\t')

                if d[1] == 'CID':
                    h = d[2]
                    t = d[3]
                    r = 1

                    label = dict(
                        h=h,
                        t=t,
                        r=r,
                    )

                    labels.append(label)

                else:
                    entity_start = int(d[1])
                    entity_end = int(d[2])
                    entity = d[3]
                    entity_type = d[4]
                    entity_id = d[5].split('|')[0]

                    if entity_id != '-1':

                        e = {
                            'id': entity_id,
                            'name': entity,
                            'type': entity_type,
                            'pos': [entity_start, entity_end],
                        }

                        if entity_id not in entities.keys():
                            entities[entity_id] = [e]
                        else:
                            entities[entity_id].append(e)

    f_read.close()

    return data

def process(data, nlp, name_prefix):

    pro_data = []

    for d in tqdm.tqdm(data):
        ins = {}

        char_to_index = {}
        sents = d['sents']
        char_index = 0
        word_index = 0
        sent_index = 0
        for sent in sents:
            word_sent = sent.split(' ')

            for word in word_sent:

                for char in word:
                    if char_index not in char_to_index.keys():
                        char_to_index[char_index] = {
                            'char': char,
                            'word_index': word_index,
                            'sent_index': sent_index,
                        }

                    char_index = char_index + 1

                if char_index not in char_to_index.keys():
                    char_to_index[char_index] = {
                        'char': '#w',
                        'word_index': word_index,
                        'sent_index': sent_index,
                    }

                char_index = char_index + 1

                word_index = word_index + len(nlp.word_tokenize(word))

            sent_index = sent_index + 1

        ins['title'] = d['title']
        ins['sents'] = [nlp.word_tokenize(s) for s in d['sents']]

        id_to_index = {}
        vertexSet = []
        for index, e_id in enumerate(d['entities'].keys()):

            if e_id not in id_to_index.keys():
                id_to_index[e_id] = index

            entity = d['entities'][e_id]
            entity_set = []

            for mention in entity:

                m_id = mention['id']
                m_name = mention['name']
                m_type = mention['type']
                pos_start = mention['pos'][0]
                pos_end = mention['pos'][1]

                assert pos_start in char_to_index.keys() and pos_end in char_to_index.keys()

                sent_id = char_to_index[pos_start]['sent_index']

                pos_start = char_to_index[pos_start]['word_index']
                pos_end = char_to_index[pos_end]['word_index'] + 1

                mention_dic = {
                    'id': m_id,
                    'name': m_name,
                    'type': m_type,
                    'pos': [pos_start, pos_end],
                    'sent_id': sent_id,
                }

                entity_set.append(mention_dic)

            vertexSet.append(entity_set)

        ins['vertexSet'] = vertexSet

        pro_labels = []
        for label in d['labels']:
            h = label['h']
            t = label['t']
            r = label['r']

            if h not in id_to_index or t not in id_to_index:
                continue

            h_index = id_to_index[h]
            t_index = id_to_index[t]

            intrain = False
            indev_train = False

            pro_label = {
                'h': h_index,
                't': t_index,
                'r': r,
                'intrain': intrain,
                'indev_train': indev_train,
            }

            pro_labels.append(pro_label)

        ins['labels'] = pro_labels

        pro_data.append(ins)

    json.dump(pro_data, open(os.path.join(out_path, name_prefix + '.json'), "w"))

    return

def main():
    nlp_path = 'stanford-corenlp-full-2018-10-05'
    nlp = StanfordCoreNLP(nlp_path)

    for data_set in data_sets:
        data = read_ori(data_set)

        print("Processing data: %s"%data_set)

        name_prefix = data_set.replace('.txt', '')

        process(data, nlp, name_prefix)

        print("finished!")

    return

if __name__ == '__main__':
    main()
