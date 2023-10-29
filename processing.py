import re
import random
import numpy as np
import copy
import torch
import tqdm
from collections import defaultdict
from transformers import BertTokenizer

dis2idx = np.zeros((1000), dtype='int64')
dis2idx[1] = 1
dis2idx[2:] = 2
dis2idx[4:] = 3
dis2idx[8:] = 4
dis2idx[16:] = 5
dis2idx[32:] = 6
dis2idx[64:] = 7
dis2idx[128:] = 8
dis2idx[256:] = 9

mention_edge2id = {
    'blank': 0,
    'self': 1,
    'intra-coref': 2,
    'intra-relate': 3,
    'inter-coref': 4,
    'inter-relate': 5
}

entity_node2id = {
    'blank': 0,
    'na': 1,
    'start': 2,
    'end': 3,
    'bridge': 4,
}

class Process():
    def __init__(self):

        self.tokenizer = BertTokenizer.from_pretrained("scibert_scivocab_uncased")

        return

    def process_train_data(self, train_data, config):
        print("start process train data...")

        batch_data = []

        train_data_size = len(train_data)
        train_order = list(range(train_data_size))

        random.shuffle(train_order)
        batch_num = train_data_size // config.batch_size

        if train_data_size % batch_num != 0:
            batch_num = batch_num + 1

        batch_index = [i for i in range(batch_num)]

        for i in tqdm.tqdm(batch_index):
            start_index = i * config.batch_size
            cur_index = min(config.batch_size, train_data_size - start_index)
            cur_order = list(train_order[start_index: start_index + cur_index])

            input_ids = np.zeros((cur_index, config.max_tokens), dtype=np.int32)
            pos_ids = np.zeros((cur_index, config.max_tokens), dtype=np.int32)

            pos = np.zeros((cur_index, config.max_tokens), dtype=np.int32)
            ner = np.zeros((cur_index, config.max_tokens), dtype=np.int32)

            w2m_mapping = np.zeros((cur_index, config.max_mention_num, config.max_tokens), dtype=np.bool)

            mention_bias_mat = np.full((cur_index, config.max_mention_num, config.max_mention_num),
                                       fill_value=-9e10)
            mention_edge_mat = np.zeros((cur_index, config.max_mention_num, config.max_mention_num), dtype=np.int32)

            m2e_mapping = np.zeros((cur_index, config.max_entity_num, config.max_mention_num), dtype=np.bool)

            path_bias_mat = np.full((cur_index, config.max_mention_num, config.max_mention_num, config.max_mention_num),
                                    fill_value=-9e10)
            path_edge_mat = np.zeros((cur_index, config.max_mention_num, config.max_mention_num, config.max_mention_num),
                                     dtype=np.int32)

            ht_pos = np.zeros((cur_index, config.max_mention_num, config.max_mention_num), dtype=np.int32)

            relation_multi_label = np.zeros((cur_index, config.max_entity_num, config.max_entity_num, config.relation_num), dtype=np.int32)
            relation_mask = np.zeros((cur_index, config.max_entity_num, config.max_entity_num), dtype=np.bool)

            max_token = 0
            max_mention_num = 0
            max_entity_num = 0
            max_ht_mun = 0

            for k, index in enumerate(cur_order):
                ins = train_data[index]

                words = []
                for sent in ins['sents']:
                    words += sent

                dl = 0
                word_dl = []
                p_id = 0

                _input_ids = []
                _pos_ids = []

                for w in words:
                    tokens = self.tokenizer.encode(w, add_special_tokens=False)
                    _input_ids += tokens
                    _pos_ids += [p_id] * len(tokens)
                    word_dl.append([x + dl for x in range(len(tokens))])

                    dl += len(tokens)

                    if len(tokens) > 0:
                        p_id += 1

                    if p_id >= config.max_len:
                        break

                _input_ids = [self.tokenizer.cls_token_id] + _input_ids + [self.tokenizer.sep_token_id]
                _pos_ids = [0] + _pos_ids + [_pos_ids[-1]]

                max_token = max(max_token, len(_input_ids))

                _input_ids = _input_ids + [0 for _ in range(config.max_tokens-len(_input_ids))]
                _pos_ids = _pos_ids + [0 for _ in range(config.max_tokens-len(_pos_ids))]

                input_ids[k, ] = _input_ids
                pos_ids[k, ] = _pos_ids

                cur_mention_bias_mat = np.full((config.max_mention_num, config.max_mention_num), fill_value=-9e10)
                cur_mention_edge_mat = np.zeros((config.max_mention_num, config.max_mention_num), dtype=np.int32)

                cur_mention_bias_mat, cur_mention_edge_mat = create_mention_graph(ins, cur_mention_bias_mat,
                                                                                  cur_mention_edge_mat)
                mention_bias_mat[k,] = cur_mention_bias_mat
                mention_edge_mat[k,] = cur_mention_edge_mat

                mention_num = 0
                entity_num = 0

                mention_to_entity = dict()

                vertexSet = copy.deepcopy(ins['vertexSet'])
                for idx, vertex in enumerate(vertexSet, 1):

                    start_entity = mention_num
                    end_entity = mention_num + len(vertex)

                    m2e_mapping[k, entity_num, start_entity: end_entity] = 1

                    mention_to_entity[entity_num] = [mention_num, mention_num + len(vertex)]

                    entity_num = entity_num + 1

                    for v in vertex:
                        # context encode layer
                        start_index, end_index = v['pos'][0], v['pos'][1]

                        if start_index > config.max_len or end_index > config.max_len:
                            mention_num = mention_num + 1
                            continue

                        start_index = word_dl[start_index][0] if len(word_dl[start_index]) else word_dl[start_index + 1][0]
                        end_index = word_dl[end_index - 1][-1] + 1 if len(word_dl[end_index - 1]) else word_dl[end_index - 2][-1] + 1

                        pos[k, start_index:end_index] = idx
                        ner[k, start_index:end_index] = config.ner2id[v['type']]

                        # w2m mapping layer
                        w2m_mapping[k, mention_num, start_index:end_index] = 1

                        mention_num = mention_num + 1

                max_mention_num = max(max_mention_num, mention_num)
                max_entity_num = max(max_entity_num, entity_num)

                d_index = 0
                for vertex in vertexSet:
                    for v in vertex:
                        ht_pos[k, d_index, :mention_num] += v['pos'][0]
                        ht_pos[k, :mention_num, d_index] -= v['pos'][0]
                        d_index = d_index + 1

                cur_p_bias_mat = np.full((config.max_mention_num, config.max_mention_num, config.max_mention_num),
                                         fill_value=-9e10)
                cur_p_edge_mat = np.zeros((config.max_mention_num, config.max_mention_num, config.max_mention_num),
                                          dtype=np.int32)

                cur_p_bias_mat, cur_p_edge_mat = create_e2e_graph(ins, cur_p_bias_mat, cur_p_edge_mat, entity_num, mention_to_entity)

                path_bias_mat[k,] = cur_p_bias_mat
                path_edge_mat[k,] = cur_p_edge_mat

                for m_1 in range(mention_num):
                    for m_2 in range(mention_num):
                        if ht_pos[k, m_1, m_2] < 0:
                            ht_pos[k, m_1, m_2] = dis2idx[-ht_pos[k, m_1, m_2]] + 9
                        else:
                            ht_pos[k, m_1, m_2] = dis2idx[ht_pos[k, m_1, m_2]]

                ht_pos[k, ht_pos[k] == 0] = 19

                all_labels = ins['labels']

                relation_multi_label[k, ..., 0] = 1

                for triple in all_labels:
                    h = triple['h']
                    t = triple['t']
                    r = triple['r']

                    relation_multi_label[k, h, t, 0] = 0
                    relation_multi_label[k, h, t, r] = 1

                j = 0
                for h in range(entity_num):
                    for t in range(entity_num):

                        relation_mask[k, h, t] = 1

                        if h != t:

                            j = j + 1

                max_ht_mun = max(max_ht_mun, j)

            batch_data.append([
                torch.LongTensor(input_ids[:cur_index, :max_token]),
                torch.LongTensor(pos_ids[:cur_index, :max_token]),
                torch.LongTensor(pos[:cur_index, :max_token-2]),
                torch.LongTensor(ner[:cur_index, :max_token-2]),
                torch.BoolTensor(w2m_mapping[:cur_index, :max_mention_num, :max_token-2]),
                torch.FloatTensor(mention_bias_mat[:cur_index, :max_mention_num, :max_mention_num]),
                torch.LongTensor(mention_edge_mat[:cur_index, :max_mention_num, :max_mention_num]),
                torch.BoolTensor(m2e_mapping[:cur_index, :max_entity_num, :max_mention_num]),
                torch.FloatTensor(path_bias_mat[:cur_index, :max_mention_num, :max_mention_num, :max_mention_num]),
                torch.LongTensor(path_edge_mat[:cur_index, :max_mention_num, :max_mention_num, :max_mention_num]),
                torch.LongTensor(ht_pos[:cur_index, :max_mention_num, :max_mention_num]),
                torch.FloatTensor(relation_multi_label[:cur_index, :max_entity_num, :max_entity_num, :]),
                torch.BoolTensor(relation_mask[:cur_index, :max_entity_num, :max_entity_num]),
            ])

        print("finish!")

        return batch_data


    def process_dev_data(self, dev_data, config):
        print("start process dev data...")
        batch_data = []

        test_data_size = len(dev_data)
        test_order = list(range(test_data_size))

        batch_num = test_data_size // config.batch_size

        if test_data_size % batch_num != 0:
            batch_num = batch_num + 1

        batch_index = [i for i in range(batch_num)]

        two_men_num, three_men_num, four_men_num, five_men_num = 0, 0, 0, 0
        con_num, mp_num, non_num = 0, 0, 0
        zero_dis_num, one_dis_num, two_dis_num, three_dis_num = 0, 0, 0, 0

        for i in tqdm.tqdm(batch_index):
            start_index = i * config.batch_size
            cur_index = min(config.batch_size, test_data_size-start_index)
            cur_order = list(test_order[start_index: start_index+cur_index])

            input_ids = np.zeros((cur_index, config.max_tokens), dtype=np.int32)
            pos_ids = np.zeros((cur_index, config.max_tokens), dtype=np.int32)

            pos = np.zeros((cur_index, config.max_tokens), dtype=np.int32)
            ner = np.zeros((cur_index, config.max_tokens), dtype=np.int32)

            w2m_mapping = np.zeros((cur_index, config.max_mention_num, config.max_tokens), dtype=np.bool)

            mention_bias_mat = np.full((cur_index, config.max_mention_num, config.max_mention_num), fill_value=-9e10)
            mention_edge_mat = np.zeros((cur_index, config.max_mention_num, config.max_mention_num), dtype=np.int32)

            m2e_mapping = np.zeros((cur_index, config.max_entity_num, config.max_mention_num), dtype=np.bool)

            path_bias_mat = np.full((cur_index, config.max_mention_num, config.max_mention_num, config.max_mention_num),
                                    fill_value=-9e10)
            path_edge_mat = np.zeros((cur_index, config.max_mention_num, config.max_mention_num, config.max_mention_num),
                                     dtype=np.int32)

            ht_pos = np.zeros((cur_index, config.max_mention_num, config.max_mention_num), dtype=np.int32)

            intra_mask = np.zeros((cur_index, config.max_entity_num, config.max_entity_num, config.relation_num), dtype=np.bool)
            inter_mask = np.ones((cur_index, config.max_entity_num, config.max_entity_num, config.relation_num), dtype=np.bool)
            intrain_mask = np.zeros((cur_index, config.max_entity_num, config.max_entity_num, config.relation_num), dtype=np.bool)

            two_men_mask = np.zeros((cur_index, config.max_entity_num, config.max_entity_num, config.relation_num), dtype=np.bool)
            three_men_mask = np.zeros((cur_index, config.max_entity_num, config.max_entity_num, config.relation_num), dtype=np.bool)
            four_men_mask = np.zeros((cur_index, config.max_entity_num, config.max_entity_num, config.relation_num), dtype=np.bool)
            five_men_mask = np.zeros((cur_index, config.max_entity_num, config.max_entity_num, config.relation_num), dtype=np.bool)

            con_mask = np.zeros((cur_index, config.max_entity_num, config.max_entity_num, config.relation_num), dtype=np.bool)
            mp_mask = np.zeros((cur_index, config.max_entity_num, config.max_entity_num, config.relation_num), dtype=np.bool)
            non_mask = np.zeros((cur_index, config.max_entity_num, config.max_entity_num, config.relation_num), dtype=np.bool)

            zero_dis_mask = np.zeros((cur_index, config.max_entity_num, config.max_entity_num, config.relation_num), dtype=np.bool)
            one_dis_mask = np.zeros((cur_index, config.max_entity_num, config.max_entity_num, config.relation_num),
                                     dtype=np.bool)
            two_dis_mask = np.zeros((cur_index, config.max_entity_num, config.max_entity_num, config.relation_num),
                                     dtype=np.bool)
            three_dis_mask = np.zeros((cur_index, config.max_entity_num, config.max_entity_num, config.relation_num),
                                     dtype=np.bool)

            relation_multi_label = np.zeros((cur_index, config.max_entity_num, config.max_entity_num, config.relation_num), dtype=np.float32)
            relation_mask = np.zeros((cur_index, config.max_entity_num, config.max_entity_num), dtype=np.bool)
            predict_mask = np.zeros((cur_index, config.max_entity_num, config.max_entity_num), dtype=np.bool)

            max_token = 0
            max_mention_num = 0
            max_entity_num = 0
            max_ht_mun = 0

            labels = []

            L_vertex = []
            titles = []
            indexes = []

            for k, index in enumerate(cur_order):
                ins = dev_data[index]

                words = []
                for sent in ins['sents']:
                    words += sent

                dl = 0
                word_dl = []
                p_id = 0

                _input_ids = []
                _pos_ids = []

                for w in words:
                    tokens = self.tokenizer.encode(w, add_special_tokens=False)
                    _input_ids += tokens
                    _pos_ids += [p_id] * len(tokens)
                    word_dl.append([x + dl for x in range(len(tokens))])

                    dl += len(tokens)

                    if len(tokens) > 0:
                        p_id += 1

                    if p_id >= config.max_len:
                        break

                _input_ids = [self.tokenizer.cls_token_id] + _input_ids + [self.tokenizer.sep_token_id]
                _pos_ids = [0] + _pos_ids + [_pos_ids[-1]]

                max_token = max(max_token, len(_input_ids))

                _input_ids = _input_ids + [0 for _ in range(config.max_tokens - len(_input_ids))]
                _pos_ids = _pos_ids + [0 for _ in range(config.max_tokens - len(_pos_ids))]

                input_ids[k,] = _input_ids
                pos_ids[k,] = _pos_ids

                cur_mention_bias_mat = np.full((config.max_mention_num, config.max_mention_num), fill_value=-9e10)
                cur_mention_edge_mat = np.zeros((config.max_mention_num, config.max_mention_num), dtype=np.int32)

                cur_mention_bias_mat ,cur_mention_edge_mat = create_mention_graph(ins, cur_mention_bias_mat ,cur_mention_edge_mat)
                mention_bias_mat[k, ] = cur_mention_bias_mat
                mention_edge_mat[k, ] = cur_mention_edge_mat

                mention_num = 0
                entity_num = 0

                mention_to_entity = dict()

                vertexSet = copy.deepcopy(ins['vertexSet'])
                for idx, vertex in enumerate(vertexSet, 1):
                    start_entity = mention_num
                    end_entity = mention_num + len(vertex)

                    m2e_mapping[k, entity_num, start_entity: end_entity] = 1

                    mention_to_entity[entity_num] = [mention_num, mention_num + len(vertex)]

                    entity_num = entity_num + 1

                    for v in vertex:
                        # context encode layer
                        start_index, end_index = v['pos'][0], v['pos'][1]

                        if start_index > config.max_len or end_index > config.max_len:
                            mention_num = mention_num + 1
                            continue

                        start_index = word_dl[start_index][0] if len(word_dl[start_index]) else \
                        word_dl[start_index + 1][0]
                        end_index = word_dl[end_index - 1][-1] + 1 if len(word_dl[end_index - 1]) else \
                        word_dl[end_index - 2][-1] + 1

                        pos[k, start_index:end_index] = idx
                        ner[k, start_index:end_index] = config.ner2id[v['type']]

                        # w2m mapping layer
                        w2m_mapping[k, mention_num, start_index:end_index] = 1

                        mention_num = mention_num + 1

                max_mention_num = max(max_mention_num, mention_num)
                max_entity_num = max(max_entity_num, entity_num)

                d_index = 0
                for vertex in vertexSet:
                    for v in vertex:
                        ht_pos[k, d_index, :mention_num] += v['pos'][0]
                        ht_pos[k, :mention_num, d_index] -= v['pos'][0]
                        d_index = d_index + 1

                cur_p_bias_mat = np.full((config.max_mention_num, config.max_mention_num, config.max_mention_num),
                                         fill_value=-9e10)
                cur_p_edge_mat = np.zeros((config.max_mention_num, config.max_mention_num, config.max_mention_num),
                                          dtype=np.int32)

                cur_p_bias_mat, cur_p_edge_mat = create_e2e_graph(ins, cur_p_bias_mat, cur_p_edge_mat, entity_num, mention_to_entity)

                path_bias_mat[k,] = cur_p_bias_mat
                path_edge_mat[k,] = cur_p_edge_mat

                for m_1 in range(mention_num):
                    for m_2 in range(mention_num):
                        if ht_pos[k, m_1, m_2] < 0:
                            ht_pos[k, m_1, m_2] = dis2idx[-ht_pos[k, m_1, m_2]] + 9
                        else:
                            ht_pos[k, m_1, m_2] = dis2idx[ht_pos[k, m_1, m_2]]

                ht_pos[k, ht_pos[k] == 0] = 19

                all_labels = ins['labels']

                relation_multi_label[k, ..., 0] = 1

                for triple in all_labels:
                    h = triple['h']
                    t = triple['t']
                    r = triple['r']

                    relation_multi_label[k, h, t, 0] = 0
                    relation_multi_label[k, h, t, r] = 1

                j = 0

                con_path, two_path, three_path = get_entity_type(ins)
                for h_idx in range(entity_num):
                    for t_idx in range(entity_num):

                        if (vertexSet[h_idx][0]['type'] == 'Chemical') and (vertexSet[t_idx][0]['type'] == 'Disease'):
                            predict_mask[k, h_idx, t_idx] = 1
                        
                        relation_mask[k, h_idx, t_idx] = 1
                        
                        if h_idx != t_idx:

                            hlist = ins['vertexSet'][h_idx]
                            tlist = ins['vertexSet'][t_idx]

                            for men_h in hlist:
                                for men_t in tlist:
                                    if men_h['sent_id'] == men_t['sent_id']:
                                        intra_mask[k, h_idx, t_idx] = True
                                        inter_mask[k, h_idx, t_idx] = False

                            j = j + 1

                            if (len(vertexSet[h_idx]) + len(vertexSet[t_idx])) == 2:
                                two_men_mask[k, h_idx, t_idx] = True
                                two_men_num += 1
                            elif (len(vertexSet[h_idx]) + len(vertexSet[t_idx])) == 3:
                                three_men_mask[k, h_idx, t_idx] = True
                                three_men_num += 1
                            elif (len(vertexSet[h_idx]) + len(vertexSet[t_idx])) == 4:
                                four_men_mask[k, h_idx, t_idx] = True
                                four_men_num += 1
                            else:
                                five_men_mask[k, h_idx, t_idx] = True
                                five_men_num += 1

                            if t_idx in con_path[h_idx]:
                                con_mask[k, h_idx, t_idx] = True
                                con_num += 1
                            elif (t_idx in two_path[h_idx]) or (t_idx in three_path[h_idx]):
                                mp_mask[k, h_idx, t_idx] = True
                                mp_num += 1
                            else:
                                non_mask[k, h_idx, t_idx] = True
                                non_num += 1

                            min_h_t_dis = 1000
                            for men_h in hlist:
                                for men_t in tlist:
                                    h_t_dis = abs(men_t['sent_id'] - men_h['sent_id'])
                                    min_h_t_dis = min(min_h_t_dis, h_t_dis)

                            if min_h_t_dis == 0:
                                zero_dis_mask[k, h_idx, t_idx] = True
                                zero_dis_num += 1
                            elif min_h_t_dis == 1:
                                one_dis_mask[k, h_idx, t_idx] = True
                                one_dis_num += 1
                            elif min_h_t_dis == 2:
                                two_dis_mask[k, h_idx, t_idx] = True
                                two_dis_num += 1
                            else:
                                three_dis_mask[k, h_idx, t_idx] = True
                                three_dis_num += 1

                max_ht_mun = max(max_ht_mun, j)

                label_set = {}
                for label in ins['labels']:
                    label_set[(label['h'], label['t'], label['r'])] = label['in'+'dev_train']

                labels.append(label_set)

                L_vertex.append(entity_num)

                title = ins['title']
                titles.append(title)
                indexes.append(index)

            batch_data.append([
                torch.LongTensor(input_ids[:cur_index, :max_token]),
                torch.LongTensor(pos_ids[:cur_index, :max_token]),
                torch.LongTensor(pos[:cur_index, :max_token-2]),
                torch.LongTensor(ner[:cur_index, :max_token-2]),
                torch.BoolTensor(w2m_mapping[:cur_index, :max_mention_num, :max_token-2]),
                torch.FloatTensor(mention_bias_mat[:cur_index, :max_mention_num, :max_mention_num]),
                torch.LongTensor(mention_edge_mat[:cur_index, :max_mention_num, :max_mention_num]),
                torch.BoolTensor(m2e_mapping[:cur_index, :max_entity_num, :max_mention_num]),
                torch.FloatTensor(path_bias_mat[:cur_index, :max_mention_num, :max_mention_num, :max_mention_num]),
                torch.LongTensor(path_edge_mat[:cur_index, :max_mention_num, :max_mention_num, :max_mention_num]),
                torch.LongTensor(ht_pos[:cur_index, :max_mention_num, :max_mention_num]),
                torch.BoolTensor(intra_mask[:cur_index, :max_entity_num, :max_entity_num, :]),
                torch.BoolTensor(inter_mask[:cur_index, :max_entity_num, :max_entity_num, :]),
                torch.BoolTensor(intrain_mask[:cur_index, :max_entity_num, :max_entity_num, :]),
                torch.BoolTensor(two_men_mask[:cur_index, :max_entity_num, :max_entity_num, :]),
                torch.BoolTensor(three_men_mask[:cur_index, :max_entity_num, :max_entity_num, :]),
                torch.BoolTensor(four_men_mask[:cur_index, :max_entity_num, :max_entity_num, :]),
                torch.BoolTensor(five_men_mask[:cur_index, :max_entity_num, :max_entity_num, :]),
                torch.BoolTensor(con_mask[:cur_index, :max_entity_num, :max_entity_num, :]),
                torch.BoolTensor(mp_mask[:cur_index, :max_entity_num, :max_entity_num, :]),
                torch.BoolTensor(non_mask[:cur_index, :max_entity_num, :max_entity_num, :]),
                torch.BoolTensor(zero_dis_mask[:cur_index, :max_entity_num, :max_entity_num, :]),
                torch.BoolTensor(one_dis_mask[:cur_index, :max_entity_num, :max_entity_num, :]),
                torch.BoolTensor(two_dis_mask[:cur_index, :max_entity_num, :max_entity_num, :]),
                torch.BoolTensor(three_dis_mask[:cur_index, :max_entity_num, :max_entity_num, :]),
                torch.FloatTensor(relation_multi_label[:cur_index, :max_entity_num, :max_entity_num, :]),
                torch.BoolTensor(relation_mask[:cur_index, :max_entity_num, :max_entity_num]),
                torch.BoolTensor(predict_mask[:cur_index, :max_entity_num, :max_entity_num]),
                labels,
                L_vertex,
                titles,
                indexes,
            ])

        print(two_men_num)
        print(three_men_num)
        print(four_men_num)
        print(five_men_num)
        print()
        print(con_num)
        print(mp_num)
        print(non_num)
        print()
        print(zero_dis_num)
        print(one_dis_num)
        print(two_dis_num)
        print(three_dis_num)
        print()

        return batch_data


    def process_test_data(self, test_data, config):
        print("start process test data...")

        batch_data = []

        train_data_size = len(test_data)
        train_order = list(range(train_data_size))

        random.shuffle(train_order)
        batch_num = train_data_size // config.batch_size

        if train_data_size % batch_num != 0:
            batch_num = batch_num + 1

        batch_index = [i for i in range(batch_num)]

        for i in tqdm.tqdm(batch_index):
            start_index = i * config.batch_size
            cur_index = min(config.batch_size, train_data_size - start_index)
            cur_order = list(train_order[start_index: start_index + cur_index])

            input_ids = np.zeros((cur_index, config.max_tokens), dtype=np.int32)
            pos_ids = np.zeros((cur_index, config.max_tokens), dtype=np.int32)

            pos = np.zeros((cur_index, config.max_tokens), dtype=np.int32)
            ner = np.zeros((cur_index, config.max_tokens), dtype=np.int32)

            w2m_mapping = np.zeros((cur_index, config.max_mention_num, config.max_tokens), dtype=np.bool)

            mention_bias_mat = np.full((cur_index, config.max_mention_num, config.max_mention_num),
                                       fill_value=-9e10)
            mention_edge_mat = np.zeros((cur_index, config.max_mention_num, config.max_mention_num), dtype=np.int32)

            m2e_mapping = np.zeros((cur_index, config.max_entity_num, config.max_mention_num), dtype=np.bool)

            path_bias_mat = np.full((cur_index, config.max_mention_num, config.max_mention_num, config.max_mention_num),
                                    fill_value=-9e10)
            path_edge_mat = np.zeros((cur_index, config.max_mention_num, config.max_mention_num, config.max_mention_num),
                                     dtype=np.int32)

            ht_pos = np.zeros((cur_index, config.max_mention_num, config.max_mention_num), dtype=np.int32)

            max_token = 0
            max_mention_num = 0
            max_entity_num = 0
            max_ht_mun = 0

            L_vertex = []
            titles = []

            for k, index in enumerate(cur_order):
                ins = test_data[index]

                words = []
                for sent in ins['sents']:
                    words += sent

                dl = 0
                word_dl = []
                p_id = 0

                _input_ids = []
                _pos_ids = []

                for w in words:
                    tokens = self.tokenizer.encode(w, add_special_tokens=False)
                    _input_ids += tokens
                    _pos_ids += [p_id] * len(tokens)
                    word_dl.append([x + dl for x in range(len(tokens))])

                    dl += len(tokens)

                    if len(tokens) > 0:
                        p_id += 1

                    if p_id >= config.max_len:
                        break

                _input_ids = [self.tokenizer.cls_token_id] + _input_ids + [self.tokenizer.sep_token_id]
                _pos_ids = [0] + _pos_ids + [_pos_ids[-1]]

                max_token = max(max_token, len(_input_ids))

                _input_ids = _input_ids + [0 for _ in range(config.max_tokens - len(_input_ids))]
                _pos_ids = _pos_ids + [0 for _ in range(config.max_tokens - len(_pos_ids))]

                input_ids[k,] = _input_ids
                pos_ids[k,] = _pos_ids

                cur_mention_bias_mat = np.full((config.max_mention_num, config.max_mention_num), fill_value=-9e10)
                cur_mention_edge_mat = np.zeros((config.max_mention_num, config.max_mention_num), dtype=np.int32)

                cur_mention_bias_mat, cur_mention_edge_mat = create_mention_graph(ins, cur_mention_bias_mat,
                                                                                  cur_mention_edge_mat)
                mention_bias_mat[k,] = cur_mention_bias_mat
                mention_edge_mat[k,] = cur_mention_edge_mat

                mention_num = 0
                entity_num = 0

                mention_to_entity = dict()

                vertexSet = copy.deepcopy(ins['vertexSet'])
                for idx, vertex in enumerate(vertexSet, 1):

                    start_entity = mention_num
                    end_entity = mention_num + len(vertex)

                    m2e_mapping[k, entity_num, start_entity: end_entity] = 1

                    mention_to_entity[entity_num] = [mention_num, mention_num + len(vertex)]

                    entity_num = entity_num + 1

                    for v in vertex:
                        # context encode layer
                        start_index, end_index = v['pos'][0], v['pos'][1]

                        if start_index > config.max_len or end_index > config.max_len:
                            mention_num = mention_num + 1
                            continue

                        start_index = word_dl[start_index][0] if len(word_dl[start_index]) else \
                        word_dl[start_index + 1][0]
                        end_index = word_dl[end_index - 1][-1] + 1 if len(word_dl[end_index - 1]) else \
                        word_dl[end_index - 2][-1] + 1

                        pos[k, start_index:end_index] = idx
                        ner[k, start_index:end_index] = config.ner2id[v['type']]

                        # w2m mapping layer
                        w2m_mapping[k, mention_num, start_index:end_index] = 1

                        mention_num = mention_num + 1

                max_mention_num = max(max_mention_num, mention_num)
                max_entity_num = max(max_entity_num, entity_num)

                d_index = 0
                for vertex in vertexSet:
                    for v in vertex:
                        ht_pos[k, d_index, :mention_num] += v['pos'][0]
                        ht_pos[k, :mention_num, d_index] -= v['pos'][0]
                        d_index = d_index + 1

                cur_p_bias_mat = np.full((config.max_mention_num, config.max_mention_num, config.max_mention_num),
                                         fill_value=-9e10)
                cur_p_edge_mat = np.zeros((config.max_mention_num, config.max_mention_num, config.max_mention_num),
                                          dtype=np.int32)

                cur_p_bias_mat, cur_p_edge_mat = create_e2e_graph(ins, cur_p_bias_mat, cur_p_edge_mat, entity_num, mention_to_entity)

                path_bias_mat[k,] = cur_p_bias_mat
                path_edge_mat[k,] = cur_p_edge_mat

                for m_1 in range(mention_num):
                    for m_2 in range(mention_num):
                        if ht_pos[k, m_1, m_2] < 0:
                            ht_pos[k, m_1, m_2] = dis2idx[-ht_pos[k, m_1, m_2]] + 9
                        else:
                            ht_pos[k, m_1, m_2] = dis2idx[ht_pos[k, m_1, m_2]]

                ht_pos[k, ht_pos[k] == 0] = 19

                j = 0
                for h in range(entity_num):
                    for t in range(entity_num):
                        if h != t:

                            j = j + 1

                max_ht_mun = max(max_ht_mun, j)

                L_vertex.append(entity_num)

                title = ins['title']
                titles.append(title)

            batch_data.append([
                torch.LongTensor(input_ids[:cur_index, :max_token]),
                torch.LongTensor(pos_ids[:cur_index, :max_token]),
                torch.LongTensor(pos[:cur_index, :max_token-2]),
                torch.LongTensor(ner[:cur_index, :max_token-2]),
                torch.BoolTensor(w2m_mapping[:cur_index, :max_mention_num, :max_token-2]),
                torch.FloatTensor(mention_bias_mat[:cur_index, :max_mention_num, :max_mention_num]),
                torch.LongTensor(mention_edge_mat[:cur_index, :max_mention_num, :max_mention_num]),
                torch.BoolTensor(m2e_mapping[:cur_index, :max_entity_num, :max_mention_num]),
                torch.FloatTensor(path_bias_mat[:cur_index, :max_mention_num, :max_mention_num, :max_mention_num]),
                torch.LongTensor(path_edge_mat[:cur_index, :max_mention_num, :max_mention_num, :max_mention_num]),
                torch.LongTensor(ht_pos[:cur_index, :max_mention_num, :max_mention_num]),
                L_vertex,
                titles,
            ])

        print("finish!")

        return batch_data


def create_mention_graph(ins, cur_m_bias_mat, cur_m_edge_mat):
    vertexSet = ins['vertexSet']

    co_occurence = dict()
    corefence = dict()

    men_in_sent = dict()

    mention_index = 0
    for vertex in vertexSet:
        men_h = mention_index
        for v_id, v in enumerate(vertex):
            sent_id = v['sent_id']
            men_in_sent[mention_index] = int(sent_id)

            men_t = mention_index + 1
            mention_index += 1

            if v_id >= (len(vertex)-1):
                break

            corefence[(men_h, men_t)] = True
            corefence[(men_t, men_h)] = True

    for m1 in range(mention_index):
        for m2 in range(mention_index):
            if men_in_sent[m1] == men_in_sent[m2]:
                co_occurence[(m1, m2)] = True

    for m1 in range(mention_index):
        for m2 in range(mention_index):
            if m1 == m2:
                cur_m_edge_mat[m1, m2] = mention_edge2id['self']
                cur_m_bias_mat[m1][m2] = 0.0
                continue

            if ((m1, m2) in co_occurence) and ((m1, m2) in corefence):
                cur_m_edge_mat[m1, m2] = mention_edge2id['intra-coref']
                cur_m_edge_mat[m2, m1] = mention_edge2id['intra-coref']

                cur_m_bias_mat[m1][m2] = 0.0
                cur_m_bias_mat[m2][m1] = 0.0
            elif ((m1, m2) in co_occurence) and ((m1, m2) not in corefence):
                cur_m_edge_mat[m1, m2] = mention_edge2id['intra-relate']
                cur_m_edge_mat[m2, m1] = mention_edge2id['intra-relate']

                cur_m_bias_mat[m1][m2] = 0.0
                cur_m_bias_mat[m2][m1] = 0.0
            elif ((m1, m2) not in co_occurence) and ((m1, m2) in corefence):
                cur_m_edge_mat[m1, m2] = mention_edge2id['inter-coref']
                cur_m_edge_mat[m2, m1] = mention_edge2id['inter-coref']

                cur_m_bias_mat[m1][m2] = 0.0
                cur_m_bias_mat[m2][m1] = 0.0
            elif ((m1, m2) not in co_occurence) and ((m1, m2) not in corefence):
                cur_m_edge_mat[m1, m2] = mention_edge2id['inter-relate']
                cur_m_edge_mat[m2, m1] = mention_edge2id['inter-relate']

                cur_m_bias_mat[m1][m2] = 0.0
                cur_m_bias_mat[m2][m1] = 0.0

    return cur_m_bias_mat, cur_m_edge_mat

def create_e2e_graph(ins, cur_p_bias_mat, cur_p_edge_mat, entity_num, men_to_entity):

    sents = ins['sents']

    nodes = [[] for _ in range(len(ins['sents']))]
    e2e_sent = defaultdict(dict)

    for ns_no, ns in enumerate(ins['vertexSet']):
        for n in ns:
            sent_id = int(n['sent_id'])
            nodes[sent_id].append(ns_no)

    for sent_id in range(len(sents)):
        for n1 in nodes[sent_id]:
            for n2 in nodes[sent_id]:
                if n1 == n2:
                    continue
                if n2 not in e2e_sent[n1]:
                    e2e_sent[n1][n2] = set()
                e2e_sent[n1][n2].add(sent_id)

    # 2-hop graph
    path_two = defaultdict(dict)
    entityNum = len(ins['vertexSet'])
    for n1 in range(entityNum):
        for n2 in range(entityNum):
            if n1 == n2:
                continue
            for n3 in range(entityNum):
                if n3 == n1 or n3 == n2:
                    continue
                if not (n3 in e2e_sent[n1] and n2 in e2e_sent[n3]):
                    continue

                for s1 in e2e_sent[n1][n3]:
                    for s2 in e2e_sent[n3][n2]:
                        if s1 == s2:
                            continue

                        if n2 not in path_two[n1]:
                            path_two[n1][n2] = []
                        cand_sents = [s1, s2]
                        cand_sents.sort()

                        path_two[n1][n2].append((cand_sents, n3))

    # 3-hop graph
    path_three = defaultdict(dict)
    for n1 in range(entityNum):
        for n2 in range(entityNum):
            if n1 == n2:
                continue
            for n3 in range(entityNum):
                if n3 == n1 or n3 == n2:
                    continue
                if n3 in e2e_sent[n1] and n2 in path_two[n3]:
                    for cand1 in e2e_sent[n1][n3]:
                        for cand2 in path_two[n3][n2]:
                            if cand1 in cand2[0]:
                                continue

                            if cand2[1] == n1:
                                continue

                            if n2 not in path_three[n1]:
                                path_three[n1][n2] = []
                            cand_sents = [cand1] + cand2[0]
                            cand_sents.sort()

                            path_three[n1][n2].append((cand_sents, [n3, cand2[1]]))

    # Consecutive graph
    consecutive = defaultdict(dict)
    for h in range(entityNum):
        for t in range(h + 1, entityNum):
            for n1 in ins['vertexSet'][h]:
                for n2 in ins['vertexSet'][t]:
                    gap = abs(n1['sent_id'] - n2['sent_id'])
                    if gap > 1:
                        continue

                    if t not in consecutive[h]:
                        consecutive[h][t] = []
                        consecutive[t][h] = []
                    if n1['sent_id'] < n2['sent_id']:
                        beg, end = n1['sent_id'], n2['sent_id']
                    else:
                        beg, end = n2['sent_id'], n1['sent_id']

                    consecutive[h][t].append([[i for i in range(beg, end + 1)]])
                    consecutive[t][h].append([[i for i in range(beg, end + 1)]])

    for e1 in range(entity_num):
        for e2 in range(entity_num):
            for e3 in range(entity_num):
                if e2 == e3 or e1 == e3:
                    cur_p_bias_mat[men_to_entity[e1][0]:men_to_entity[e1][1], men_to_entity[e2][0]:men_to_entity[e2][1], men_to_entity[e3][0]:men_to_entity[e3][1]] = 0.0


                    if e3 in consecutive[e2]:
                        cur_p_edge_mat[men_to_entity[e1][0]:men_to_entity[e1][1], men_to_entity[e2][0]:men_to_entity[e2][1], men_to_entity[e2][0]:men_to_entity[e2][1]] = \
                            entity_node2id['start']
                        cur_p_edge_mat[men_to_entity[e1][0]:men_to_entity[e1][1], men_to_entity[e2][0]:men_to_entity[e2][1], men_to_entity[e3][0]:men_to_entity[e3][1]] =\
                            entity_node2id['end']
                    elif e3 in path_two[e2]:
                        cur_p_edge_mat[men_to_entity[e1][0]:men_to_entity[e1][1], men_to_entity[e2][0]:men_to_entity[e2][1], men_to_entity[e2][0]:men_to_entity[e2][1]] =\
                            entity_node2id['start']

                        for path in path_two[e2][e3]:
                            bridge = path[1]
                            cur_p_edge_mat[men_to_entity[e1][0]:men_to_entity[e1][1], men_to_entity[e2][0]:men_to_entity[e2][1], men_to_entity[bridge][0]:men_to_entity[bridge][1]] = \
                                entity_node2id['bridge']
                            cur_p_bias_mat[men_to_entity[e1][0]:men_to_entity[e1][1], men_to_entity[e2][0]:men_to_entity[e2][1], men_to_entity[bridge][0]:men_to_entity[bridge][1]] = \
                                0.0
                        cur_p_edge_mat[men_to_entity[e1][0]:men_to_entity[e1][1], men_to_entity[e2][0]:men_to_entity[e2][1], men_to_entity[e3][0]:men_to_entity[e3][1]] = \
                            entity_node2id['end']
                    elif e3 in path_three[e2]:
                        cur_p_edge_mat[men_to_entity[e1][0]:men_to_entity[e1][1], men_to_entity[e2][0]:men_to_entity[e2][1], men_to_entity[e2][0]:men_to_entity[e2][1]] = \
                            entity_node2id['start']

                        for path in path_three[e2][e3]:
                            bridges = path[1]

                            for bridge in bridges:
                                cur_p_edge_mat[men_to_entity[e1][0]:men_to_entity[e1][1], men_to_entity[e2][0]:men_to_entity[e2][1], men_to_entity[bridge][0]:men_to_entity[bridge][1]] = \
                                    entity_node2id['bridge']
                                cur_p_bias_mat[men_to_entity[e1][0]:men_to_entity[e1][1], men_to_entity[e2][0]:men_to_entity[e2][1], men_to_entity[bridge][0]:men_to_entity[bridge][1]] = \
                                    0.0
                        cur_p_edge_mat[men_to_entity[e1][0]:men_to_entity[e1][1], men_to_entity[e2][0]:men_to_entity[e2][1], men_to_entity[e3][0]:men_to_entity[e3][1]] = \
                            entity_node2id['end']
                    else:
                        if e2 != e3:
                            cur_p_edge_mat[men_to_entity[e1][0]:men_to_entity[e1][1], men_to_entity[e2][0]:men_to_entity[e2][1], men_to_entity[e3][0]:men_to_entity[e3][1]] = \
                                entity_node2id['na']
                        else:
                            cur_p_edge_mat[men_to_entity[e1][0]:men_to_entity[e1][1], men_to_entity[e2][0]:men_to_entity[e2][1], men_to_entity[e3][0]:men_to_entity[e3][1]] = \
                                entity_node2id['start']

    return cur_p_bias_mat, cur_p_edge_mat

def get_entity_type(ins):
    sents = ins['sents']

    nodes = [[] for _ in range(len(ins['sents']))]
    e2e_sent = defaultdict(dict)

    for ns_no, ns in enumerate(ins['vertexSet']):
        for n in ns:
            sent_id = int(n['sent_id'])
            nodes[sent_id].append(ns_no)

    for sent_id in range(len(sents)):
        for n1 in nodes[sent_id]:
            for n2 in nodes[sent_id]:
                if n1 == n2:
                    continue
                if n2 not in e2e_sent[n1]:
                    e2e_sent[n1][n2] = set()
                e2e_sent[n1][n2].add(sent_id)

    # 2-hop graph
    path_two = defaultdict(dict)
    entityNum = len(ins['vertexSet'])
    for n1 in range(entityNum):
        for n2 in range(entityNum):
            if n1 == n2:
                continue
            for n3 in range(entityNum):
                if n3 == n1 or n3 == n2:
                    continue
                if not (n3 in e2e_sent[n1] and n2 in e2e_sent[n3]):
                    continue

                for s1 in e2e_sent[n1][n3]:
                    for s2 in e2e_sent[n3][n2]:
                        if s1 == s2:
                            continue

                        if n2 not in path_two[n1]:
                            path_two[n1][n2] = []
                        cand_sents = [s1, s2]
                        cand_sents.sort()

                        path_two[n1][n2].append((cand_sents, n3))

    # 3-hop graph
    path_three = defaultdict(dict)
    for n1 in range(entityNum):
        for n2 in range(entityNum):
            if n1 == n2:
                continue
            for n3 in range(entityNum):
                if n3 == n1 or n3 == n2:
                    continue
                if n3 in e2e_sent[n1] and n2 in path_two[n3]:
                    for cand1 in e2e_sent[n1][n3]:
                        for cand2 in path_two[n3][n2]:
                            if cand1 in cand2[0]:
                                continue

                            if cand2[1] == n1:
                                continue

                            if n2 not in path_three[n1]:
                                path_three[n1][n2] = []
                            cand_sents = [cand1] + cand2[0]
                            cand_sents.sort()

                            path_three[n1][n2].append((cand_sents, [n3, cand2[1]]))

    # Consecutive graph
    consecutive = defaultdict(dict)
    for h in range(entityNum):
        for t in range(h + 1, entityNum):
            for n1 in ins['vertexSet'][h]:
                for n2 in ins['vertexSet'][t]:
                    gap = abs(n1['sent_id'] - n2['sent_id'])
                    if gap > 1:
                        continue

                    if t not in consecutive[h]:
                        consecutive[h][t] = []
                        consecutive[t][h] = []
                    if n1['sent_id'] < n2['sent_id']:
                        beg, end = n1['sent_id'], n2['sent_id']
                    else:
                        beg, end = n2['sent_id'], n1['sent_id']

                    consecutive[h][t].append([[i for i in range(beg, end + 1)]])
                    consecutive[t][h].append([[i for i in range(beg, end + 1)]])

    return consecutive, path_two, path_three
