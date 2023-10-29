import re
import numpy as np
import json
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from sklearn.metrics import auc
import random
import time
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

from model import Config
from model import Model
import processing

data_fix = ['CDR_TrainingSet.PubTator.json', 'CDR_DevelopmentSet.PubTator.json', 'CDR_TestSet.PubTator.json']


class Accuracy(object):
    def __init__(self):
        self.correct = 0
        self.total = 0

    def add(self, is_correct):
        self.total += 1
        if is_correct:
            self.correct += 1

    def get(self):
        if self.total == 0:
            return 0.0
        else:
            return float(self.correct) / self.total

    def clear(self):
        self.correct = 0
        self.total = 0


def read_vec(origin_path):

    ner2id_path = origin_path + '/' + 'ner2id.json'

    ner2id = json.load(open(ner2id_path, 'r'))

    return ner2id


def read_data(origin_path):

    train_data_path = origin_path + '/' + data_fix[0]
    dev_data_path = origin_path + '/' + data_fix[1]
    test_data_path = origin_path + '/' + data_fix[2]

    train_data = json.load(open(train_data_path, 'r'))
    dev_data = json.load(open(dev_data_path, 'r'))
    test_data = json.load(open(test_data_path, 'r'))

    return train_data, dev_data, test_data

def view(data):
    max_mention_num = 0
    max_entity_num = 0

    label_num = 0

    for d in data:
        mention_num = 0
        entity_num = 0

        for vertex in d['vertexSet']:
            m_n = len(vertex)
            mention_num = mention_num + m_n

        e_n = len(d['vertexSet'])
        entity_num = entity_num + e_n

        max_mention_num = max(max_mention_num, mention_num)

        max_entity_num = max(max_entity_num, entity_num)

        l_n = len(d['labels'])
        label_num = label_num + l_n

    return max_mention_num, max_entity_num, label_num

def seed_everywhere(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.

def logging(s, print_=True):
    if print_:
        print(s)


def train(train_set, train_index, model, criterion, optimizer, scheduler):
    model.train()

    loss_list = []

    acc_na = Accuracy()
    acc_not_na = Accuracy()
    acc_total = Accuracy()

    acc_na.clear()
    acc_not_na.clear()
    acc_total.clear()

    for i in train_index:
        batch_data = train_set[i]

        if use_gpu:
            batch_data = [data.cuda() for data in batch_data]

        input_ids = batch_data[0]
        pos_ids = batch_data[1]
        pos = batch_data[2]
        ner = batch_data[3]
        w2m_mapping = batch_data[4]
        mention_bias_mat = batch_data[5]
        mention_edge_mat = batch_data[6]
        m2e_mapping = batch_data[7]
        path_bias_mat = batch_data[8]
        path_edge_mat = batch_data[9]
        ht_pos = batch_data[10]
        relation_multi_label = batch_data[11]
        relation_mask = batch_data[12]

        dis = ht_pos

        predict_re = model(
            input_ids=input_ids,
            pos_ids=pos_ids,
            ner=ner,
            pos=pos,
            w2m_mapping=w2m_mapping,
            m_bias_mat=mention_bias_mat,
            m_edge_mat=mention_edge_mat,
            p_bias_mat=path_bias_mat,
            p_edge_mat=path_edge_mat,
            m2e_mapping=m2e_mapping,
            dis=dis,
        )

        relation_mask[:, range(relation_mask.size(1)), range(relation_mask.size(2))] = 0

        loss = criterion(predict_re[relation_mask], relation_multi_label[relation_mask])

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        loss_list.append(loss.cpu().item())

        scheduler.step()

    mean_loss = np.mean(loss_list)

    print("Loss {:.5f}".format(mean_loss))

    return mean_loss


def devel(epoch, dev_set, model, criterion, optimizer):
    model.eval()

    loss_list = []

    test_result = []

    label_result = []

    intrain_list = []

    intra_list = []

    inter_list = []

    total_recall = 0
    intra_recall = 0
    inter_recall = 0

    with torch.no_grad():
        for i, batch_data in enumerate(dev_set):

            if use_gpu:
                batch_data = [data.cuda() for data in batch_data[: -4]]

            input_ids = batch_data[0]
            pos_ids = batch_data[1]
            pos = batch_data[2]
            ner = batch_data[3]
            w2m_mapping = batch_data[4]
            mention_bias_mat = batch_data[5]
            mention_edge_mat = batch_data[6]
            m2e_mapping = batch_data[7]
            path_bias_mat = batch_data[8]
            path_edge_mat = batch_data[9]
            ht_pos = batch_data[10]
            intra_mask = batch_data[11]
            inter_mask = batch_data[12]
            intrain_mask = batch_data[13]
            relation_multi_label = batch_data[25]
            relation_mask = batch_data[26]
            predict_mask = batch_data[27]

            dis = ht_pos

            outputs = model(
                input_ids=input_ids,
                pos_ids=pos_ids,
                ner=ner,
                pos=pos,
                w2m_mapping=w2m_mapping,
                m_bias_mat=mention_bias_mat,
                m_edge_mat=mention_edge_mat,
                p_bias_mat=path_bias_mat,
                p_edge_mat=path_edge_mat,
                m2e_mapping=m2e_mapping,
                dis=dis
            )
            outputs = torch.sigmoid(outputs)

            relation_mask[:, range(relation_mask.size(1)),range(relation_mask.size(2))] = 0

            loss = criterion(outputs[relation_mask], relation_multi_label[relation_mask])
            optimizer.zero_grad()

            loss_list.append(loss.cpu().item())

            b, N, _ = predict_mask.size()
            predict_mask = predict_mask.data.cpu().numpy()
            outputs = outputs.data.cpu().numpy()
            for b_i in range(b):
                for s_i in range(N):
                    for o_i in range(N):
                        if predict_mask[b_i][s_i][o_i] == False:
                            outputs[b_i][s_i][o_i] = [0.999, 0.001]

            outputs = torch.FloatTensor(outputs).to(input_ids.device)


            labels = relation_multi_label[..., 1:][relation_mask].contiguous().view(-1)
            outputs = outputs[..., 1:][relation_mask].contiguous().view(-1)
            intra_mask = intra_mask[..., 1:][relation_mask].contiguous().view(-1)
            inter_mask = inter_mask[..., 1:][relation_mask].contiguous().view(-1)
            intrain_mask = intrain_mask[..., 1:][relation_mask].contiguous().view(-1)

            label_result.append(labels)
            test_result.append(outputs)
            intrain_list.append(intrain_mask)
            intra_list.append(intra_mask)
            inter_list.append(inter_mask)
            total_recall += labels.sum().item()
            intra_recall += (intra_mask + labels).eq(2).sum().item()
            inter_recall += (inter_mask + labels).eq(2).sum().item()

    label_result = torch.cat(label_result)
    test_result = torch.cat(test_result)
    test_result, indices = torch.sort(test_result, descending=True)
    correct = np.cumsum(label_result[indices].cpu().numpy(), dtype=np.float)
    pr_x = correct / total_recall
    pr_y = correct / np.arange(1, len(correct) + 1)

    f1_arr = (2 * pr_x * pr_y / (pr_x + pr_y + 1e-20))

    auc_score = auc(x=pr_x, y=pr_y)

    intrain_list = torch.cat(intrain_list)
    intrain = np.cumsum(intrain_list[indices].cpu().numpy(), dtype=np.int)
    nt_pr_y = (correct - intrain)
    nt_pr_y[nt_pr_y != 0] /= (np.arange(1, len(correct) + 1) - intrain)[nt_pr_y != 0]

    nt_f1_arr = (2 * pr_x * nt_pr_y / (pr_x + nt_pr_y + 1e-20))
    nt_f1 = nt_f1_arr.max()
    nt_f1_pos = nt_f1_arr.argmax()
    theta = test_result[nt_f1_pos].cpu().item()

    intra_mask = torch.cat(intra_list)[indices][:nt_f1_pos]
    inter_mask = torch.cat(inter_list)[indices][:nt_f1_pos]

    intra_correct = label_result[indices][:nt_f1_pos][intra_mask].sum().item()
    inter_correct = label_result[indices][:nt_f1_pos][inter_mask].sum().item()

    intra_r = intra_correct / intra_recall
    intra_p = intra_correct / intra_mask.sum().item()
    intra_f1 = (2 * intra_p * intra_r) / (intra_p + intra_r)

    inter_r = inter_correct / inter_recall
    inter_p = inter_correct / inter_mask.sum().item()
    inter_f1 = (2 * inter_p * inter_r) / (inter_p + inter_r)

    print("*****************\n")
    print("ALL : Epoch: %s | NT F1: %s | F1: %s | Intra F1: %s | Inter F1: %s | Precision: %s | Recall: %s | AUC: %s | THETA: %s"%(
        str(epoch), str(nt_f1), str(f1_arr[nt_f1_pos]), str(intra_f1), str(inter_f1), str(pr_y[nt_f1_pos]), str(pr_x[nt_f1_pos]), str(auc_score), str(theta)
    ))

    mean_loss = np.mean(loss_list)

    return nt_f1, theta, mean_loss


def test(test_set, model, theta):
    model.eval()

    test_result = []
    with torch.no_grad():
        for i, batch_data in enumerate(test_set):
            L_vertex = batch_data[11]
            titles = batch_data[12]

            if use_gpu:
                batch_data = [data.cuda() for data in batch_data[: -2]]

            input_ids = batch_data[0]
            pos_ids = batch_data[1]
            pos = batch_data[2]
            ner = batch_data[3]
            w2m_mapping = batch_data[4]
            mention_bias_mat = batch_data[5]
            mention_edge_mat = batch_data[6]
            m2e_mapping = batch_data[7]
            path_bias_mat = batch_data[8]
            path_edge_mat = batch_data[9]
            ht_pos = batch_data[10]

            dis = ht_pos

            outputs = model(
                input_ids=input_ids,
                pos_ids=pos_ids,
                ner=ner,
                pos=pos,
                w2m_mapping=w2m_mapping,
                m_bias_mat=mention_bias_mat,
                m_edge_mat=mention_edge_mat,
                p_bias_mat=path_bias_mat,
                p_edge_mat=path_edge_mat,
                m2e_mapping=m2e_mapping,
                dis=dis
            )

            outputs = torch.sigmoid(outputs)
            outputs = outputs.cpu().numpy()

            for i in range(len(outputs)):
                L = L_vertex[i]

                k = 0
                for h_idx in range(L):
                    for t_idx in range(L):
                        if h_idx != t_idx:
                            for r in range(1, relation_num):
                                test_result.append((float(outputs[i, h_idx, t_idx, r]), titles[i], r,
                                                    h_idx, t_idx, r))

                            k = k + 1

    test_result.sort(key=lambda x: x[0], reverse=True)

    w = 0
    for i, item in enumerate(test_result):
        if item[0] > theta:
            w = i

    output = [{'h_idx': x[3], 't_idx': x[4], 'r_idx': x[-1], 'r': x[-4], 'title': x[1]}
              for x in test_result[:w + 1]]

    return output


def print_params(model):
    print('total parameters:', sum([np.prod(list(p.size())) for p in model.parameters()]))

    print('total trainable parameters:', sum([np.prod(list(p.size())) for p in model.parameters() if p.requires_grad]))

origin_path = 'prepro_data'
output_path = '../output'
save_model_path = '../output/model.pt'
max_len = 512
relation_num = 2
h_t_limit = 1800
batch_size = 4
epoches = 10
use_gpu = 1
device = 1
seed = 1698064017

def main():
    train_data, dev_data, test_data = read_data(origin_path=origin_path)
    data = train_data + dev_data + test_data

    print("seed: %s" % str(seed))
    seed_everywhere(seed)

    random.shuffle(data)

    train_data = data[: 1000]
    dev_data = data[1000: ]

    ner2id = read_vec(origin_path=origin_path)

    # max_mention_num, max_entity_num, label_num = view(train_data)
    # print("train set max mention num: %s; max entity num: %s, max label num: %s"%(str(max_mention_num), str(max_entity_num), str(label_num)))
    #
    # max_mention_num, max_entity_num, label_num = view(dev_data)
    # print("dev set max mention num: %s; max entity num: %s, max label num: %s" % (
    # str(max_mention_num), str(max_entity_num), str(label_num)))
    #
    # max_mention_num, max_entity_num, label_num = view(test_data)
    # print("test set max mention num: %s; max entity num: %s, max label num: %s" % (
    # str(max_mention_num), str(max_entity_num), str(label_num)))

    config = Config.config(
        max_len=512,
        relation_num=relation_num,
        h_t_limit=h_t_limit,
        batch_size=batch_size,
        use_gpu=use_gpu,
        ner2id=ner2id,
    )

    print('Processing Data...')
    process = processing.Process()

    test_set = process.process_dev_data(dev_data=dev_data, config=config)

    p_best_f1 = 0.0

    lrs = [1e-5]
    bert_lrs = [5e-5]

    BEST_F1 = 0.0
    BEST_LR = 0.0
    BEST_Bert_LR = 0.0

    for lr in lrs:
        for bert_lr in bert_lrs:

            model = Model.Model_GAT(config=config)
            print_params(model)

            if use_gpu and torch.cuda.is_available():
                torch.cuda.set_device(device=device)

            if use_gpu:
                model = model.cuda()

            criterion = Model.AsymmetricLossOptimized(gamma_neg=3)

            bert_param_ids = list(map(id, model.bert.parameters()))
            bgat_param_ids = list(map(id, model.bias_gat_layer.parameters()))
            base_params = filter(lambda p: p.requires_grad and id(p) not in bert_param_ids + bgat_param_ids,
                                 model.parameters())

            optimizer = AdamW([
                {'params': model.bert.parameters(), 'lr': bert_lr},
                {'params': model.bias_gat_layer.parameters(), 'lr': lr},
                {'params': base_params, 'weight_decay': config.lr}
            ], lr=lr, eps=config.eps)

            train_set = process.process_train_data(train_data=train_data, config=config)

            total_steps = len(train_set) * epoches
            warmup_steps = int(total_steps * config.warmup_ratio)

            scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=warmup_steps,
                                                        num_training_steps=total_steps)

            train_loss = []
            test_loss = []

            best_F1 = 0
            input_theta = 0

            train_index = [i for i in range(len(train_set))]

            for epoch in range(epoches):

                random.shuffle(train_index)

                train_epoch_loss = train(train_set=train_set, train_index=train_index, model=model,
                                         criterion=criterion, optimizer=optimizer, scheduler=scheduler)
                train_loss.append(train_epoch_loss)

                train_set = process.process_train_data(train_data=train_data, config=config)

                print("Test set devel: ")
                f1, theta, test_epoch_loss = devel(epoch=epoch, dev_set=test_set, model=model,
                                                   criterion=criterion,
                                                   optimizer=optimizer)
                test_loss.append(test_epoch_loss)

                if f1 > best_F1:
                    best_F1 = f1
                    input_theta = theta

            print("Best NT F1: %s, Theta: %s" % (str(best_F1), str(input_theta)))

            if BEST_F1 < best_F1:
                BEST_F1 = best_F1
                BEST_LR = lr
                BEST_Bert_LR = bert_lr
                torch.save(model.state_dict(), save_model_path)

            if BEST_F1 < best_F1:
                BEST_F1 = best_F1
                BEST_LR = lr
                BEST_Bert_LR = bert_lr

    print("BEST LR: %s; BEST Bert LR: %s; BEST F1: %s" % (str(BEST_LR), str(BEST_Bert_LR), str(BEST_F1)))

    return

if __name__ == '__main__':
    main()
