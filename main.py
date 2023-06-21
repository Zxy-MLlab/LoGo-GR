import re
import numpy as np
import json
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from sklearn.metrics import auc
import random

from model import Config, Model
import processing


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
    vec_path = origin_path + '/' + 'vec.npy'

    vec = np.load(vec_path, allow_pickle=True)

    word2id_path = origin_path + '/' + 'word2id.json'

    word2id = json.load(open(word2id_path, 'r'))

    ner2id_path = origin_path + '/' + 'ner2id.json'

    ner2id = json.load(open(ner2id_path, 'r'))

    id2rel_path = origin_path + '/' + 'id2rel.json'

    id2rel = json.load(open(id2rel_path, 'r'))

    return vec, word2id, ner2id, id2rel


def read_data(origin_path):
    train_data_path = origin_path + '/' + 'dev_train.json'
    dev_data_path = origin_path + '/' + 'dev_dev.json'
    test_data_path = origin_path + '/' + 'dev_test.json'

    train_data = json.load(open(train_data_path, 'r'))
    dev_data = json.load(open(dev_data_path, 'r'))
    test_data = json.load(open(test_data_path, 'r'))

    return train_data, dev_data, test_data


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

        x_len = batch_data[12]

        if use_gpu:
            batch_data = [data.cuda() for data in batch_data[: -1]]

        x = batch_data[0]
        pos = batch_data[1]
        ner = batch_data[2]
        w2m_mapping = batch_data[3]
        mention_bias_mat = batch_data[4]
        mention_edge_mat = batch_data[5]
        m2e_mapping = batch_data[6]
        path_bias_mat = batch_data[7]
        path_edge_mat = batch_data[8]
        ht_pos = batch_data[9]
        relation_multi_label = batch_data[10]
        relation_mask = batch_data[11]

        dis = ht_pos

        predict_re = model(
            x=x,
            ner=ner,
            pos=pos,
            x_lens=x_len,
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
            x_len = batch_data[19]

            if use_gpu:
                batch_data = [data.cuda() for data in batch_data[: -5]]

            x = batch_data[0]
            pos = batch_data[1]
            ner = batch_data[2]
            w2m_mapping = batch_data[3]
            mention_bias_mat = batch_data[4]
            mention_edge_mat = batch_data[5]
            m2e_mapping = batch_data[6]
            path_bias_mat = batch_data[7]
            path_edge_mat = batch_data[8]
            ht_pos = batch_data[9]
            intra_mask = batch_data[10]
            inter_mask = batch_data[11]
            intrain_mask = batch_data[12]
            relation_multi_label = batch_data[13]
            relation_mask = batch_data[14]

            dis = ht_pos

            outputs = model(
                x=x,
                ner=ner,
                pos=pos,
                x_lens=x_len,
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

    print("ALL : Epoch: %s | NT F1: %s | F1: %s | Intra F1: %s | Inter F1: %s | Precision: %s | Recall: %s | AUC: %s | THETA: %s"%(
        str(epoch), str(nt_f1), str(f1_arr[nt_f1_pos]), str(intra_f1), str(inter_f1), str(pr_y[nt_f1_pos]), str(pr_x[nt_f1_pos]), str(auc_score), str(theta)
    ))

    mean_loss = np.mean(loss_list)

    return nt_f1, theta, mean_loss


def test(test_set, model, theta, id2rel):
    model.eval()

    test_result = []
    with torch.no_grad():
        for i, batch_data in enumerate(test_set):
            x_len = batch_data[10]
            L_vertex = batch_data[11]
            titles = batch_data[12]

            if use_gpu:
                batch_data = [data.cuda() for data in batch_data[: -3]]

            x = batch_data[0]
            pos = batch_data[1]
            ner = batch_data[2]
            w2m_mapping = batch_data[3]
            mention_bias_mat = batch_data[4]
            mention_edge_mat = batch_data[5]
            m2e_mapping = batch_data[6]
            path_bias_mat = batch_data[7]
            path_edge_mat = batch_data[8]
            ht_pos = batch_data[9]

            dis = ht_pos

            outputs = model(
                x=x,
                ner=ner,
                pos=pos,
                x_lens=x_len,
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
                                test_result.append((float(outputs[i, h_idx, t_idx, r]), titles[i], id2rel[str(r)],
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
output_path = 'output'
save_model_path = 'output/model.pt'
max_len = 512
relation_num = 97
h_t_limit = 1800
batch_size = 8
epoches = 100
use_gpu = 0
device = 0

def main():
    train_data, dev_data, test_data = read_data(origin_path=origin_path)
    vec, word2id, ner2id, id2rel = read_vec(origin_path=origin_path)

    config = Config.config(
        vec=vec,
        max_len=512,
        relation_num=relation_num,
        h_t_limit=h_t_limit,
        batch_size=batch_size,
        word2id=word2id,
        ner2id=ner2id,
        id2rel=id2rel,
        use_gpu=use_gpu,
    )

    print('Processing Data...')
    process = processing.Process()

    train_set = process.process_train_data(train_data=train_data, config=config)
    dev_set = process.process_dev_data(dev_data=dev_data, config=config)
    test_set = process.process_test_data(test_data=test_data, config=config)

    model = Model.Model_GAT(config=config)
    print_params(model)

    if use_gpu and torch.cuda.is_available():
        torch.cuda.set_device(device=device)

    if use_gpu:
        model = model.cuda()

    criterion = Model.AsymmetricLossOptimized(gamma_neg=3)

    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.base_lr)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

    train_loss = []
    test_loss = []

    best_F1 = 0
    input_theta = 0

    train_index = [i for i in range(len(train_set))]

    for epoch in range(epoches):

        random.shuffle(train_index)

        train_epoch_loss = train(train_set=train_set, train_index=train_index, model=model, criterion=criterion, optimizer=optimizer, scheduler=scheduler)
        train_loss.append(train_epoch_loss)

        f1, theta, test_epoch_loss = devel(epoch=epoch, dev_set=dev_set, model=model, criterion=criterion, optimizer=optimizer)
        test_loss.append(test_epoch_loss)

        if f1 > best_F1:
            best_F1 = f1
            input_theta = theta
            torch.save(model.state_dict(), save_model_path)

    print("Best NT F1: %s, Theta: %s"%(str(best_F1), str(input_theta)))

    output = test(test_set=test_set, model=model, theta=input_theta, id2rel=id2rel)

    result_path = output_path + '/' + 'result.json'
    with open(result_path, "w", encoding="utf-8") as f_write:
        json.dump(output, f_write)
    f_write.close()

    train_loss_path = output_path + '/' + 'train_loss.csv'
    with open(train_loss_path, 'a', encoding='utf-8') as f_write:
        for i, loss in enumerate(train_loss):
            f_write.write(str(i))
            f_write.write('\t')
            f_write.write(str(loss))
            f_write.write('\n')
    f_write.close()

    test_loss_path = output_path + '/' + 'test_loss.csv'
    with open(test_loss_path, 'a', encoding='utf-8') as f_write:
        for i, loss in enumerate(test_loss):
            f_write.write(str(i))
            f_write.write('\t')
            f_write.write(str(loss))
            f_write.write('\n')
    f_write.close()

    return

if __name__ == '__main__':
    main()