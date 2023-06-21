import re

class config():
    def __init__(self, vec, max_len, relation_num, h_t_limit, batch_size, word2id, ner2id, id2rel, use_gpu):
        self.vec = vec

        self.max_mention_num = 100
        self.max_entity_num = 50

        self.lr = 1e-3
        self.base_lr = 1e-5

        self.use_gpu = use_gpu

        self.freeze_epoches = 50

        self.max_len = max_len
        self.relation_num = relation_num
        self.h_t_limit = h_t_limit
        self.batch_size = batch_size

        self.word2id = word2id
        self.ner2id = ner2id
        self.id2rel = id2rel

        self.use_entity_embed = True
        self.use_coref_embed = True
        self.use_dis_embed = True

        self.entity_embed_size = 7
        self.coref_embed_size = self.max_len

        self.me_embed_size = 6

        self.en_embed_size = 5

        self.dis_embed_size = 20

        self.embed_drop_rate = 0.5

        # Context Enocder

        self.word_embed_dim = self.vec.shape[1]
        self.entity_embed_dim = 20
        self.coref_embed_dim = 20

        self.me_embed_dim = 20
        self.en_embed_dim = 20

        self.dis_embed_dim = 20

        self.cencode_input_dim = self.word_embed_dim + self.entity_embed_dim
        self.cencode_hidden_dim = 512

        self.dropout_rate = 0.33

        self.residual = True

        # Bias-Gat-Layer

        self.in_channels = 512
        self.out_channels= 128

        self.pre_channels = 1024 + self.dis_embed_dim

        self.layer_num = 3

        # Predict

        self.hidden_dim = 512

        return