import re

class config():
    def __init__(self, max_len, relation_num, h_t_limit, batch_size, use_gpu, ner2id):

        self.max_mention_num = 100
        self.max_entity_num = 30

        self.max_tokens = 1000
        self.lr = 5e-5
        self.gat_lr = 5e-5
        self.base_lr = 1e-5
        self.bert_lr = 1e-5

        self.eps = 1e-6
        self.warmup_ratio = 0.06

        self.use_gpu = use_gpu

        self.freeze_epoches = 50

        self.ner2id = ner2id

        self.max_len = max_len
        self.relation_num = relation_num
        self.h_t_limit = h_t_limit
        self.batch_size = batch_size

        self.use_entity_embed = True
        self.use_coref_embed = True
        self.use_dis_embed = True

        self.entity_embed_size = 7
        self.coref_embed_size = self.max_len

        self.me_embed_size = 6

        self.en_embed_size = 5

        self.dis_embed_size = 20

        self.embed_drop_rate = 0.0

        # Context Enocder

        self.word_embed_dim = 768
        self.entity_embed_dim = 20
        self.coref_embed_dim = 20

        self.me_embed_dim = 20
        self.en_embed_dim = 20

        self.dis_embed_dim = 20

        self.cencode_input_dim = self.word_embed_dim + self.entity_embed_dim
        self.cencode_hidden_dim = self.word_embed_dim

        self.gat_drop_rate = 0.5
        self.dropout_rate = 0.3

        self.residual = True

        # Bias-Gat-Layer

        self.in_channels = self.word_embed_dim + self.entity_embed_dim
        self.out_channels = 128

        self.pre_channels = self.dis_embed_dim

        self.layer_num = 3

        # Predict

        self.hidden_dim = self.in_channels // 2

        return

    def set_dp(self, embed_dp, gat_dp, base_dp):
        self.embed_drop_rate = embed_dp

        self.gat_drop_rate = gat_dp

        self.dropout_rate = base_dp

        return
