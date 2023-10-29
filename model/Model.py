import re

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
from transformers import BertModel

class AsymmetricLossOptimized(nn.Module):
    ''' Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations'''

    def __init__(self, gamma_neg=4, gamma_pos=0, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
        super(AsymmetricLossOptimized, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        # prevent memory allocation and gpu uploading every iteration, and encourages inplace operations
        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        self.targets = y
        self.anti_targets = 1 - y

        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)

        # Basic CE calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch._C.set_grad_enabled(False)
            self.xs_pos = self.xs_pos * self.targets
            self.xs_neg = self.xs_neg * self.anti_targets
            self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                          self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
            if self.disable_torch_grad_focal_loss:
                torch._C.set_grad_enabled(True)
            self.loss *= self.asymmetric_w

        return -self.loss.sum()

class Embedding(torch.nn.Module):
    def __init__(self, config):
        super(Embedding, self).__init__()

        self.config = config

        self.use_entity_embed = self.config.use_entity_embed
        self.use_coref_embed = self.config.use_coref_embed

        self.entity_embed_size = self.config.entity_embed_size
        self.coref_embed_size = self.config.coref_embed_size

        self.me_embed_size = self.config.me_embed_size
        self.en_embed_size = self.config.en_embed_size

        self.dis_embed_size = self.config.dis_embed_size

        self.entity_embed_dim = self.config.entity_embed_dim
        self.coref_embed_dim = self.config.coref_embed_dim

        self.me_embed_dim = self.config.me_embed_dim
        self.en_embed_dim = self.config.en_embed_dim

        self.dis_embed_dim = self.config.dis_embed_dim

        self.entity_embedding = nn.Embedding(self.entity_embed_size, self.entity_embed_dim, padding_idx=0)
        self.coref_embedding = nn.Embedding(self.coref_embed_size, self.coref_embed_dim, padding_idx=0)

        self.me_embedding = nn.Embedding(self.me_embed_size, self.me_embed_dim, padding_idx=0)
        self.en_embedding = nn.Embedding(self.en_embed_size, self.en_embed_dim, padding_idx=0)

        self.dis_embedding = nn.Embedding(self.dis_embed_size, self.dis_embed_dim, padding_idx=0)

        return

    def forward(self, x, ner, pos):

        if self.use_entity_embed:
            ner_embed = self.entity_embedding(ner)
            x_embed = torch.cat([x, ner_embed], dim=-1)
        else:
            assert 0 == 1

        # if self.use_coref_embed:
        #     coref_embed = self.coref_embedding(pos)
        #     x_embed = torch.cat([x_embed, coref_embed], dim=-1)

        return x_embed

class MLP(torch.nn.Module):
    def __init__(self, n_in, n_out, dropout=0):
        super(MLP, self).__init__()

        self.linear = nn.Linear(n_in, n_out)
        self.activation = nn.LeakyReLU(negative_slope=0.1)
        self.dropout = nn.Dropout(dropout)

        return

    def forward(self, x):
        x = self.dropout(x)
        x = self.linear(x)
        x = self.activation(x)

        return x

class Context_Encoder(torch.nn.Module):
    def __init__(self):
        super(Context_Encoder, self).__init__()

        self.encoder = BertModel.from_pretrained('scibert_scivocab_uncased')

        return

    def forward(self, input_ids, mask, pos_ids):

        outs = self.encoder(input_ids, attention_mask=mask, position_ids=pos_ids,
                            token_type_ids=torch.zeros_like(pos_ids, device=pos_ids.device, dtype=torch.long))[0]

        outs = outs[:, 1:-1]

        return outs

class BiasGatLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, pre_channels, me_embed_dim, en_embed_dim, dropout_rate=0.1, residual=True):
        super(BiasGatLayer, self).__init__()

        self.residual = residual

        self.in_dropout_rate = dropout_rate
        self.out_dropout_rate = dropout_rate
        self.dropout_rate = dropout_rate

        self.x_conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.y_conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.x_conv2 = nn.Conv1d(out_channels, 1, kernel_size=1)
        self.y_conv2 = nn.Conv1d(out_channels, 1, kernel_size=1)

        self.p_conv = nn.Conv2d(en_embed_dim, 1, kernel_size=1, bias=False)
        self.e_conv = nn.Conv2d(me_embed_dim, 1, kernel_size=1, bias=False)

        self.conv = nn.Sequential(
            nn.Dropout2d(self.dropout_rate),
            nn.Conv2d(2*out_channels+pre_channels, out_channels, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

        self.score_layer = nn.Conv2d(out_channels, 1, kernel_size=1, bias=False)

        self.x_linear = nn.Linear(in_channels, in_channels, bias=False)
        self.y_linear = nn.Linear(in_channels, in_channels, bias=False)

        self.leakyrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.in_dropout = nn.Dropout(self.in_dropout_rate)
        self.out_dropout = nn.Dropout(self.out_dropout_rate)

        return

    def forward(self, x, y, m_bias_mat, edge_mat, edge_embed_layer,
                p_bias_mat, path_mat, path_embed_layer, pre_out=None):
        ori_x, ori_y = x, y
        entity_num = p_bias_mat.shape[1]

        B, _, M = x.size()
        B, _, N = y.size()

        x_f = self.x_conv1(x)
        y_f = self.y_conv1(y)

        if self.in_dropout_rate != 0:
            x_f = self.in_dropout(x_f)
            y_f = self.in_dropout(y_f)

        x_self = self.x_conv2(x_f)
        y_other = self.y_conv2(y_f)

        logist = self.leakyrelu(x_self + torch.transpose(y_other, 2, 1))
        # if self.out_dropout_rate != 0.0:
        #     logist = self.out_dropout(logist)

        logist = logist.unsqueeze(1)

        # if self.in_dropout_rate != 0.0:
        #     x_f = self.in_dropout(x_f)
        #     y_f = self.in_dropout(y_f)

        rets = []
        for i in range(entity_num):
            cur_bias_mat = p_bias_mat[:, i, :, :].unsqueeze(1)
            cur_path_mat = path_mat[:, i, :, :]

            cur_path_embed = path_embed_layer(cur_path_mat).permute(0, 3, 1, 2)
            cur_path_logist = self.leakyrelu(self.p_conv(cur_path_embed))

            # if self.out_dropout_rate != 0.0:
            #     cur_path_logist = self.out_dropout(cur_path_logist)

            local_coefs = logist + cur_path_logist + cur_bias_mat

            x_ret = torch.matmul(F.softmax(local_coefs, -1), y_f.transpose(1, 2).view(B, N, 1, -1).transpose(1, 2))
            x_ret = x_ret.transpose(1, 2).contiguous().view(B, M, -1)
            y_ret = torch.matmul(F.softmax(local_coefs.transpose(2, 3), -1), x_f.transpose(1, 2).view(B, M, 1, -1).transpose(1, 2))
            y_ret = y_ret.transpose(1, 2).contiguous().view(B, N, -1)

            x_ret = self.leakyrelu(x_ret + x_f.transpose(1, 2)).transpose(1, 2)
            y_ret = self.leakyrelu(y_ret + y_f.transpose(1, 2)).transpose(1, 2)

            ret = torch.cat([x_ret, y_ret], dim=1)

            rets.append(ret)

        path_fea = torch.cat(rets, dim=-1).view(B, -1, entity_num, entity_num)

        if pre_out is not None:
            men2rel_fea = torch.cat([path_fea, pre_out], 1)
        else:
            assert 1 == 2
            # men2rel_fea = path_fea

        men2rel_fea = self.conv(men2rel_fea)
        scores = self.leakyrelu(self.score_layer(men2rel_fea))

        edge_embed = edge_embed_layer(edge_mat).permute(0, 3, 1, 2)
        edge_logist = self.leakyrelu(self.e_conv(edge_embed))

        # if self.out_dropout_rate != 0.0:
        #     edge_logist = self.out_dropout(edge_logist)

        m_bias_mat = m_bias_mat.unsqueeze(1)

        global_coefs = logist + scores + edge_logist + m_bias_mat

        x = self.x_linear(x.transpose(1, 2))
        y = self.y_linear(y.transpose(1, 2))

        out_x = torch.matmul(F.softmax(global_coefs, -1), y.view(B, N, 1, -1).transpose(1, 2))
        out_x = out_x.transpose(1, 2).contiguous().view(B, M, -1)
        out_y = torch.matmul(F.softmax(global_coefs.transpose(2, 3), -1), x.view(B, M, 1, -1).transpose(1,2))
        out_y = out_y.transpose(1, 2).contiguous().view(B, N, -1)

        # out_x = self.leakyrelu(out_x + x)
        # out_y = self.leakyrelu(out_y + y)

        if self.out_dropout_rate != 0.0:
            out_x = self.out_dropout(out_x)
            out_y = self.out_dropout(out_y)

        if self.residual:
            out_x = out_x.transpose(1, 2) + ori_x
            out_y = out_y.transpose(1, 2) + ori_y

        return out_x, out_y, men2rel_fea

class BiasGat(torch.nn.Module):
    def __init__(self, config):
        super(BiasGat, self).__init__()

        self.in_channels = config.in_channels
        self.out_channels = config.out_channels
        self.pre_channels = config.pre_channels

        self.en_embed_dim = config.en_embed_dim
        self.me_embed_dim = config.me_embed_dim

        self.dropout_rate = config.gat_drop_rate
        self.residual = config.residual

        self.layer_num = config.layer_num

        self.layers = nn.ModuleList([
            BiasGatLayer(in_channels=self.in_channels, out_channels=self.out_channels, pre_channels=self.pre_channels if i == 0 else self.out_channels,
                         en_embed_dim=self.en_embed_dim, me_embed_dim=self.me_embed_dim,
                         dropout_rate=self.dropout_rate, residual=self.residual) for i in range(self.layer_num)
        ])

        return

    def forward(self, x, y, m_bias_mat, edge_mat, edge_embed_layer,
                p_bias_mat, path_mat, path_embed_layer, pre_out):

        fea_list = []
        for layer in self.layers:
            x, y, men2rel_fea = layer(x, y, m_bias_mat, edge_mat, edge_embed_layer,
                                      p_bias_mat, path_mat, path_embed_layer, pre_out)

            pre_out = men2rel_fea

            fea_list.append(men2rel_fea)

        return x, y, pre_out.permute(0, 2, 3, 1).contiguous()

class Mapping(torch.nn.Module):
    def __init__(self):
        super(Mapping, self).__init__()

        return

    def forward(self, x, mapping):

        mapping_x = torch.matmul(mapping, x)

        return mapping_x

class Biaffine(nn.Module):
    def __init__(self, n_in, n_out=1, bias_x=True, bias_y=True):
        super(Biaffine, self).__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.bias_x = bias_x
        self.bias_y = bias_y
        weight = torch.zeros((n_out, n_in + int(bias_x), n_in + int(bias_y)))
        nn.init.xavier_normal_(weight)
        self.weight = nn.Parameter(weight, requires_grad=True)

    def extra_repr(self):
        s = f"n_in={self.n_in}, n_out={self.n_out}"
        if self.bias_x:
            s += f", bias_x={self.bias_x}"
        if self.bias_y:
            s += f", bias_y={self.bias_y}"

        return s

    def forward(self, x, y):
        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), -1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), -1)
        # [batch_size, n_out, seq_len, seq_len]
        s = torch.einsum('bxi,oij,byj->boxy', x, self.weight, y)
        # remove dim 1 if n_out == 1
        s = s.permute(0, 2, 3, 1)

        return s

class Model_GAT(torch.nn.Module):
    def __init__(self, config):
        super(Model_GAT, self).__init__()

        self.config = config

        self.embed_layer = Embedding(config=self.config)
        self.bert = BertModel.from_pretrained("scibert_scivocab_uncased")

        self.bias_gat_layer = BiasGat(config=self.config)

        self.sub_mlp = MLP(n_in=config.in_channels, n_out=config.hidden_dim, dropout=config.dropout_rate)
        self.obj_mlp = MLP(n_in=config.in_channels, n_out=config.hidden_dim, dropout=config.dropout_rate)

        self.rel_mlp = MLP(n_in=config.out_channels, n_out=config.out_channels, dropout=config.dropout_rate)

        self.biaffne = Biaffine(n_in=config.hidden_dim, n_out=config.relation_num, bias_x=True, bias_y=True)
        self.rel_linear = nn.Linear(config.out_channels, config.relation_num)

        self.embed_dropout = nn.Dropout(p=self.config.embed_drop_rate)
        self.dropout = nn.Dropout(p=self.config.dropout_rate)

        return

    def forward(self, input_ids, pos_ids, ner, pos, w2m_mapping,
                m_bias_mat, m_edge_mat, p_bias_mat, p_edge_mat,
                m2e_mapping, dis):


        # Bert Layer
        mask = input_ids.ne(0)
        encode_x = self.bert(input_ids, attention_mask=mask, position_ids=pos_ids,
                                token_type_ids=torch.zeros_like(pos_ids, device=pos_ids.device, dtype=torch.long))[0]

        encode_x = encode_x[:, 1:-1]

        # Embedding
        encode_x = self.embed_layer(encode_x, ner, pos)
        encode_x = self.embed_dropout(encode_x)

        # MaxPool Layer
        min_m_value = torch.min(encode_x).item()
        max_m = w2m_mapping.size(1)

        _encode_x = encode_x.unsqueeze(1).expand(-1, max_m, -1, -1)
        _encode_x = torch.masked_fill(_encode_x, w2m_mapping.eq(0).unsqueeze(-1), min_m_value)

        encode_mention, _ = torch.max(_encode_x, dim=2)

        _, N, _ = encode_mention.size()

        dis_embed = self.embed_layer.dis_embedding(dis)

        # pre_out = torch.cat([encode_mention.unsqueeze(2).repeat_interleave(N, 2), encode_mention.unsqueeze(1).repeat_interleave(N, 1), dis_embed],
        #                     dim=-1).permute(0, 3, 1, 2).contiguous()

        pre_out = dis_embed.permute(0, 3, 1, 2)

        encode_mention = encode_mention.transpose(1, 2)
        sub_men, obj_men, men2rel_fea = self.bias_gat_layer(encode_mention, encode_mention, m_bias_mat,
                                                            m_edge_mat, self.embed_layer.me_embedding, p_bias_mat,
                                                            p_edge_mat, self.embed_layer.en_embedding, pre_out)

        sub_men = sub_men.transpose(1, 2)
        obj_men = obj_men.transpose(1, 2)

        # Global MaxPool Layer
        min_sub_value = torch.min(sub_men).item()
        max_e = m2e_mapping.size(1)

        sub_men = sub_men.unsqueeze(1).expand(-1, max_e, -1, -1)
        sub_men = torch.masked_fill(sub_men, m2e_mapping.eq(0).unsqueeze(-1), min_sub_value)
        sub_men, _ = torch.max(sub_men, dim=2)

        min_obj_value = torch.min(obj_men).item()

        obj_men = obj_men.unsqueeze(1).expand(-1, max_e, -1, -1)
        obj_men = torch.masked_fill(obj_men, m2e_mapping.eq(0).unsqueeze(-1), min_obj_value)
        obj_men, _ = torch.max(obj_men, dim=2)

        sub_ent = self.dropout(self.sub_mlp(sub_men))
        obj_ent = self.dropout(self.obj_mlp(obj_men))

        re_outputs1 = self.biaffne(sub_ent, obj_ent)

        # Local MaxPool Layer
        min_f_value = torch.min(men2rel_fea).item()

        men2rel_fea = men2rel_fea.unsqueeze(1).repeat(1, max_e, 1, 1, 1)
        men2rel_fea = torch.masked_fill(men2rel_fea, m2e_mapping.eq(0)[:, :, :, None, None], min_f_value)
        men2rel_fea, _ = torch.max(men2rel_fea, dim=2)

        men2rel_fea = men2rel_fea.unsqueeze(1).repeat(1, max_e, 1, 1, 1)
        men2rel_fea = torch.masked_fill(men2rel_fea, m2e_mapping.eq(0)[:, :, None, :, None], min_f_value)
        men2rel_fea, _ = torch.max(men2rel_fea, dim=3)

        men2rel_fea = self.dropout(self.rel_mlp(men2rel_fea))
        re_outputs2 = self.rel_linear(men2rel_fea)

        re_outputs = re_outputs1 + re_outputs2

        return re_outputs
