# -*- coding: utf-8 -*-
import torch
from torch import nn
import torch.nn.functional as F
from utility import auto_rnn_bilstm, fit_seq_max_len, auto_rnn_bigru


class SamePadConv1D(nn.Conv1d):
    """
    自定义一维卷积核，实现与Keras中padding=same的操作
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 dilation=1, groups=1, bias=True):
        if (kernel_size - 1) % 2 != 0:
            pad = (kernel_size - 1) // 2
            self.padding_shape = (pad, pad + 1)
        else:
            pad = (kernel_size - 1) // 2
            self.padding_shape = (pad, pad)
        super(SamePadConv1D, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias
        )

    def forward(self, inputs):
        x = F.pad(inputs, self.padding_shape)
        return super(SamePadConv1D, self).forward(x)


class StackCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, kernel_sizes=[2, 3], out_channels=[800, 800], embed_weight=None):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, _weight=embed_weight)

        self.stack_conv = nn.Sequential()
        in_channels = embed_dim
        for ks, oc in zip(kernel_sizes, out_channels):
            self.stack_conv.add_module(SamePadConv1D(in_channels, oc, ks))
            self.stack_conv(nn.ReLU(inplace=True))
            in_channels = oc
        self.cos = nn.CosineSimilarity(dim=1)

    def forward(self, question, answer):
        question, question_length = question
        answer, answer_length = answer
        bs = question.size(0)
        sent = torch.cat((question, answer), dim=0)
        embed = self.embed(sent)
        embed = embed.transpose(1, 2)

        conv3 = self.conv3(embed)
        conv4 = self.conv4(embed)
        stack_conv = self.stack_conv(embed)
        out3 = torch.max(conv3, dim=2)[0]
        out4 = torch.max(conv4, dim=2)[0]
        stack_out = torch.max(stack_conv, dim=2)[0]

        q_out = torch.cat((out3[:bs], out4[:bs], stack_out[:bs]), dim=1)
        a_out = torch.cat((out3[bs:], out4[bs:], stack_out[bs:]), dim=1)

        out = self.cos(q_out, a_out)
        return out


class StackMultiCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, embed_weight=None):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, _weight=embed_weight)
        self.conv3 = SamePadConv1D(embed_dim, 800, 3)
        self.conv4 = SamePadConv1D(embed_dim, 800, 4)
        self.stack_conv = nn.Sequential(
            SamePadConv1D(embed_dim, 800, 2),
            nn.ReLU(inplace=True),
            SamePadConv1D(800, 800, 3),
            nn.ReLU(inplace=True)
        )
        self.cos = nn.CosineSimilarity(dim=1)

    def forward(self, question, answer):
        question, question_length = question
        answer, answer_length = answer
        bs = question.size(0)
        sent = torch.cat((question, answer), dim=0)
        embed = self.embed(sent)
        embed = embed.transpose(1, 2)

        conv3 = self.conv3(embed)
        conv4 = self.conv4(embed)
        stack_conv = self.stack_conv(embed)
        out3 = torch.max(conv3, dim=2)[0]
        out4 = torch.max(conv4, dim=2)[0]
        stack_out = torch.max(stack_conv, dim=2)[0]

        q_out = torch.cat((out3[:bs], out4[:bs], stack_out[:bs]), dim=1)
        a_out = torch.cat((out3[bs:], out4[bs:], stack_out[bs:]), dim=1)

        out = self.cos(q_out, a_out)
        return out


class NormStackMultiCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, sent_length, embed_weight=None):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, _weight=embed_weight)
        self.norm_embed = nn.BatchNorm1d(sent_length)
        self.conv3 = nn.Conv1d(embed_dim, 800, 3)
        self.norm3 = nn.BatchNorm1d(800)
        self.conv4 = nn.Conv1d(embed_dim, 800, 4)
        self.norm4 = nn.BatchNorm1d(800)

        self.stack_conv = nn.Sequential(
            nn.Conv1d(embed_dim, 800, 2),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(800),
            nn.Conv1d(800, 800, 3),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(800)
        )
        self.cos = nn.CosineSimilarity(dim=1)

    def forward(self, question, answer):
        question, question_length = question
        answer, answer_length = answer
        bs = question.size(0)
        sent = torch.cat((question, answer), dim=0)
        embed = self.embed(sent)
        embed = self.norm_embed(embed)
        embed = embed.transpose(1, 2)  # (B, D, T)

        conv3 = self.conv3(embed)
        conv3 = self.norm3(conv3)
        conv4 = self.conv4(embed)
        conv4 = self.norm4(conv4)
        stack_conv = self.stack_conv(embed)
        out3 = torch.max(conv3, dim=2)[0]
        out4 = torch.max(conv4, dim=2)[0]
        stack_out = torch.max(stack_conv, dim=2)[0]

        q_out = torch.cat((out3[:bs], out4[:bs], stack_out[:bs]), dim=1)
        a_out = torch.cat((out3[bs:], out4[bs:], stack_out[bs:]), dim=1)

        out = self.cos(q_out, a_out)
        return out


class QA_StackMultiAttentionCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, embed_weight=None):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, _weight=embed_weight)
        self.multi_conv = nn.ModuleList([
            nn.Conv1d(embed_dim, 400, 3),
            nn.Conv1d(embed_dim, 400, 4)
        ])
        self.stack_conv = nn.Sequential(
            nn.Conv1d(embed_dim, 400, 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(400, 400, 3),
            nn.ReLU(inplace=True)
        )
        self.cos = nn.CosineSimilarity(dim=1)

    def forward(self, question, answer):
        question, question_length = question
        answer, answer_length = answer
        bs = question.size(0)
        sent = torch.cat((question, answer), dim=0)
        embed = self.embed(sent)
        embed = embed.transpose(1, 2)

        multi_out = []
        for conv in self.multi_conv:
            multi_out.append(torch.max(conv(embed), dim=2)[0])
        stack_conv = self.stack_conv(embed)
        stack_out = torch.max(stack_conv, dim=2)[0]
        multi_out.append(stack_out)

        out = torch.cat(multi_out, dim=1)
        sim = self.cos(out[:bs], out[bs:])
        return sim


class QA_AP_StackMultiCNN(nn.Module):
    '''
    use Attentive Polling
    '''

    def __init__(self, vocab_size, embed_dim, embed_weight=None):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, _weight=embed_weight)
        self.U = nn.Parameter(torch.rand(400, 400, requires_grad=True))  # 使用卷积后的结果做attentiv，所以维度用conv的输出维度
        self.multi_conv = nn.ModuleList([
            nn.Conv1d(embed_dim, 400, 3),
            nn.Conv1d(embed_dim, 400, 4)
        ])
        self.stack_conv = nn.Sequential(
            nn.Conv1d(embed_dim, 400, 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(400, 400, 3),
            nn.ReLU(inplace=True)
        )
        self.cos = nn.CosineSimilarity(dim=1)

    def attentive_pooling(self, q, a):
        """Attentive pooling
        Args:
            q: encoder output for question (batch_size, q_len, vector_size)
            a: encoder output for question (batch_size, a_len, vector_size)
        Returns:
            final representation Tensor r_q, r_a for q and a (batch_size, vector_size)
        """
        batch_size = q.size(0)
        U_batch = self.U.unsqueeze(0).repeat(batch_size, 1, 1)
        G = torch.tanh(torch.matmul(torch.matmul(q, U_batch), a.transpose(1, 2)))
        g_q = torch.max(G, dim=2)[0]
        g_a = torch.max(G, dim=1)[0]
        sigma_q = F.softmax(g_q, dim=-1)
        sigma_a = F.softmax(g_a, dim=-1)
        r_q = torch.matmul(q.transpose(1, 2), sigma_q.unsqueeze(2)).squeeze(2)
        r_a = torch.matmul(a.transpose(1, 2), sigma_a.unsqueeze(2)).squeeze(2)
        return r_q, r_a

    def forward(self, question, answer):
        question, question_length = question
        answer, answer_length = answer
        bs = question.size(0)
        sent = torch.cat((question, answer), dim=0)
        embed = self.embed(sent)
        embed = embed.transpose(1, 2)

        multi_out = []
        for conv in self.multi_conv:
            multi_out.append(conv(embed))
        stack_out = self.stack_conv(embed)
        multi_out.append(stack_out)
        out = torch.cat(multi_out, dim=2)
        out = out.transpose(1, 2)
        q, a = self.attentive_pooling(out[:bs], out[bs:])
        sim = self.cos(q, a)
        return sim


class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size=200, dropout_r=0.1, embed_weight=None):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embedding_dim, _weight=embed_weight)

        self.lstm_q = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, dropout=dropout_r,
                              num_layers=1, batch_first=True, bidirectional=True)

        self.lstm_a = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, dropout=dropout_r,
                              num_layers=1, batch_first=True, bidirectional=True)
        # lstm_out_dim = hidden_size
        # self.liner_q = nn.Sequential(nn.Dropout(dropout_r, inplace=True),
        #                              nn.Linear(lstm_out_dim, hidden_size),
        #                              nn.Tanh())
        # self.liner_a = nn.Sequential(nn.Dropout(dropout_r, inplace=True),
        #                              nn.Linear(lstm_out_dim, hidden_size),
        #                              nn.Tanh())

        self.similarity = nn.CosineSimilarity(dim=1)

    def forward(self, question, answer):
        question, question_length = question
        answer, answer_length = answer
        question = fit_seq_max_len(question, question_length)
        answer = fit_seq_max_len(answer, answer_length)

        embed_q = self.embed(question)  # (bs, sent, vector)
        embed_a = self.embed(answer)  # (bs, sent, vector)

        lstm_q, (hn_q, cn_q) = auto_rnn_bilstm(self.lstm_q, embed_q,
                                               question_length)  # lstm_out (bs, T=sent, D=2*hidden_size)
        lstm_a, (hn_a, cn_a) = auto_rnn_bilstm(self.lstm_a, embed_a,
                                               answer_length)  # lstm_out (bs, T=sent, D=2*hidden_size)

        # select last hidden state
        lstm_q = torch.cat([hn_q[-2], hn_q[-1]], dim=1)
        lstm_a = torch.cat([hn_a[-2], hn_a[-1]], dim=1)

        # out_q = self.liner_q(lstm_q)
        # out_a = self.liner_q(lstm_a)

        sim = self.similarity(lstm_q, lstm_a)

        return sim


class StackBiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, mlp_d=1600, dropout_r=0.1, embed_weight=None):
        super(StackBiLSTM, self).__init__()
        self.embed = nn.Embedding(vocab_size, embedding_dim, _weight=embed_weight)

        self.lstm_1 = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size[0], dropout=dropout_r,
                              num_layers=1, batch_first=True, bidirectional=True)

        self.lstm_2 = nn.LSTM(input_size=(embedding_dim + hidden_size[0] * 2), hidden_size=hidden_size[1],
                              dropout=dropout_r, num_layers=1, batch_first=True, bidirectional=True)

        self.lstm_3 = nn.LSTM(input_size=(embedding_dim + (hidden_size[0] + hidden_size[1]) * 2), dropout=dropout_r,
                              hidden_size=hidden_size[2], num_layers=1, batch_first=True, bidirectional=True)

        self.mlp_1 = nn.Linear(hidden_size[2] * 4, mlp_d)
        self.mlp_2 = nn.Linear(mlp_d, mlp_d)
        self.sm = nn.Linear(mlp_d, 1)

        self.classifier = nn.Sequential(*[self.mlp_1, nn.ReLU(), nn.Dropout(dropout_r),
                                          self.mlp_2, nn.ReLU(), nn.Dropout(dropout_r),
                                          self.sm])

    def forward(self, question, answer):
        question, question_length = question
        answer, answer_length = answer
        bs = question.size(0)
        sent = torch.cat((question, answer), dim=0)  # (bs, sen)
        sent_len = torch.cat((question_length, answer_length), dim=0)
        sent = fit_seq_max_len(sent, sent_len)

        embed = self.embed(sent)  # (B, T, D)
        lstm_layer1_out, (hn, cn) = auto_rnn_bilstm(self.lstm_1, embed, sent_len)  # lstm_out (B, T, D=2*hidden_size)

        layer2_in = torch.cat([embed, lstm_layer1_out], dim=2)
        lstm_layer2_out, (hn, cn) = auto_rnn_bilstm(self.lstm_2, layer2_in, sent_len)

        layer3_in = torch.cat([embed, lstm_layer1_out, lstm_layer2_out], dim=2)
        lstm_layer3_out, (hn, cn) = auto_rnn_bilstm(self.lstm_3, layer3_in, sent_len)

        # lstm_layer3_maxout = torch.max(lstm_layer3_out, dim=1)[0]  # (bs, D)
        lstm_layer3_last_hn = torch.cat([hn[-2], hn[-1]], dim=1)  # (B, D=hidden_size[2]*2)

        features = torch.cat([lstm_layer3_last_hn[:bs], lstm_layer3_last_hn[bs:],
                              torch.abs(lstm_layer3_last_hn[:bs] - lstm_layer3_last_hn[bs:]),
                              lstm_layer3_last_hn[:bs] * lstm_layer3_last_hn[bs:]],
                             dim=1)

        out = self.classifier(features)
        return out


class BiGRU(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size=200, dropout_r=0.1, embed_weight=None):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embedding_dim, _weight=embed_weight)

        self.gru_q = nn.GRU(input_size=embedding_dim, hidden_size=hidden_size, dropout=dropout_r,
                            num_layers=1, batch_first=True, bidirectional=True)

        self.gru_a = nn.GRU(input_size=embedding_dim, hidden_size=hidden_size, dropout=dropout_r,
                            num_layers=1, batch_first=True, bidirectional=True)

        self.similarity = nn.CosineSimilarity(dim=1)

    def forward(self, question, answer):
        question, question_length = question
        answer, answer_length = answer
        question = fit_seq_max_len(question, question_length)
        answer = fit_seq_max_len(answer, answer_length)

        embed_q = self.embed(question)  # (bs, sent, vector)
        embed_a = self.embed(answer)  # (bs, sent, vector)

        gru_q, hn_q = auto_rnn_bigru(self.gru_q, embed_q, question_length)  # gru_q (bs, T=sent, D=2*hidden_size)
        gru_a, hn_a = auto_rnn_bigru(self.gru_a, embed_a, answer_length)  # gru_a (bs, T=sent, D=2*hidden_size)

        # select last hidden state
        out_q = torch.cat([hn_q[-2], hn_q[-1]], dim=1)
        out_a = torch.cat([hn_a[-2], hn_a[-1]], dim=1)
        sim = self.similarity(out_q, out_a)
        return sim


class ShareBiGRU(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size=200, dropout_r=0.1, embed_weight=None):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embedding_dim, _weight=embed_weight, )

        self.gru_qa = nn.GRU(input_size=embedding_dim, hidden_size=hidden_size, dropout=dropout_r,
                             num_layers=1, batch_first=True, bidirectional=True)

        self.similarity = nn.CosineSimilarity(dim=1)

    def forward(self, question, answer):
        question, question_length = question
        answer, answer_length = answer
        bs = question.size(0)
        sent = torch.cat([question, answer], dim=0)
        sent_len = torch.cat([question_length, answer_length], dim=0)
        sent = fit_seq_max_len(sent, sent_len)
        # question = fit_seq_max_len(question, question_length)
        # answer = fit_seq_max_len(answer, answer_length)

        # embed_q = self.embed(question)  # (bs, sent, vector)
        # embed_a = self.embed(answer)  # (bs, sent, vector)
        embed = self.embed(sent)  # (bs, sent, vector)

        # gru_q, hn_q = auto_rnn_bigru(self.gru_qa, embed_q, question_length)  # gru_q (bs, T=sent, D=2*hidden_size)
        # gru_a, hn_a = auto_rnn_bigru(self.gru_qa, embed_a, answer_length)
        gru_sent, hn_sent = auto_rnn_bigru(self.gru_qa, embed, sent_len)

        # select last hidden state
        # out_q = torch.cat([hn_q[-2], hn_q[-1]], dim=1)
        # out_a = torch.cat([hn_a[-2], hn_a[-1]], dim=1)
        out = torch.cat([hn_sent[-2], hn_sent[-1]], dim=1)
        sim = self.similarity(out[:bs], out[bs:])
        return sim


class StackBiGRU(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, sent_max_length, mlp_d=1024, dropout_r=0.1,
                 embed_weight=None):
        super(StackBiGRU, self).__init__()
        self.embed = nn.Embedding(vocab_size, embedding_dim, _weight=embed_weight)

        self.gru_1 = nn.GRU(input_size=embedding_dim, hidden_size=hidden_size[0], dropout=dropout_r,
                            num_layers=1, batch_first=True, bidirectional=True)
        self.norm_1 = nn.BatchNorm1d(sent_max_length)

        self.gru_2 = nn.GRU(input_size=(embedding_dim + hidden_size[0] * 2), hidden_size=hidden_size[1],
                            dropout=dropout_r, num_layers=1, batch_first=True, bidirectional=True)
        self.norm_2 = nn.BatchNorm1d(sent_max_length)

        self.gru_3 = nn.GRU(input_size=(embedding_dim + (hidden_size[0] + hidden_size[1]) * 2), dropout=dropout_r,
                            hidden_size=hidden_size[2], num_layers=1, batch_first=True, bidirectional=True)
        self.norm_3 = nn.BatchNorm1d(hidden_size[2] * 2)

        self.mlp_1 = nn.Linear(hidden_size[2] * 2 * 4, mlp_d)
        self.mlp_2 = nn.Linear(mlp_d, mlp_d)
        self.similarity = nn.Linear(mlp_d, 1)

        self.classifier = nn.Sequential(self.mlp_1, nn.ReLU(), nn.BatchNorm1d(mlp_d), nn.Dropout(dropout_r),
                                        self.mlp_2, nn.ReLU(), nn.BatchNorm1d(mlp_d), nn.Dropout(dropout_r),
                                        self.similarity)

    def forward(self, question, answer):
        question, question_length = question
        answer, answer_length = answer
        bs = question.size(0)
        sent = torch.cat((question, answer), dim=0)  # (bs, sen)
        sent_len = torch.cat((question_length, answer_length), dim=0)
        # sent = fit_seq_max_len(sent, sent_len)

        embed = self.embed(sent)  # (B, T, D)
        gru_layer1_out, hn_1 = auto_rnn_bigru(self.gru_1, embed, sent_len,
                                              fix_length=True)  # gru_out (B, T, D=2*hidden_size)
        gru_layer1_out = self.norm_1(gru_layer1_out)

        layer2_in = torch.cat([embed, gru_layer1_out], dim=2)
        gru_layer2_out, hn_2 = auto_rnn_bigru(self.gru_2, layer2_in, sent_len, fix_length=True)
        gru_layer2_out = self.norm_2(gru_layer2_out)

        layer3_in = torch.cat([embed, gru_layer1_out, gru_layer2_out], dim=2)
        gru_layer3_out, hn_3 = auto_rnn_bigru(self.gru_3, layer3_in, sent_len, fix_length=True)

        # gru_layer3_maxout = torch.max(gru_layer3_out, dim=1)[0]  # (bs, D)
        gru_layer3_last_hn = torch.cat([hn_3[-2], hn_3[-1]], dim=1)  # (B, D=hidden_size[2]*2)
        gru_layer3_last_hn = self.norm_3(gru_layer3_last_hn)

        features = torch.cat([gru_layer3_last_hn[:bs], gru_layer3_last_hn[bs:],
                              torch.abs(gru_layer3_last_hn[:bs] - gru_layer3_last_hn[bs:]),
                              gru_layer3_last_hn[:bs] * gru_layer3_last_hn[bs:]],
                             dim=1)

        out = self.classifier(features)
        return out


class CnnShareBiGRU(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size=200, cnn_channel=500, dropout_r=0, embed_weight=None):
        super().__init__()
        print({
            'vocab_size': vocab_size,
            'embedding_dim': embedding_dim,
            'hidden_size': hidden_size,
            'cnn_channel': cnn_channel,
            'dropout_r': dropout_r
        })
        self.embed = nn.Embedding(vocab_size, embedding_dim, _weight=embed_weight)

        self.gru_qa = nn.GRU(input_size=embedding_dim, hidden_size=hidden_size, dropout=dropout_r,
                             num_layers=1, batch_first=True, bidirectional=True)
        self.cnns = nn.ModuleList()
        for ks in [1, 2, 3, 5]:
            self.cnns.append(nn.Sequential(
                SamePadConv1D(hidden_size * 2, cnn_channel, kernel_size=ks),
                nn.ReLU(),
                # nn.BatchNorm1d(cnn_channel),
                nn.Dropout(dropout_r)
            ))

        self.similarity = nn.CosineSimilarity(dim=1)

    def forward(self, question, answer):
        question, question_length = question
        answer, answer_length = answer
        # 合并问题和答案，加速网络运行
        bs = question.size(0)
        sent = torch.cat([question, answer], dim=0)
        sent_len = torch.cat([question_length, answer_length], dim=0)
        sent = fit_seq_max_len(sent, sent_len)

        embed = self.embed(sent)  # (bs, sent, vector)

        gru_sent, hn_sent = auto_rnn_bigru(self.gru_qa, embed, sent_len)
        gru_sent = gru_sent.transpose(1, 2)  # (B, T, D) => (B, D, T)
        sent_cnns = torch.cat([cnn(gru_sent) for cnn in self.cnns], dim=1)
        sent_pool = torch.max(sent_cnns, dim=2)[0]

        sim = self.similarity(sent_pool[:bs], sent_pool[bs:])
        return sim

