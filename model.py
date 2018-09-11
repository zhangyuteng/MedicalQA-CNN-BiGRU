# -*- coding: utf-8 -*-
import torch
from torch import nn
import torch.nn.functional as F
from utility import auto_rnn_bilstm, fit_seq_max_len


class QA_StackMultiCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, embed_weight=None):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, _weight=embed_weight)
        self.conv3 = nn.Conv1d(embed_dim, 800, 3)
        self.conv4 = nn.Conv1d(embed_dim, 800, 4)
        self.stack_conv = nn.Sequential(
            nn.Conv1d(embed_dim, 800, 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(800, 800, 3),
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


class StackBiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, mlp_d=1600, dropout_r=0.1, embed_weight=None):
        super(StackBiLSTM, self).__init__()
        self.embed = nn.Embedding(vocab_size, embedding_dim, _weight=embed_weight)

        self.lstm_1 = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size[0],
                              num_layers=1, batch_first=True, bidirectional=True)

        self.lstm_2 = nn.LSTM(input_size=(embedding_dim + hidden_size[0] * 2), hidden_size=hidden_size[1],
                              num_layers=1, batch_first=True, bidirectional=True)

        self.lstm_3 = nn.LSTM(input_size=(embedding_dim + (hidden_size[0] + hidden_size[1]) * 2),
                              hidden_size=hidden_size[2], num_layers=1, batch_first=True, bidirectional=True)

        self.mlp_1 = nn.Linear(hidden_size[2] * 2 * 4, mlp_d)
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

        embed = self.embed(sent)  # (bs, sent, vector)
        lstm_layer1_out = auto_rnn_bilstm(self.lstm_1, embed, sent_len)  # lstm_out (bs, T=sent, D=2*hidden_size)

        layer2_in = torch.cat([embed, lstm_layer1_out], dim=2)
        lstm_layer2_out = auto_rnn_bilstm(self.lstm_2, layer2_in, sent_len)

        layer3_in = torch.cat([embed, lstm_layer1_out, lstm_layer2_out], dim=2)
        lstm_layer3_out = auto_rnn_bilstm(self.lstm_3, layer3_in, sent_len)

        lstm_layer3_maxout = torch.max(lstm_layer3_out, dim=1)[0]  # (bs, D)

        features = torch.cat([lstm_layer3_maxout[:bs], lstm_layer3_maxout[bs:],
                              torch.abs(lstm_layer3_maxout[:bs] - lstm_layer3_maxout[bs:]),
                              lstm_layer3_maxout[:bs] * lstm_layer3_maxout[bs:]],
                             dim=1)

        out = self.classifier(features)
        return out
