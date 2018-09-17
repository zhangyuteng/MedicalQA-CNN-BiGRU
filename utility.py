import random
import os
import torch
from torch import nn
import torchtext
import logging
import time
import pandas as pd


def get_accuracy(qids, predictions, labels, topk=1):
    tmp = list(zip(qids, predictions, labels))
    random.shuffle(tmp)
    qids, predictions, labels = zip(*tmp)
    qres = {}
    for i,qid in enumerate(qids):
        pre = predictions[i]
        label = labels[i]
        if qid in qres:
            qres[qid]['labels'].append(label)
            qres[qid]['predictions'].append(pre)
        else:
            qres[qid] = {'labels': [label], 'predictions': [pre]}
    correct = 0
    for qid,res in qres.items():
        label_index = [i for i,v in enumerate(res['labels']) if v == 1]
        pre_index = sorted(enumerate(res['predictions']), key=lambda x:x[1], reverse=True)[:topk]
        is_correct = [(k,v) for k,v in pre_index if k in label_index]
        if len(is_correct) > 0:
            correct += 1
    return correct / len(qres)


def pack_for_rnn_seq(embed_seq, seq_len):
    """
    the format of seq is batch first.
    :param embed_seq:
    :param seq_len:
    :return:
    """
    _, idx_sort = torch.sort(seq_len, dim=0, descending=True)
    _, idx_unsort = torch.sort(idx_sort, dim=0)

    embed_seq = embed_seq.index_select(0, idx_sort)
    sent_len = seq_len[idx_sort]
    packed_seq = nn.utils.rnn.pack_padded_sequence(embed_seq, sent_len, batch_first=True)
    return packed_seq, idx_unsort


def unpack_from_rnn_seq(packed_seq, idx_unsort, total_length=None):
    unpacked_seq, _ = nn.utils.rnn.pad_packed_sequence(packed_seq, batch_first=True, total_length=total_length)
    unsort_seq = unpacked_seq.index_select(0, idx_unsort)
    return unsort_seq


def fit_seq_max_len(seq, seq_len):
    '''
    自动匹配序列的最大长度
    由于使用torchtext中的fix_length函数，得到的序列长度都是400维，而大部分情况下，
    一个batch的最大长度往往不够400维，使用此函数能将序列的最大长度固定到一个batch中的最大长度。
    :param embed_seq:
    :param seq_len:
    :return:
    '''
    packed_seq, idx_unsort = pack_for_rnn_seq(seq, seq_len)
    seq = unpack_from_rnn_seq(packed_seq, idx_unsort)
    return seq


def auto_rnn_bilstm(lstm, embed_seq, lengths, fix_length=False):
    packed_seq, idx_unsort = pack_for_rnn_seq(embed_seq, lengths)
    output, (hn, cn) = lstm(packed_seq)
    total_length = None
    if fix_length:
        total_length = embed_seq.size(1)
    unpacked_output = unpack_from_rnn_seq(output, idx_unsort, total_length)
    hn_unsort = hn.index_select(1, idx_unsort)
    cn_unsort = cn.index_select(1, idx_unsort)
    return unpacked_output, (hn_unsort, cn_unsort)


def auto_rnn_bigru(gru, embed_seq, lengths, fix_length=False):
    """
    自动对变长序列做pack和pad操作
    :param gru:
    :param embed_seq:
    :param lengths:
    :param fix_length: 是否固定输出的结果，默认会变长序列中最长那个序列长度
    :return:
    """
    packed_seq, idx_unsort = pack_for_rnn_seq(embed_seq, lengths)
    output, hn = gru(packed_seq)
    total_length = None
    if fix_length:
        total_length = embed_seq.size(1)
    unpacked_output = unpack_from_rnn_seq(output, idx_unsort, total_length)
    hn_unsort = hn.index_select(1, idx_unsort)
    return unpacked_output, hn_unsort


def get_logger(output_dir):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s')

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    ch = logging.FileHandler(f'{output_dir}/run.log', mode='a')
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


def get_dataset(args, logger):
    start = time.time()
    dataset_dir = args.dataset_dir
    fix_length = args.fix_length
    question = pd.read_csv(f'{dataset_dir}/questions.csv', index_col='que_id')
    answer = pd.read_csv(f'{dataset_dir}/answers.csv', index_col='ans_id')
    train_candidates = pd.read_csv(f'{dataset_dir}/train_candidates.txt')
    dev_candidates = pd.read_csv(f'{dataset_dir}/dev_candidates.txt', header=1,
                                 names=['question_id', 'ans_id', 'num', 'label'])
    test_candidates = pd.read_csv(f'{dataset_dir}/test_candidates.txt', header=1,
                                  names=['question_id', 'ans_id', 'num', 'label'])
    # 记录问题和答案的id到文本的映射
    qid2text = {index: item['content'] for index, item in question.iterrows()}
    aid2text = {index: item['content'] for index, item in answer.iterrows()}
    logger.info(f'load data use {time.time()-start}')
    start = time.time()
    # 定义数据loader
    ID_FIELD = torchtext.data.Field(sequential=False, use_vocab=False, batch_first=True)
    TEXT_FIELD = torchtext.data.Field(batch_first=True, tokenize=lambda x: list(x), fix_length=fix_length,
                                      include_lengths=True)
    LABEL_FIELD = torchtext.data.Field(sequential=False, use_vocab=False, batch_first=True)
    # 问题
    examples = []
    fields = [('id', ID_FIELD), ('content', TEXT_FIELD)]
    for que_id, content in question.content.items():
        example_list = [que_id, content]
        example = torchtext.data.Example.fromlist(example_list, fields)
        examples.append(example)
    question_dataset = torchtext.data.Dataset(examples, fields)
    logger.info(f'buid question data use {time.time()-start}')
    start = time.time()
    # 答案
    examples = []
    fields = [('id', ID_FIELD), ('content', TEXT_FIELD)]
    for ans_id, content in answer.content.items():
        example_list = [ans_id, content]
        example = torchtext.data.Example.fromlist(example_list, fields)
        examples.append(example)
    answer_dataset = torchtext.data.Dataset(examples, fields)
    logger.info(f'buid answer data use {time.time()-start}')
    start = time.time()
    # 训练集
    if args.train_rate < 1:
        # 使用一定数量的训练集，加快速度
        train_candidates = train_candidates.head(round(len(train_candidates) * args.train_rate))
    examples = []
    fields = [('id', ID_FIELD), ('question', TEXT_FIELD), ('pos_answer', TEXT_FIELD), ('neg_answer', TEXT_FIELD)]
    for question_id, pos_ans_id, neg_ans_id in zip(train_candidates.question_id.values,
                                                   train_candidates.pos_ans_id.values,
                                                   train_candidates.neg_ans_id.values):
        example_list = [question_id, qid2text[question_id], aid2text[pos_ans_id], aid2text[neg_ans_id]]
        example = torchtext.data.Example.fromlist(example_list, fields)
        examples.append(example)
    train_dataset = torchtext.data.Dataset(examples, fields)
    logger.info(f'buid train data use {time.time()-start}')
    start = time.time()
    # 验证集
    examples = []
    fields = [('id', ID_FIELD), ('question', TEXT_FIELD), ('answer', TEXT_FIELD), ('label', LABEL_FIELD)]
    for question_id, ans_id, label in zip(dev_candidates.question_id.values, dev_candidates.ans_id.values,
                                          dev_candidates.label.values):
        example_list = [question_id, qid2text[question_id], aid2text[ans_id], label]
        example = torchtext.data.Example.fromlist(example_list, fields)
        examples.append(example)
    dev_dataset = torchtext.data.Dataset(examples, fields)
    logger.info(f'buid dev data use {time.time()-start}')
    start = time.time()
    # 测试集
    examples = []
    fields = [('id', ID_FIELD), ('question', TEXT_FIELD), ('answer', TEXT_FIELD), ('label', LABEL_FIELD)]
    for question_id, ans_id, label in zip(test_candidates.question_id.values, test_candidates.ans_id.values,
                                          test_candidates.label.values):
        example_list = [question_id, qid2text[question_id], aid2text[ans_id], label]
        example = torchtext.data.Example.fromlist(example_list, fields)
        examples.append(example)
    test_dataset = torchtext.data.Dataset(examples, fields)
    logger.info(f'buid test data use {time.time()-start}')
    start = time.time()
    # 载入预训练的词向量
    pre_vectors = None
    if args.word_vectors:
        vector_dir, vector_file = os.path.split(args.word_vectors)
        pre_vectors = torchtext.vocab.Vectors(name=vector_file, cache=vector_dir)
        logger.info(f'load vector use {time.time()-start}')
    start = time.time()
    # 构建词表 时间较长 3分钟左右
    TEXT_FIELD.build_vocab(question_dataset, answer_dataset, vectors=pre_vectors)
    logger.info(f'buid vocab use {time.time()-start}')
    vocab = TEXT_FIELD.vocab  # 词表
    vectors = TEXT_FIELD.vocab.vectors  # 预训练的词向量
    return train_dataset, dev_dataset, test_dataset, vocab, vectors