# -*- coding: utf-8 -*-
import os
import itertools
import random
import argparse
import logging
import pprint
import time
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
from tensorboardX import SummaryWriter
import torchtext
from utility import get_accuracy
from model import *


def parse_args():
    parser = argparse.ArgumentParser(
        'Train a Chinese Medical Question Answer Matching model.'
    )
    parser.add_argument("--arch",
                        choices=['stack_multi', 'stack_multi_atten', 'ap_stack_multi', 'lstm'],
                        default='QA_StackMultiCNN',
                        help="model architecture to use (default: stack_multi)")
    parser.add_argument("--dataset-dir",
                        type=str,
                        default='./cMedQA',
                        help="dataset directory, default is cMedQA")
    parser.add_argument("--out-dir",
                        type=str,
                        default='./output',
                        help="output directory, default is current directory")
    parser.add_argument("--remark", type=str, default='', help='remark')
    parser.add_argument('--seed', type=int, default=1234,
                        help="random seed, default 1234,"
                             "set seed to -1 if need a random seed"
                             "between 1 and 100000")
    parser.add_argument('--train-rate', type=float, default=1,
                        help='Use the rate of the training data set (default: 1)')
    parser.add_argument('--tensorboard', action='store_true', default=False,
                        help='use TensorBoard to visualize training (default: false)')
    parser.add_argument('--word_vectors', help='word vectors file',
                        default='/home/ailab/zhangyuteng/Castor-data/embeddings/ChineseVector/WordCharacter/'
                                'sgns.target.word-character.char1-2.dynwin5.thr10.neg5.dim300.iter5')
    parser.add_argument('--device', type=int, default=0, help='GPU device, -1 for CPU (default: 0)')
    parser.add_argument('--fix-length', type=int, default=400, help='limit sentence length (default: 400)')
    parser.add_argument('--batch-size', type=int, default=128, help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train (default: 10)')
    parser.add_argument('--optimizer', type=str, default='adam', help='optimizer to use: adam or sgd (default: adam)')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 0.001)')
    parser.add_argument('--lr-reduce-factor', type=float, default=0.3,
                        help='learning rate reduce factor after plateau (default: 0.3)')
    # 当监测值不再改善时，该回调函数将中止训练
    parser.add_argument('--early-stopping', type=float, default=0.00002,
                        help='Stop training when a monitored quantity has stopped improving.(default: 0.00002)')
    # 如果patience个epoch后
    parser.add_argument('--patience', type=float, default=2,
                        help='learning rate patience after seeing plateau (default: 2)')
    # 载入模型
    parser.add_argument('--resume-snapshot', type=str, default=None)
    parser.add_argument('--skip-training', help='will load pre-trained model', action='store_true', default=False)

    arguments = parser.parse_args()
    assert 0 < arguments.train_rate <= 1, '--train-rate must be greater than 0 and less than or equal to 1'
    return arguments


def get_logger(output_dir):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s')

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    ch = logging.FileHandler(f'{output_dir}/run.log')
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


def train_epoch(epoch, data_loader, model, optimizer, loss_fn, device):
    """
    进行一次迭代
    """
    model.train()
    pbar = tqdm(data_loader, desc='Train Epoch {}'.format(epoch))
    total_loss = []
    for batch_idx, batch in enumerate(pbar):
        optimizer.zero_grad()

        batch_size = batch.batch_size
        target = torch.ones(batch_size, requires_grad=True).to(device)
        question = batch.question[0].repeat(2, 1)
        question_length = batch.question[1].repeat(2)
        answer = torch.cat([batch.pos_answer[0], batch.neg_answer[0]], dim=0)
        answer_length = torch.cat([batch.pos_answer[1], batch.neg_answer[1]], dim=0)
        sim = model((question, question_length), (answer, answer_length))

        loss = loss_fn(sim[:batch_size], sim[batch_size:], target)
        total_loss.append(loss.item())
        pbar.set_postfix(batch_loss=loss.item())
        loss.backward()
        optimizer.step()
    return np.mean(total_loss)


def evaluate(date_loader, model):
    """
    在dev上进行测试
    """
    model.eval()
    pbar = tqdm(date_loader, desc=f'Evaluate')
    # 记录预测结果，计算Top-1正确率
    qids = []
    predictions = []
    true_labels = []
    with torch.no_grad():
        for batch in pbar:
            qids.extend(batch.id.cpu().numpy())
            true_labels.extend(batch.label.cpu().numpy())
            output = model(batch.question, batch.answer)
            predictions.extend(output.cpu().numpy())

    accuracy = get_accuracy(qids, predictions, true_labels)
    return accuracy


def run():
    args = parse_args()
    # 初始化随机数种子，以便于复现实验结果
    start_epoch = 1
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device != -1:
        torch.cuda.manual_seed(args.seed)
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() and args.device >= 0 else 'cpu')
    # 输出目录
    base_dir = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    output_dir = os.path.join(args.out_dir, base_dir)
    model_dir = os.path.join(output_dir, 'save_model')
    os.makedirs(output_dir)  # 创建输出根目录
    os.makedirs(model_dir)
    # 输出参数
    logger = get_logger(output_dir)
    logger.info(pprint.pformat(vars(args)))
    logger.info(f'output dir is {output_dir}')
    # 获取数据集
    train_dataset, dev_dataset, test_dataset, vocab, vectors = get_dataset(args, logger)
    # 创建迭代器
    train_loader = torchtext.data.BucketIterator(train_dataset, args.batch_size, device=device, train=True,
                                                 shuffle=True, sort=False, repeat=False)
    dev_loader = torchtext.data.BucketIterator(dev_dataset, args.batch_size, device=device, train=False,
                                               shuffle=False, sort=False, repeat=False)
    test_loader = torchtext.data.BucketIterator(test_dataset, args.batch_size, device=device, train=False,
                                                shuffle=False, sort=False, repeat=False)
    # 创建模型，优化器，损失函数
    if args.arch == 'stack_multi':
        model = QA_StackMultiCNN(vocab_size=len(vocab), embed_dim=vectors.size(1), embed_weight=vectors).to(device)
    elif args.arch == 'stack_multi_atten':
        model = QA_StackMultiAttentionCNN(vocab_size=len(vocab), embed_dim=vectors.size(1), embed_weight=vectors).to(device)
    elif args.arch == 'ap_stack_multi':
        model = QA_AP_StackMultiCNN(vocab_size=len(vocab), embed_dim=vectors.size(1), embed_weight=vectors).to(
            device)
    elif args.arch == 'lstm':
        model = QA_LSTM(vocab_size=len(vocab), embed_dim=vectors.size(1), embed_weight=vectors).to(device)
    else:
        raise ValueError("--arch is unknown")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0)
    loss_fn = torch.nn.MarginRankingLoss(margin=0.05)
    architecture = model.__class__.__name__
    # 载入以训练的数据
    if args.resume_snapshot:
        state = torch.load(args.resume_snapshot)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        start_epoch = state['epoch'] + 1
        logger.info(f"load state {args.resume_snapshot}, the state best dev score = {state['best_dev_score']}")
    # 记录参数
    with open(f'{output_dir}/arguments.csv', 'w') as f:
        for k, v in vars(args).items():
            f.write(f'{k},{v}\n')
    # 将日志写入到TensorBoard中
    writer = SummaryWriter(output_dir)
    # 记录模型的计算图
    data_loader_iter = iter(dev_loader)
    batch = next(data_loader_iter)
    try:
        writer.add_graph(model, (batch.question, batch.answer))
    except Exception as e:
        logger.error("Failed to save model graph: {}".format(e))
        # exit()
    del data_loader_iter, batch
    # 开始训练
    best_dev_score = -1  # 记录最优的结果
    prev_loss = 0
    # 自动调整学习率
    # TODO:暂不启用，Adam已经能够自动调整学习率了
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=args.lr_reduce_factor,
                                                           patience=args.patience, verbose=True)
    if not args.skip_training:
        for epoch in range(start_epoch, start_epoch + args.epochs):
            start_time = time.time()
            # train epoch
            loss = train_epoch(epoch, train_loader, model, optimizer, loss_fn, device)
            writer.add_scalar('train/loss', loss, epoch)
            logger.info(f'Train Epoch {epoch}: loss={loss}')
            # evaluate
            accuracy = evaluate(dev_loader, model)
            logger.info(f'Evaluation metrices: dev accuracy = {100. * accuracy}%')
            writer.add_scalar('dev/lr', optimizer.param_groups[0]['lr'], epoch)
            writer.add_scalar('dev/acc', accuracy, epoch)

            duration = time.time() - start_time
            logger.info('Epoch {} finished in {:.2f} minutes'.format(epoch, duration / 60))
            if accuracy > best_dev_score:
                best_dev_score = accuracy
                # 保存模型
                save_state = {'epoch': epoch, 'best_dev_score': best_dev_score, 'architecture': architecture,
                              'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
                torch.save(save_state, f'{model_dir}/{architecture}_epoch_{epoch}.pth')

            if abs(prev_loss - loss) <= 0.00002:
                logger.info('Early stopping. Loss changed by less than 0.0002.')
                break
            prev_loss = loss

    accuracy = evaluate(test_loader, model)
    logger.info(f'Evaluation metrices: test accuracy = {100. * accuracy}%')
    writer.add_scalar('test/acc', accuracy, 1)


if __name__ == '__main__':
    run()
