# -*- coding: utf-8 -*-
import argparse
import os
import pprint
import random
import time

import numpy as np
import torchtext
import torch
from tensorboardX import SummaryWriter
from tqdm import tqdm

from model import *
from utility import get_accuracy, get_logger, get_dataset
from args import parse_args


def train_epoch(epoch, data_loader, model, optimizer, loss_fn, device):
    """
    进行一次迭代
    """
    model.train()
    pbar = tqdm(data_loader, desc='Train Epoch {}'.format(epoch))
    total_loss = []
    for batch_idx, batch in enumerate(pbar):
        optimizer.zero_grad()

        target = torch.ones(batch.batch_size, requires_grad=True).to(device)
        pos_sim = model(batch.question, batch.pos_answer)
        neg_sim = model(batch.question, batch.neg_answer)

        # bs = batch.batch_size
        # question = batch.question[0].repeat(2, 1)
        # question_len = batch.question[1].repeat(2)
        # answer = torch.cat([batch.pos_answer[0], batch.neg_answer[0]], dim=0)
        # answer_len = torch.cat([batch.pos_answer[1], batch.neg_answer[1]], dim=0)
        # sim = model((question,question_len), (answer,answer_len))
        # pos_sim, neg_sim = sim[:bs], sim[bs:]
        # target = torch.ones(bs, requires_grad=True).to(device)

        loss = loss_fn(pos_sim, neg_sim, target)
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
    if torch.cuda.is_available() and args.device >= 0:
        # 开启这个flag需要保证输入数据的维度不变，不然每次cudnn都要重新优化，反而更加耗时
        # 现在RNN部分输入会进行fit length，CNN那里可以启用这个参数
        if args.arch in ['stack_multi', 'norm_stack_multi', 'stack_multi_atten', 'ap_stack_multi']:
            torch.backends.cudnn.benchmark = True
    # 输出目录
    if args.resume_snapshot:
        # 判断文件是否存在
        assert os.path.exists(args.resume_snapshot), f'{args.resume_snapshot} don"t exist!'
        model_dir, model_file = os.path.split(args.resume_snapshot)
        output_dir, _ = os.path.split(model_dir)
    else:
        base_dir = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
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
        model = StackMultiCNN(vocab_size=len(vocab), embed_dim=vectors.size(1), embed_weight=vectors).to(device)
    elif args.arch == 'norm_stack_multi':
        model = NormStackMultiCNN(vocab_size=len(vocab), embed_dim=vectors.size(1), sent_length=args.fix_length,
                                  embed_weight=vectors).to(device)
    elif args.arch == 'stack_multi_atten':
        model = QA_StackMultiAttentionCNN(vocab_size=len(vocab), embed_dim=vectors.size(1), embed_weight=vectors).to(
            device)
    elif args.arch == 'ap_stack_multi':
        model = QA_AP_StackMultiCNN(vocab_size=len(vocab), embed_dim=vectors.size(1), embed_weight=vectors).to(
            device)
    elif args.arch == 'bilstm':
        assert args.hidden_size.find(',') == -1, '--hidden-size must be a int for LSTM model'
        hidden_size = int(args.hidden_size)
        model = BiLSTM(vocab_size=len(vocab), embedding_dim=vectors.size(1), hidden_size=hidden_size,
                       dropout_r=args.dropout, embed_weight=vectors).to(device)
    elif args.arch == 'stack_bilstm':
        hidden_size = [int(i) for i in args.hidden_size.split(',')]
        model = StackBiLSTM(vocab_size=len(vocab), embedding_dim=vectors.size(1), hidden_size=hidden_size,
                            mlp_d=args.mlp_d, dropout_r=args.dropout, embed_weight=vectors).to(device)
    elif args.arch == 'bigru':
        assert args.hidden_size.find(',') == -1, '--hidden-size must be a int for BiLSTM/BiGRU model'
        hidden_size = int(args.hidden_size)
        model = BiGRU(vocab_size=len(vocab), embedding_dim=vectors.size(1), hidden_size=hidden_size,
                      dropout_r=args.dropout, embed_weight=vectors).to(device)
    elif args.arch == 'share_bigru':
        assert args.hidden_size.find(',') == -1, '--hidden-size must be a int for BiLSTM/BiGRU model'
        hidden_size = int(args.hidden_size)
        model = ShareBiGRU(vocab_size=len(vocab), embedding_dim=vectors.size(1), hidden_size=hidden_size,
                           dropout_r=args.dropout, embed_weight=vectors).to(device)
    elif args.arch == 'stack_bigru':
        hidden_size = [int(i) for i in args.hidden_size.split(',')]
        model = StackBiGRU(vocab_size=len(vocab), embedding_dim=vectors.size(1), hidden_size=hidden_size, mlp_d=args.mlp_d,
                           sent_max_length=args.fix_length, dropout_r=args.dropout, embed_weight=vectors).to(device)
    elif args.arch == 'cnn_share_bigru':
        assert args.hidden_size.find(',') == -1, '--hidden-size must be a int for BiLSTM/BiGRU model'
        hidden_size = int(args.hidden_size)
        model = CnnShareBiGRU(vocab_size=len(vocab), embedding_dim=vectors.size(1), hidden_size=hidden_size,
                              cnn_channel=args.cnn_channel, dropout_r=args.dropout, embed_weight=vectors).to(device)
    else:
        raise ValueError("--arch is unknown")
    # 为特定模型指定特殊的优化函数
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr)
    elif args.optimizer == 'adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lr)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    else:
        raise ValueError("--optimizer is unknown")
    loss_fn = torch.nn.MarginRankingLoss(margin=args.margin)
    architecture = model.__class__.__name__
    # 载入以训练的数据
    if args.resume_snapshot:
        state = torch.load(args.resume_snapshot)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        epoch = state['epoch']
        start_epoch = state['epoch'] + 1
        if 'best_dev_score' in state:
            # 适配旧版本保存的模型参数
            dev_acc = state['best_dev_score']
            test_acc = 0
        else:
            dev_acc = state['dev_accuracy']
            test_acc = state['test_accuracy']
        logger.info(f"load state {args.resume_snapshot}, dev accuracy {dev_acc}, test accuracy {test_acc}")
    # 记录参数
    with open(f'{output_dir}/arguments.csv', 'a') as f:
        for k, v in vars(args).items():
            f.write(f'{k},{v}\n')
    # 将日志写入到TensorBoard中
    writer = SummaryWriter(output_dir)
    # 记录模型的计算图
    try:
        q = torch.randint_like(torch.Tensor(1, args.fix_length), 2, 100, dtype=torch.long)
        ql = torch.Tensor([args.fix_length]).type(torch.int)
        writer.add_graph(model, ((q, ql), (q, ql)))
    except Exception as e:
        logger.error("Failed to save model graph: {}".format(e))
        # exit()
    # 开始训练
    best_dev_score = -1  # 记录最优的结果
    best_test_score = -1  # 记录最优的结果
    prev_loss = 0
    # 自动调整学习率
    # TODO:暂不启用，Adam已经能够自动调整学习率了
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=args.lr_reduce_factor,
    #                                                        patience=args.patience, verbose=True)
    if not args.skip_training:
        for epoch in range(start_epoch, start_epoch + args.epochs):
            start_time = time.time()
            # train epoch
            loss = train_epoch(epoch, train_loader, model, optimizer, loss_fn, device)
            writer.add_scalar('train/loss', loss, epoch)
            logger.info(f'Train Epoch {epoch}: loss={loss}')
            # evaluate
            dev_accuracy = evaluate(dev_loader, model)
            logger.info(f'Evaluation metrices: dev accuracy = {100. * dev_accuracy}%')
            writer.add_scalar('dev/lr', optimizer.param_groups[0]['lr'], epoch)
            writer.add_scalar('dev/acc', dev_accuracy, epoch)
            # 进行测试
            test_accuracy = evaluate(test_loader, model)
            logger.info(f'Evaluation metrices: test accuracy = {100. * test_accuracy}%')
            writer.add_scalar('test/acc', test_accuracy, epoch)

            # 保存模型
            save_state = {'epoch': epoch, 'dev_accuracy': dev_accuracy, 'test_accuracy': test_accuracy,
                          'architecture': architecture, 'model': model.state_dict(),
                          'optimizer': optimizer.state_dict()}
            torch.save(save_state, f'{model_dir}/{architecture}_epoch_{epoch}.pth')
            logger.info('Save best model: epoch {}, dev accuracy {}, test accuracy'.format(epoch, dev_accuracy, test_accuracy))
            # 计算模型运行时间
            duration = time.time() - start_time
            logger.info('Epoch {} finished in {:.2f} minutes'.format(epoch, duration / 60))

            if abs(prev_loss - loss) <= args.early_stopping:
                logger.info('Early stopping. Loss changed by less than {}.'.format(args.early_stopping))
                break
            prev_loss = loss
    else:
        # 进行测试
        test_accuracy = evaluate(test_loader, model)
        logger.info(f'Evaluation metrices: test accuracy = {100. * test_accuracy}%')
    # 保存embedding到tensorboard做可视化
    # writer.add_embedding(model.embed.weight.detach(), vocab.itos, global_step=epoch)


if __name__ == '__main__':
    run()
