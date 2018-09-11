# -*- coding: utf-8 -*-
import argparse
import os
import pprint
import random
import time

import numpy as np
import torchtext
from tensorboardX import SummaryWriter
from tqdm import tqdm

from model import *
from utility import get_accuracy, get_logger, get_dataset, parse_args


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
    elif args.arch == 'stack_lstm':
        hidden_size = [int(i) for i in args.hidden_size.split(',')]
        model = StackBiLSTM(vocab_size=len(vocab), embedding_dim=vectors.size(1), hidden_size=hidden_size,
                            dropout_r=args.dropout, embed_weight=vectors).to(device)
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
