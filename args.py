import argparse
import socket


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        'Train a Chinese Medical Question Answer Matching model.'
    )
    parser.add_argument("--arch",
                        choices=['stack', 'multi', 'stack_multi', 'bigru', 'bigru_cnn'
                                 # 'norm_stack_multi',
                                 # 'stack_multi_atten', 'ap_stack_multi',
                                 # 'bilstm', 'stack_bilstm',
                                 #  'stack_bigru'
                                 ],
                        default='bigru_cnn',
                        help="model architecture to use (default: bigru_cnn)")
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
    parser.add_argument('--topk', type=str, default='1',
                        help='evaluate top k (default: 1)')
    parser.add_argument('--tensorboard', action='store_true', default=True,
                        help='use TensorBoard to visualize training (default: false)')
    vector_path = None
    hostname = socket.gethostname()
    if hostname == 'ailab-pc':
        vector_path = '/home/ailab/zhangyuteng/Castor-data/embeddings/ChineseVector/WordCharacter/' \
                      'sgns.target.word-character.char1-2.dynwin5.thr10.neg5.dim300.iter5'
    elif hostname == 'sys51-Default-string':
        vector_path = './embedding/sgns.target.word-character.char1-2.dynwin5.thr10.neg5.dim300.iter5'
    required = False if vector_path is not None else True
    parser.add_argument('--word-vectors', help='word vectors file', required=required, default=vector_path)
    parser.add_argument('--device', type=int, default=0, help='GPU device, -1 for CPU (default: 0)')
    parser.add_argument('--margin', type=float, default=0.05, help='MarginRankingLoss')
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--fix-length', type=int, default=400, help='limit sentence length (default: 400)')
    parser.add_argument('--batch-size', type=int, default=128, help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train (default: 10)')
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['adam', 'rmsprop', 'adagrad', 'sgd'],
                        help='optimizer to use: adam or sgd (default: adam)')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 0.001)')
    # parser.add_argument('--lr-reduce-factor', type=float, default=0.3,
    #                     help='learning rate reduce factor after plateau (default: 0.3)')
    # 当监测值不再改善时，该回调函数将中止训练
    parser.add_argument('--early-stopping', type=float, default=-1,
                        help='Stop training when a monitored quantity has stopped improving.(default: -1)')
    # 如果patience个epoch后
    # parser.add_argument('--patience', type=float, default=2,
    #                     help='learning rate patience after seeing plateau (default: 2)')
    # 载入模型
    parser.add_argument('--resume-snapshot', type=str, default=None)
    parser.add_argument('--skip-training', help='will load pre-trained model', action='store_true', default=False)

    parser.add_argument('--stack-kernel-sizes', type=str, default='3,4')
    parser.add_argument('--stack-out-channels', type=str, default='500,500')
    parser.add_argument('--multi-kernel-sizes', type=str, default='1,2,3,5')
    parser.add_argument('--multi-out-channels', type=str, default='500,500,500,500')

    stack_lstm_group = parser.add_argument_group('BiGRU')
    stack_lstm_group.add_argument('--hidden-size', type=str, default='200')
    stack_lstm_group.add_argument('--mlp-d', type=int, default='1024')

    cnn_share_bigru_group = parser.add_argument_group('BiGRU-CNN')
    cnn_share_bigru_group.add_argument('--cnn-channel', type=int, default='500')

    arguments = parser.parse_args(args)
    assert 0 < arguments.train_rate <= 1, '--train-rate must be greater than 0 and less than or equal to 1'
    arguments.stack_kernel_sizes = [int(i) for i in arguments.stack_kernel_sizes.split(',')]
    arguments.stack_out_channels = [int(i) for i in arguments.stack_out_channels.split(',')]
    arguments.multi_kernel_sizes = [int(i) for i in arguments.multi_kernel_sizes.split(',')]
    arguments.multi_out_channels = [int(i) for i in arguments.multi_out_channels.split(',')]
    if isinstance(arguments.topk, int):
        arguments.topk = [arguments.topk]
    else:
        arguments.topk = [int(i) for i in arguments.topk.split(',')]
    return arguments
