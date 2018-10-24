#!/usr/bin/env bash
set -e
## blue to echo
function blue(){
    echo -e "\033[35m$1\033[0m"
}
## green to echo
function green(){
    echo -e "\033[32m$1\033[0m"
}
## Error to warning with blink
function bred(){
    echo -e "\033[31m\033[01m\033[05m$1\033[0m"
}
## Error to warning with blink
function byellow(){
    echo -e "\033[33m\033[01m\033[05m$1\033[0m"
}
## Error
function red(){
    echo -e "\033[31m\033[01m$1\033[0m"
}
## warning
function yellow(){
    echo -e "\033[33m\033[01m$1\033[0m"
}
byellow '-----------------'
byellow '卷积核数量的对比实验'
byellow '-----------------'

#blue 'Stack-CNN'
#for num in 100 300 700
#do
#    common="python train.py --train-rate 1 --device 1 --arch stack --batch-size 128 --lr 0.001 --optimizer adam --stack-kernel-sizes 3,4 --stack-out-channels ${num},${num} --topk 1 --epoch 5"
#    green "$common"
#    ${common}
#done

blue 'Multi-CNN'
for num in 300 700 900
do
    common="python train.py --train-rate 1 --device 1 --arch multi --batch-size 128 --lr 0.001 --optimizer adam --multi-kernel-sizes 1,2,3,5 --multi-out-channels ${num},${num},${num},${num} --topk 1 --epoch 5"
    green "$common"
    ${common}
done

#blue 'Stack-CNN+Multi-CNN'
#for num in 100 300 700
#do
#    common="python train.py --train-rate 1 --device 0 --arch stack_multi --batch-size 128 --lr 0.001 --optimizer adam --multi-kernel-sizes 1,2,3,5 --multi-out-channels ${num},${num},${num},${num} --topk 1 --epoch 5"
#    green "$common"
#    ${common}
#done

#blue 'SingleCNN'
#for num in 600 800 1000
#do
#    common="python train.py --train-rate 1 --device 1 --arch multi --batch-size 128 --lr 0.001 --optimizer adam --multi-kernel-sizes 2 --multi-out-channels ${num} --topk 1 --epoch 5"
#    green "$common"
#    ${common}
#done