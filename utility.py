import numpy as np
import random

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