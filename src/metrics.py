import torch
import numpy as np
from torchmetrics.classification import MultilabelF1Score
from torchmetrics.classification import MultilabelRecall
from torchmetrics.classification import MultilabelPrecision

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

f1 = MultilabelF1Score(num_labels=18)
f1 = f1.to(device)
recall = MultilabelRecall(num_labels=18)
recall = recall.to(device)
precision = MultilabelPrecision(num_labels=18)
precision = precision.to(device)

def metrics(results, truths):
    f1_score = f1(results, truths)
    recall = recall(results, truths)
    precision = precision(results, truths)
    acc = multiclass_acc(results, truths)

    return acc, prec, recall, f1_score

def multiclass_acc(results, truths):
    preds = results.view(-1).cpu().detach().numpy()
    truth = truths.view(-1).cpu().detach().numpy()

    return np.sum(preds == truths) / float(len(truths))