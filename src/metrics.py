import torch
import numpy as np
from torchmetrics.classification import MultilabelF1Score
from torchmetrics.classification import MultilabelRecall
from torchmetrics.classification import MultilabelPrecision
from torchmetrics.classification import MultilabelAccuracy
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

f1 = MultilabelF1Score(num_labels=18, avarage='micro')
f1 = f1.to(device)
recall = MultilabelRecall(num_labels=18, avarage='micro')
recall = recall.to(device)
precision = MultilabelPrecision(num_labels=18, avarage='micro')
precision = precision.to(device)
accuracy = MultilabelAccuracy(num_labels=18, avarage='micro')
accuracy = accuracy.to(device)

def metrics(results, truths):
    f1_score = f1(results, truths)
    recall_score = recall(results, truths)
    precision_score = precision(results, truths)
    acc_score = accuracy(results, truths)

    return acc_score, precision_score, recall_score, f1_score

# def multiclass_acc(results, truths):
#     preds = results.view(-1).cpu().detach().numpy()
#     truth = truths.view(-1).cpu().detach().numpy()

#     preds = np.where(preds > 0.5, 1, 0)
#     truth = np.where(truth > 0.5, 1, 0)
    
#     return np.sum(preds == truths) / float(len(truths))