import torch
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


def accuracy(output, target):
    pred = torch.argmax(output, dim=1)
    return accuracy_score(target, pred)


def precision(output, target):
    pred = torch.argmax(output, dim=1)
    return precision_score(target, pred, average='weighted', zero_division=True)


def recall(output, target):
    pred = torch.argmax(output, dim=1)
    return recall_score(target, pred, average='weighted', zero_division=True)


def f1(output, target):
    pred = torch.argmax(output, dim=1)
    return f1_score(target, pred, average='weighted')
