
from scipy.stats import kendalltau

import torch
from torch import Tensor
from torchmetrics import Metric


class Accuracy(Metric):
    def __init__(self, type_=None):
        super().__init__()
        self.add_state("correct", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.), dist_reduce_fx="sum")

        self.type = type_

    def update(self, preds, gt_orders, masks):
        for (pred, gt_order, mask) in zip(preds, gt_orders, masks):
            gt_order = gt_order[: mask.sum()]

            if self.type == 's':
                if len(gt_order) > 3:
                    continue
            if self.type == 'l':
                if len(gt_order) <= 3:
                    continue

            if self.type == 'pos':
                natural_order = torch.arange(len(gt_order)).to(gt_order.device)
                if not gt_order.tolist() == natural_order.tolist():
                    continue
            if self.type == 'neg':
                natural_order = torch.arange(len(gt_order)).to(gt_order.device)
                if gt_order.tolist() == natural_order.tolist():
                    continue

            final_pred = gt_order[pred]
            natural_order = torch.arange(len(gt_order)).to(gt_order.device)
            self.correct += torch.sum(final_pred == natural_order)
            self.total += len(gt_order)

    def compute(self):
        return self.correct / self.total


class Kendall_Tau(Metric):
    def __init__(self, type_=None):
        super().__init__()
        self.add_state("kendall_score", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.), dist_reduce_fx="sum")

        self.type = type_

    def update(self, preds, gt_orders, masks):
        # Note that the input pred and target are for a single collection instance.

        for (pred, gt_order, mask) in zip(preds, gt_orders, masks):
            gt_order = gt_order[: mask.sum()]

            if self.type == 's':
                if len(gt_order) > 3:
                    continue
            if self.type == 'l':
                if len(gt_order) <= 3:
                    continue

            if self.type == 'pos':
                natural_order = torch.arange(len(gt_order)).to(gt_order.device)
                if not gt_order.tolist() == natural_order.tolist():
                    continue
            if self.type == 'neg':
                natural_order = torch.arange(len(gt_order)).to(gt_order.device)
                if gt_order.tolist() == natural_order.tolist():
                    continue

            final_pred = gt_order[pred]
            natural_order = torch.arange(len(gt_order)).to(gt_order.device)
            kt = kendall_tau(final_pred.tolist(), natural_order.tolist())

            self.kendall_score += kt
            self.total += 1

    def compute(self):
        return self.kendall_score / self.total


class PMR(Metric):
    # perfect matching rate
    def __init__(self, type_=None):
        super().__init__()
        self.add_state("correct", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.), dist_reduce_fx="sum")

        self.type = type_

    def update(self, preds, gt_orders, masks):
        assert len(preds) == len(gt_orders) == len(masks)

        for (pred, gt_order, mask) in zip(preds, gt_orders, masks):
            gt_order = gt_order[: mask.sum()]

            if self.type == 's':
                if len(gt_order) > 3:
                    continue
            if self.type == 'l':
                if len(gt_order) <= 3:
                    continue

            if self.type == 'pos':
                natural_order = torch.arange(len(gt_order)).to(gt_order.device)
                if not gt_order.tolist() == natural_order.tolist():
                    continue
            if self.type == 'neg':
                natural_order = torch.arange(len(gt_order)).to(gt_order.device)
                if gt_order.tolist() == natural_order.tolist():
                    continue

            final_pred = gt_order[pred]
            natural_order = torch.arange(len(gt_order)).to(gt_order.device)
            if final_pred.equal(natural_order):
                self.correct += 1
            self.total += 1

    def compute(self):
        return self.correct / self.total


class RougeS(Metric):
    # perfect matching rate
    def __init__(self, type_=None):
        super().__init__()
        self.add_state("rouges", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.), dist_reduce_fx="sum")

        self.type = type_

    def update(self, preds, gt_orders, masks):
        assert len(preds) == len(gt_orders) == len(masks)

        for (pred, gt_order, mask) in zip(preds, gt_orders, masks):
            gt_order = gt_order[: mask.sum()]

            if self.type == 's':
                if len(gt_order) > 3:
                    continue
            if self.type == 'l':
                if len(gt_order) <= 3:
                    continue

            if self.type == 'pos':
                natural_order = torch.arange(len(gt_order)).to(gt_order.device)
                if not gt_order.tolist() == natural_order.tolist():
                    continue
            if self.type == 'neg':
                natural_order = torch.arange(len(gt_order)).to(gt_order.device)
                if gt_order.tolist() == natural_order.tolist():
                    continue

            final_pred = gt_order[pred]
            natural_order = torch.arange(len(gt_order)).to(gt_order.device)

            rouges = rouge_s(natural_order.tolist(), final_pred.tolist())
            self.rouges += rouges
            self.total += 1

    def compute(self):
        return self.rouges / self.total


class LCS(Metric):
    def __init__(self, type_=None):
        super().__init__()
        self.add_state("lcs", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.), dist_reduce_fx="sum")

        self.type = type_

    def update(self, preds, gt_orders, masks):
        for (pred, gt_order, mask) in zip(preds, gt_orders, masks):
            gt_order = gt_order[: mask.sum()]

            if self.type == 's':
                if len(gt_order) > 3:
                    continue
            if self.type == 'l':
                if len(gt_order) <= 3:
                    continue

            if self.type == 'pos':
                natural_order = torch.arange(len(gt_order)).to(gt_order.device)
                if not gt_order.tolist() == natural_order.tolist():
                    continue
            if self.type == 'neg':
                natural_order = torch.arange(len(gt_order)).to(gt_order.device)
                if gt_order.tolist() == natural_order.tolist():
                    continue

            final_pred = gt_order[pred]
            natural_order = torch.arange(len(gt_order)).to(gt_order.device)
            self.lcs += longest_common_subsequence(final_pred.tolist(), natural_order.tolist())
            self.total += len(gt_order)

    def compute(self):
        return self.lcs / self.total


# Code from https://github.com/fabrahman/ReBART/blob/main/eval/evaluation.py

def kendall_tau(order, ground_truth):
    """
    Computes the kendall's tau metric
    between the predicted sentence order and true order

    Input:
            order: list of ints denoting the predicted output order
            ground_truth: list of ints denoting the true sentence order

    Returns:
            kendall's tau - float
    """

    if len(ground_truth) <= 1:
        return 1.0

    reorder_dict = {}

    for i in range(len(ground_truth)):
        reorder_dict[ground_truth[i]] = i

    new_order = [0] * len(order)
    for i in range(len(new_order)):
        if order[i] in reorder_dict.keys():
            new_order[i] = reorder_dict[order[i]]

    corr, _ = kendalltau(new_order, list(range(len(order))))
    return corr


def longest_common_subsequence(X, Y):
    """
    Computes the longest common subsequence between two sequences

    Input:
            X: list of ints
            Y: list of ints

    Returns:
            LCS: int
    """
    m = len(X)
    n = len(Y)

    L = [[None] * (n + 1) for i in range(m + 1)]

    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                L[i][j] = 0
            elif X[i - 1] == Y[j - 1]:
                L[i][j] = L[i - 1][j - 1] + 1
            else:
                L[i][j] = max(L[i - 1][j], L[i][j - 1])

    return L[m][n]


def skip_bigrams(arr):
    """
    Utility function for Rouge-S metric
    """
    bigrams = set()
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            bigrams.add((arr[i], arr[j]))
    return bigrams


def rouge_s(gold, pred):
    """
    Rouge-S metric between two sequence

    Input:
            gold: list of ints
            pred: list of ints

    Returns:
            Rouge-S score
    """

    if len(gold) == 1 or len(pred) == 1:
        return int(gold[0] == pred[0])

    gold_bigrams = skip_bigrams(gold)
    pred_bigrams = skip_bigrams(pred)

    total = len(gold_bigrams)
    same = len(gold_bigrams.intersection(pred_bigrams))
    return same / (total + 1e-12)
