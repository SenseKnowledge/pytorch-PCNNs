# -*- coding: utf-8 -*
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_auc_score
from sklearn.preprocessing import label_binarize
import torch


class PRScore(torch.nn.Module):

    def __init__(self, num_classes, ignore_ids=None):
        super(PRScore, self).__init__()
        self.num_classes = num_classes
        self.ignore_id = ignore_ids if isinstance(ignore_ids, (tuple, list)) else [0]

    def forward(self, input, target):

        input = input.cpu().numpy()
        target = target.cpu().numpy()
        target = label_binarize(target, classes=list(range(self.num_classes)))

        precision = dict()
        recall = dict()
        average_precision = dict()

        for i in range(self.num_classes):
            if i in self.ignore_id:
                continue

            precision[i], recall[i], _ = precision_recall_curve(target[:, i], input[:, i])
            average_precision[i] = average_precision_score(target[:, i], input[:, i])

        precision["micro"], recall["micro"], _ = precision_recall_curve(target.ravel(), input.ravel())

        mask = [i for i in range(self.num_classes) if i not in self.ignore_id]
        average_precision["micro"] = average_precision_score(target[:, mask], input[:, mask], average="micro")

        roc = roc_auc_score(target[:, mask], input[:, mask], average="micro", multi_class='ovr')

        return roc, average_precision, precision, recall

    def plot_pr_curve(self, average_precision, recall, precision):
        import matplotlib.pyplot as plt

        plt.figure()
        plt.step(recall['micro'], precision['micro'], color='b', alpha=0.2,
                 where='post')
        plt.fill_between(recall["micro"], precision["micro"], step='post', alpha=0.2,
                         color='b')

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title(
            'Average precision score, micro-averaged over all classes: AP={0:0.2f}'.format(average_precision["micro"]))


from collections import Counter


class Score:

    def __init__(self, ignore_id=0):
        self.origins = []
        self.founds = []
        self.rights = []
        self.probs = []

        self.ignore_id = ignore_id

    def update(self, input, target):
        for y_pred, y_true in zip(input, target):

            y_pred, y_true = int(y_pred), int(y_true)
            self.founds.append(y_pred)
            self.origins.append(y_true)
            if y_pred == y_true:
                self.rights.append(y_pred)

    def compute(self):
        info = {}
        origins = Counter(self.origins)
        founds = Counter(self.founds)
        rights = Counter(self.rights)

        origin_all = 0
        found_all = 0
        right_all = 0
        for t, count in origins.items():

            # ignore NA
            if t == self.ignore_id:
                continue

            found = founds.get(t, 0)
            right = rights.get(t, 0)
            info[t] = self._score(count, found, right)
            origin_all += count
            found_all += found
            right_all += right

        return self._score(origin_all, found_all, right_all), info

    @staticmethod
    def _score(origin, found, right):
        """ Score Utils
        """
        p = 0 if found == 0 else (right / found)
        r = 0 if origin == 0 else (right / origin)
        f1 = 0. if r + p == 0 else (2 * p * r) / (p + r)
        return {'p': p, 'r': r, 'f1': f1}
