# -*- coding: utf-8 -*-
from collections import Counter


class Score:

    def __init__(self, ignore_id=0):
        self.origins = []
        self.founds = []
        self.rights = []
        self.score = []

        self.ignore_id = ignore_id

    def update(self, input, target, target_pred_confidence=None):
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
