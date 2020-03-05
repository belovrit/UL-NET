import torch
import numpy as np
import pandas as pd
import csv
from os.path import join
from src.utils import *

class Tester(object):
    class IndexScore:
        """
        The score of a tail when h and r is given.
        It's used in the ranking task to facilitate comparison and sorting.
        Print w as 3 digit precision float.
        """

        def __init__(self, index, score):
            self.index = index
            self.score = score

        def __lt__(self, other):
            return self.score < other.score

        def __repr__(self):
            # return "(index: %d, w:%.3f)" % (self.index, self.score)
            return "(%d, %.3f)" % (self.index, self.score)

        def __str__(self):
            return "(index: %d, w:%.3f)" % (self.index, self.score)

    def __init__(self, KGEModel):
        self.KGEModel = KGEModel
        # below for test data
        self.test_triplets = []
        self.hr_map = {}

    def load_test_triplets_conf_task(self, load_path):
        file_path = join(load_path, 'test.tsv')
        fi = open(file_path)
        rd = csv.reader(fi, delimiter='\t')
        count = 0
        #self.test_triplets = []
        for ln in rd:
            h, r, t, c = ln
            self.test_triplets.append([h, r, t, float(c)])
            if self.hr_map.get(h) is None:
                self.hr_map[h] = {}
            if self.hr_map[h].get(r) is None:
                self.hr_map[h][r] = {t: float(c)}
            else:
                self.hr_map[h][r][t] = float(c)
            count += 1
        #self.test_triplets = np.array(self.test_triplets)
        print("Loaded confidence prediction (h,r,t,?c) queries: {} queries".format(count))

        # i = 0
        # for h in self.hr_map:
        #     print(self.hr_map[h])
        #     print(len(self.hr_map[h]))
        #     count += len(self.hr_map[h])
        #     i += 1
        #     if i == 1:
        #         break


    def load_test_triplets_ranking_task(self, load_path):
        file_path = join(load_path, 'test.tsv')
        fi = open(file_path)
        rd = csv.reader(fi, delimiter='\t')
        self.hr_map = {}
        count = 0
        for ln in rd:
            h, r, t, c = ln
            if self.hr_map.get(h) is None:
                self.hr_map[h] = {}
            if self.hr_map[h].get(r) is None:
                self.hr_map[h][r] = {t: float(c)}
            else:
                self.hr_map[h][r][t] = float(c)

        supplement_t_files = ['train.tsv', 'val.tsv']
        for file in supplement_t_files:
            file_path = join(load_path, file)
            fi = open(file_path)
            rd = csv.reader(fi, delimiter='\t')
            for ln in rd:
                h, r, t, c = ln

                # update hr_map
                if h in self.hr_map and r in self.hr_map[h]:
                    self.hr_map[h][r][t] = float(c)
        # i = 0
        # for h in self.hr_map:
        #     print(self.hr_map[h])
        #     count += len(self.hr_map[h])
        #     i += 1
        #     if i == 1:
        #         break
        print('Loaded ranking test queries. Number of (h,r,?t) queries: %d' % count)

    def get_score_batch(self, test_triplets):
        kge = self.KGEModel
        scores = [kge.predict(int(h), int(r), int(t)) for h, r, t, c in test_triplets]
        return scores

    def get_mse(self, verbose=True, save_dir='', epoch=0):
        test_triplets = self.test_triplets
        N = len(test_triplets)
        c_batch = np.array([triplet[3] for triplet in test_triplets])
        #c_batch = np.random.rand(len(test_triplets))
        scores = self.get_score_batch(test_triplets)
        mse = np.sum(np.square(scores - c_batch))
        mse = mse / N

        return mse



if __name__ == '__main__':
    tester = Tester('TransE')
    data_path = join(get_data_path(), 'cn15k')
    tester.load_test_triplets_conf_task(data_path)
    # tester.load_test_triplets_ranking_task(data_path)
    print(tester.get_mse())
    # print(type(tester.test_triplets[0][3]))
