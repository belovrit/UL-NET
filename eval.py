import csv
from os.path import join
from src.utils import *
import time
from KGEModel import *
from algo import get_beta, get_prob
from tqdm import tqdm
from collections import defaultdict

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

    def __init__(self, KGEModel, mln_lambda):
        self.KGEModel = KGEModel
        # below for test data
        self.test_triplets = []
        self.triplet2id = {}
        self.id2mln_prob = {}
        self.mln_lambda = mln_lambda
        self.hr_map = {}
        self.num_not_hidden = 0
        self.num_no_mln = 0

    def load_test_triplets_conf_task(self, load_path):
        file_path = join(load_path, 'test.tsv')
        fi = open(file_path)
        rd = csv.reader(fi, delimiter='\t')
        count = 0
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

        print("Loaded confidence prediction (h,r,t,?c) queries: {} queries".format(count))

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

        count = 0
        for h in self.hr_map:
            count += len(self.hr_map[h])
        print('Loaded ranking test queries. Number of (h,r,?t) queries: %d' % count)

    def compute_mln_pred(self, data_dict, id2ystars, w):
        rule2groundings = data_dict['rule2groundings']
        rule2weight_idx = data_dict['rule2weight_idx']
        self.triplet2id = data_dict['triplet2id']
        id2triplet = data_dict['id2triplet']
        id2weights = defaultdict(list)
        for rule, allgroundings in rule2groundings.items():
            w_idx = rule2weight_idx[rule]
            for ground in tqdm(allgroundings):
                if rule == 'ArelatedToB_and_BrelatedToC_imply_ArelatedToC' or \
                        rule == 'AcausesB_and_BcausesC_imply_AcausesC':
                    head_id = ground.head
                    b_id1, b_id2 = ground.body
                    id2weights[head_id].append(w[w_idx] * id2ystars[b_id1].item() * id2ystars[b_id2].item())
                else:
                    head_id = ground.head
                    id2weights[head_id].append(w[w_idx] * (1 - id2ystars[head_id].item()))

        for k, v in id2weights.items():
            triplet = id2triplet[k]
            rel = triplet.r
            if rel == '0':
                w_idx = rule2weight_idx['ArelatedToB_and_BrelatedToC_imply_ArelatedToC']
            elif rel == '22':
                w_idx = rule2weight_idx['AcausesB_and_BcausesC_imply_AcausesC']
            else:
                w_idx = rule2weight_idx['notHidden']
            self.id2mln_prob[k] = np.sum(np.array(v)) / (w[w_idx] * len(v))

    def get_single_mln_pred(self, h, r, t):
        triplet = Triplet(str(h), str(r), str(t))
        tid = self.triplet2id.get(triplet)
        if tid is None:
            self.num_not_hidden += 1
            return 0.0
        else:
            pred = self.id2mln_prob.get(tid)
            if pred is None:
                self.num_no_mln += 1
                return 0.0
            else:
                return pred

    def get_score(self, h, r, t, alpha_beta):
        if self.mln_lambda == 0:
            score = get_prob(self.KGEModel.predict(int(h), int(r), int(t)), alpha_beta).item()
        else:
            kge_score = get_prob(self.KGEModel.predict(int(h), int(r), int(t)), alpha_beta).item()
            mln_raw = self.get_single_mln_pred(int(h), int(r), int(t))
            if mln_raw >= 0.001:
                mln_score = shifted_sigmoid(mln_raw)
            else:
                mln_score = mln_raw
            if mln_score > 0.5:
                score = (1-self.mln_lambda) * kge_score + self.mln_lambda * mln_score
            else:
                if mln_score == 0.0 and self.mln_lambda < 1.0:
                    score = kge_score
                else:
                    if self.mln_lambda < 1.0:
                        score = kge_score - self.mln_lambda * mln_score
                    else:
                        score = mln_score

        return score

    def get_score_batch(self, test_triplets, alpha_beta):
        scores = [self.get_score(int(h), int(r), int(t), alpha_beta) for h, r, t, c in test_triplets]
        return scores

    def get_mse(self, alpha_beta, verbose=True, save_dir='', epoch=0):
        test_triplets = self.test_triplets
        N = len(test_triplets)
        c_batch = np.array([triplet[3] for triplet in test_triplets])
        scores = self.get_score_batch(test_triplets, alpha_beta)
        mse = np.sum(np.square(scores - c_batch))
        mse = mse / N

        return mse.item(), scores, self.num_not_hidden, self.num_no_mln

    def get_mae(self, alpha_beta, verbose=False, save_dir='', epoch=0):
        test_triplets = self.test_triplets
        N = len(test_triplets)
        c_batch = np.array([triplet[3] for triplet in test_triplets])
        scores = self.get_score_batch(test_triplets, alpha_beta)
        mae = np.sum(np.absolute(scores - c_batch))

        mae = mae / N
        return mae.item()

    def get_t_ranks(self, h, r, ts, alpha_beta):
        """
        Given some t index, return the ranks for each t
        :return:
        """
        # prediction
        scores = np.array([self.get_score(h, r, t, alpha_beta) for t in ts])  # predict scores for t from ground truth

        ranks = np.ones(len(ts), dtype=int)  # initialize rank as all 1

        N = self.KGEModel.nentity  # pool of t: all possible ts
        #all_scores = [self.get_score(h, r, i, alpha_beta) for i in range(N)]
        for i in range(N):  # compute scores for all possible t
            score_i = self.get_score(h, r, i, alpha_beta)
            rankplus = (scores < score_i).astype(int)  # rank+1 if score<score_i
            ranks += rankplus

        return ranks #, all_scores

    def ndcg(self, h, r, tw_truth, alpha_beta):
        """
        Compute nDCG(normalized discounted cummulative gain)
        sum(score_ground_truth / log2(rank+1)) / max_possible_dcg
        :param tw_truth: [IndexScore1, IndexScore2, ...], soreted by IndexScore.score descending
        :return:
        """
        # prediction
        ts = [tw.index for tw in tw_truth]
        ranks = self.get_t_ranks(h, r, ts, alpha_beta)

        # linear gain
        gains = np.array([tw.score for tw in tw_truth])
        discounts = np.log2(ranks + 1)
        discounted_gains = gains / discounts
        dcg = np.sum(discounted_gains)  # discounted cumulative gain
        # normalize
        max_possible_dcg = np.sum(gains / np.log2(np.arange(len(gains)) + 2))  # when ranks = [1, 2, ...len(truth)]
        ndcg = dcg / max_possible_dcg  # normalized discounted cumulative gain

        # exponential gain
        exp_gains = np.array([2 ** tw.score - 1 for tw in tw_truth])
        exp_discounted_gains = exp_gains / discounts
        exp_dcg = np.sum(exp_discounted_gains)
        # normalize
        exp_max_possible_dcg = np.sum(
            exp_gains / np.log2(np.arange(len(exp_gains)) + 2))  # when ranks = [1, 2, ...len(truth)]
        exp_ndcg = exp_dcg / exp_max_possible_dcg  # normalized discounted cumulative gain

        return ndcg, exp_ndcg

    def mean_ndcg(self, hr_map, alpha_beta):
        """
        :param hr_map: {h:{r:{t:w}}}
        :return:
        """
        ndcg_sum = 0  # nDCG with linear gain
        exp_ndcg_sum = 0  # nDCG with exponential gain
        count = 0

        t0 = time.time()

        # debug ndcg
        res = []  # [(h,r,tw_truth, ndcg)]

        for h in hr_map:
            for r in hr_map[h]:
                tw_dict = hr_map[h][r]  # {t:w}
                tw_truth = [self.IndexScore(int(t), float(w)) for t, w in tw_dict.items()]
                tw_truth.sort(reverse=True)  # descending on w
                ndcg, exp_ndcg = self.ndcg(h, r, tw_truth, alpha_beta)  # nDCG with linear gain and exponential gain
                ndcg_sum += ndcg
                exp_ndcg_sum += exp_ndcg
                count += 1
                if count % 100 == 0:
                    print('Processed %d, time %s' % (count, (time.time() - t0)))
                    print('mean ndcg (linear gain) now: %f' % (ndcg_sum / count))
                    print('mean ndcg (exponential gain) now: %f' % (exp_ndcg_sum / count))

                # debug
                ranks = self.get_t_ranks(h, r, [tw.index for tw in tw_truth], alpha_beta)
                res.append((h,r,tw_truth, ndcg, ranks))

        return ndcg_sum / count, exp_ndcg_sum / count


if __name__ == '__main__':
    kge = KGEModel("TransE", 15000, 32, 16, 0.05)
    tester = Tester(kge)
    data_path = join(get_data_path(), 'cn15k')
    # tester.load_test_triplets_conf_task(data_path)
    tester.load_test_triplets_ranking_task(data_path)
    alpha_beta = 1
    print(tester.mean_ndcg(tester.hr_map, alpha_beta))
    # print(type(tester.test_triplets[0][3]))
