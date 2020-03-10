'''
One Idea:
First pre-compute the get_conf_score[id] for all O id
...and kge_model[id] for all H id => save both into all id tensor and query from tensor
This only needs to be done once before the y_opt optimization step.
TODO:
also move the current O optimization before the H optimization and y_opt optimization
    -> we want to first get reasonable kge embeddings before applying y_opt

memoized_score = torch.zeros(number_of_triples, device=main_args.device)
for O_id in tqdm(O_ids):
    memoized_score[O_id] = conf_score[O_id]
for H_id in tqdm(H_ids):
    memoized_score[H_id] = get_prob(kge_model(h,r,t,main_args), main_args.alpha_beta).detach()

'''

import numpy as np
import random
import torch
import torch.nn.functional as F
from tqdm import tqdm
from time import time
from collections import defaultdict
from time import time

def get_beta(x, alpha_beta):
    beta = alpha_beta * torch.sigmoid(x)
    return beta

def get_prob(x, alpha_beta):
    beta = get_beta(x, alpha_beta)
    alpha = alpha_beta - beta
    return (alpha - 1) / (alpha + beta - 2)

def e_step(data_dict, main_args, w, y_opt, kge_model):
    rule2groundings = data_dict['rule2groundings']
    id2conf = data_dict['id2conf']
    O_ids = list(data_dict['O_ids']) #TODO: optimize away this code
    H_ids = list(data_dict['H_ids']) #TODO: optimize away this code
    random.shuffle(O_ids)
    random.shuffle(H_ids)
    rule2weight_idx = data_dict['rule2weight_idx']
    id2conf = data_dict['id2conf']
    id2triplet = data_dict['id2triplet']

    alpha_beta = main_args.alpha_beta
    iters_y_opt = main_args.iters_y_opt
    iters_e = main_args.iters_e
    learning_rate = main_args.lr
    batch_size = main_args.batch_size

    num_observed = len(O_ids)
    offset = num_observed

    O_y_pred = torch.zeros(len(O_ids), requires_grad=True, device=main_args.device) # TODO: batch this!
    H_y_pred = torch.zeros(len(H_ids), requires_grad=True, device=main_args.device) # TODO: batch this!

    optimizer_E_step = torch.optim.Adam(
        filter(lambda p: p.requires_grad, kge_model.parameters()),
        lr=learning_rate)
    optimizer_y_opt = torch.optim.Adam([y_opt], lr=2000*learning_rate)

    #########################################################################
    # compute beta of observed triplets from KGE model
    #########################################################################
    cur_batch = 0
    optimizer_E_step.zero_grad()
    for _ in range(iters_e):
        print('computing KGE embeddings (observed)')
        # loss = torch.Tensor([0])
        loss = torch.tensor([0], dtype=torch.float32, device=main_args.device)
        O_ids_local = O_ids[cur_batch:cur_batch+batch_size]
        for tid in tqdm(O_ids_local):
            triplet = id2triplet[tid]
            # y_pred = torch.squeeze(kge_model(triplet.h,triplet.r,triplet.t, main_args)).type('torch.FloatTensor')
            # y_true = torch.FloatTensor([float(id2conf[0])], device=main_args.device)
            y_pred = torch.squeeze(kge_model(triplet.h, triplet.r, triplet.t, main_args))
            # y_pred = torch.tensor(y_pred, dtype=torch.float32, device=main_args.device)
            y_pred = y_pred.float().detach().to(device=main_args.device)
            y_true = torch.tensor([float(id2conf[tid])], dtype=torch.float32,device=main_args.device)
            loss += F.mse_loss(y_pred, y_true) / len(O_ids_local)
            O_y_pred[tid] = y_pred.detach()
        loss.requires_grad=True
        loss.backward()
        optimizer_E_step.step()
        print('loss applied')
        cur_batch += batch_size
        if cur_batch >= len(O_ids):
            cur_batch = 0
        print('E-loss (observed): {}'.format(loss))
        print('--------------')

    #########################################################################
    # Optimize y*: Optimizefrom(rule_dict, o_triplets, h_triplets, o2conf)
    #########################################################################
    if main_args.zijies_update:
        print('computing memoized scores for zj update')
        y_memoized = torch.zeros(len(id2triplet), device=main_args.device)
        for O_id in tqdm(O_ids):
            y_memoized[O_id] = id2conf[O_id]
        for H_id in tqdm(H_ids):
            h,r,t = id2triplet[H_id]
            y_memoized[H_id] = get_prob(kge_model(h, r, t, main_args), main_args.alpha_beta).detach()
    else:
        y_memoized = None

    optimizer_y_opt.zero_grad()
    print('Optimizing y_opt')
    for _ in range(iters_y_opt):
        pw = get_loglikelihood(get_prob, rule2groundings, rule2weight_idx,
                               id2conf, id2triplet, kge_model, w.float(),
                               y_opt, y_memoized, main_args)
        loss = -pw
        loss.backward()
        optimizer_y_opt.step()
        print('y-loss: {}'.format(pw))
        print('w: {}'.format(w))
        print('y_opt: {}'.format(get_prob(y_opt, alpha_beta)))
        print('--------------')


    #########################################################################
    # compute beta of hidden triplets from KGE model
    #########################################################################
    cur_batch = 0
    optimizer_E_step.zero_grad()
    for _ in range(iters_e):
        print('computing KGE embeddings (hidden)')
        # loss = torch.Tensor([0])
        loss = torch.tensor([0], dtype=torch.float32, device=main_args.device)
        H_ids_local = H_ids[cur_batch:cur_batch+batch_size]
        for tid in tqdm(H_ids_local):
            triplet = id2triplet[tid]
            y_pred = torch.squeeze(kge_model(triplet.h,triplet.r,triplet.t, main_args))
            # y_pred = torch.tensor(y_pred, dtype=torch.float32, device=main_args.device)
            y_pred = y_pred.float().detach().to(device=main_args.device)
            loss += F.mse_loss(y_pred, y_opt[tid]) / len(H_ids_local)
            H_y_pred[tid-offset] = y_pred.detach()
        loss.backward()
        optimizer_E_step.step()
        print('loss applied')
        cur_batch += batch_size
        if cur_batch >= len(H_ids):
            cur_batch = 0
        print('E-loss (hidden): {}'.format(loss))
        print('--------------')

    # save a data structure of id2beta dictionary
    print('linking y_opts')
    t = time()
    all_ids = O_ids + H_ids
    id2ystars = {tid: get_prob(y_opt[tid].detach(), alpha_beta) for tid in all_ids}
    print("linking y_opts: {} seconds".format(time() - t))
    print('linking betas')
    t = time()

    id2betas = {tid: get_beta(H_y_pred[tid-offset].detach(), alpha_beta) for tid in H_ids}
    print("linking betas: {} seconds".format(time() - t))
    return id2betas, id2ystars

def m_step1(data_dict, id2betas, id2ystars, w, main_args):
    rule2groundings = data_dict['rule2groundings']
    id2conf = data_dict['id2conf']
    rule2weight_idx = data_dict['rule2weight_idx']
    lr = main_args.lr
    alpha_beta = main_args.alpha_beta
    iters = main_args.iters_m
    batch_size = main_args.batch_size * 3
    accu_w_grad = np.zeros_like(w.detach().cpu().numpy())
    # calculate delta_w
    print("M-step: updating weights...")
    for i in range(iters):
        loss = 0.0
        for rule, allgroundings in rule2groundings.items():
            ground_size = len(allgroundings)
            w_idx = rule2weight_idx[rule]
            random.shuffle(allgroundings)
            for ground in tqdm(allgroundings[0:batch_size]):
                hidden = False
                head_id = ground[0]
                try:
                    conf_true = id2conf[head_id]
                    ystar = id2ystars[head_id]
                    accu_w_grad[w_idx] += (conf_true - ystar) / ground_size
                    # print(conf_true - ystar)
                    loss += (conf_true - ystar) * (conf_true - ystar)
                except KeyError:
                    hidden = True
                if hidden:
                    beta = id2betas[head_id]
                    ystar = id2ystars[head_id]
                    pred_beta = (alpha_beta - beta - 1) / (alpha_beta - 2)
                    accu_w_grad[w_idx] += (pred_beta - ystar) / ground_size
                    loss += (pred_beta - ystar) * (pred_beta - ystar)
            # update weights
            w[w_idx] += lr * accu_w_grad[w_idx]
        # print("M_step iteration {}: Loss = {}".format(i, loss))


def m_step2(data_dict, id2betas, id2ystars, w, main_args):
    rule2groundings = data_dict['rule2groundings']
    id2conf = data_dict['id2conf']
    rule2weight_idx = data_dict['rule2weight_idx']
    lr = main_args.lr
    alpha_beta = main_args.alpha_beta
    iters = main_args.iters_m
    batch_size = main_args.batch_size * 3

    print("M-step: updating weights...")
    id2weights = defaultdict(list)
    id2weights_n = {}
    optimizer_M_step = torch.optim.Adam([w], lr=main_args.lr)
    optimizer_M_step.zero_grad()

    for i in range(iters):
        for rule, allgroundings in rule2groundings.items():
            ground_size = len(allgroundings)
            w_idx = rule2weight_idx[rule]
            for ground in tqdm(allgroundings):
                hidden = False
                head_id = ground[0]
                id2weights[head_id].append(w[w_idx])
        print("Aggregating rule weights...")
        for k, v in tqdm(list(id2weights.items())):
            id2weights_n[k] = torch.sigmoid(torch.mean(torch.stack(v))).clone().detach().requires_grad_(True)

        id2weights_n_pool = list(id2weights_n.items())
        random.shuffle(id2weights_n_pool)

        cur_batch = 0
        print("Optimizing Loss...")
        while cur_batch < len(id2weights_n_pool):
            # error = torch.tensor([0], dtype=torch.float32,
            #                      device=main_args.device)
            # loss = torch.nn.MSELoss()
            # conf_true = [id2conf[hid] for hid, pred in id2weights_n_pool[cur_batch:cur_batch+batch_size] if id2conf.get(hid) is not None]
            # conf_true = torch.tensor(conf_true, device=main_args.device, requires_grad=True)
            # o_pred = [pred for hid, pred in id2weights_n_pool[cur_batch:cur_batch+batch_size] if id2conf.get(hid) is not None]
            # o_pred = torch.tensor(o_pred, device=main_args.device, requires_grad=True)
            # error += loss(o_pred, conf_true)
            #
            # y_star = [id2ystars[hid] for hid, pred in id2weights_n_pool[cur_batch:cur_batch + batch_size] if id2conf.get(hid) is None]
            # y_star = torch.tensor(y_star, device=main_args.device, requires_grad=True)
            # h_pred = [pred for hid, pred in id2weights_n_pool[cur_batch:cur_batch + batch_size] if id2conf.get(hid) is None]
            # h_pred = torch.tensor(h_pred, device=main_args.device, requires_grad=True)
            # error += loss(h_pred, y_star)
            #
            # error.backward(retain_graph=True)
            # optimizer_M_step.step()
            # print(w)
            # print('M-step loss: {}'.format(loss))
            # print('--------------')
            # cur_batch += batch_size
            print("Starting to ")
            loss = torch.tensor([0], dtype=torch.float32,
                                device=main_args.device)
            for hid, pred in id2weights_n_pool[cur_batch:cur_batch+batch_size]:
                conf_true = id2conf.get(hid)
                y_star = id2ystars[hid]
                if conf_true is not None:
                    loss += (conf_true - pred) * (conf_true - pred)
                else:
                    beta = id2betas[hid]
                    pred_beta = (alpha_beta - beta - 1) / (alpha_beta - 2)
                    loss += (y_star - pred) * (y_star - pred)

                loss.backward(retain_graph=True)
                optimizer_M_step.step()
                print(w)
                print('M-step loss: {}'.format(loss))
                print('--------------')

            cur_batch += batch_size


def m_step3(data_dict, id2betas, id2ystars, w, main_args):
    rule2groundings = data_dict['rule2groundings']
    id2conf = data_dict['id2conf']
    rule2weight_idx = data_dict['rule2weight_idx']
    lr = main_args.lr
    alpha_beta = main_args.alpha_beta
    iters = main_args.iters_m
    batch_size = main_args.batch_size * 3

    print("M-step: updating weights...")
    id2weights = defaultdict(list)
    id2weights_n = {}
    optimizer_M_step = torch.optim.Adam([w],
                                        lr=main_args.lr)
    optimizer_M_step.zero_grad()

    for i in range(iters):
        loss = torch.tensor([0], dtype=torch.float32, device=main_args.device)
        for rule, allgroundings in rule2groundings.items():
            ground_size = len(allgroundings)
            w_idx = rule2weight_idx[rule]
            random.shuffle(allgroundings)
            for ground in tqdm(allgroundings[0:batch_size]):
                hidden = False
                head_id = ground[0]
                id2weights[head_id].append(w[w_idx])

        for k, v in id2weights.items():
            id2weights_n[k] = torch.sigmoid(torch.tensor(np.mean([x.detach().item() for x in v]), device=main_args.device)).detach().requires_grad_(True)

        for hid, pred in id2weights_n.items():
            conf_true = id2conf.get(hid)
            y_star = id2ystars[hid]
            if conf_true is not None:
                loss += (conf_true - y_star) * (conf_true - y_star)
            else:
                beta = id2betas[hid]
                pred_beta = (alpha_beta - beta - 1) / (alpha_beta - 2)
                loss += (pred_beta - y_star) * (pred_beta - y_star)
        loss.requires_grad = True
        loss.backward()
        optimizer_M_step.step()
        print('M-step loss: {}'.format(loss))
        print('--------------')


def get_loglikelihood(get_prob, rule2groundings, rule2weight_idx, id2conf, id2triplet, kge_model, w, y_opt, y_memoized, main_args):
    pw_memoized = torch.zeros_like(w.float(), device=main_args.device)
    for rule in rule2groundings:
        groundings = rule2groundings[rule]
        widx = rule2weight_idx[rule]
        ids0, ids1, ids2 = [],[],[] #if we add new rules, change this
        for grounding in groundings:
            head, body = grounding.head, grounding.body
            if rule == 'ArelatedToB_and_BrelatedToC_imply_ArelatedToC':
                assert len(body) == 2
                t1, t2 = body
                id0, id1, id2 = head, t1, t2
                ids0.append(int(id0))
                ids1.append(int(id1))
                ids2.append(int(id2))
            elif rule == 'AcausesB_and_BcausesC_imply_AcausesC':
                assert len(body) == 2
                t1, t2 = body
                id0, id1, id2 = head, t1, t2
                ids0.append(int(id0))
                ids1.append(int(id1))
                ids2.append(int(id2))
            elif rule == 'notHidden':
                assert len(body) == 0
                id0 = head
                # id0 = triplet2id[head]
                ids0.append(int(id0))
            else:
                assert False
        if rule == 'ArelatedToB_and_BrelatedToC_imply_ArelatedToC':
            l0 = get_prob(y_opt[ids0], main_args.alpha_beta)
            if y_memoized is not None:
                l1 = y_memoized[ids1]
                l2 = y_memoized[ids2]
            else:
                l1 = get_prob(y_opt[ids1], main_args.alpha_beta)
                l2 = get_prob(y_opt[ids2], main_args.alpha_beta)
            grounding_confidence = soft_logic((soft_logic((l1,l2),'AND', len(l0), main_args.device),l0), 'IMPLY', len(l0), main_args.device)
        elif rule == 'AcausesB_and_BcausesC_imply_AcausesC':
            l0 = get_prob(y_opt[ids0], main_args.alpha_beta)
            if y_memoized is not None:
                l1 = y_memoized[ids1]
                l2 = y_memoized[ids2]
            else:
                l1 = get_prob(y_opt[ids1], main_args.alpha_beta)
                l2 = get_prob(y_opt[ids2], main_args.alpha_beta)
            grounding_confidence = soft_logic((soft_logic((l1,l2),'AND', len(l0), main_args.device),l0), 'IMPLY', len(l0), main_args.device)
        elif rule == 'notHidden':
            l0 = get_prob(y_opt[ids0], main_args.alpha_beta)
            grounding_confidence = soft_logic(l0, 'NOT', len(l0), main_args.device)
        else:
            assert False
        pw_memoized[widx] = w[widx] * torch.sum(grounding_confidence)

    pw = torch.sum(pw_memoized)
    return pw

# soft logic and EM functions
def soft_logic(args, logic, dim, device): # TODO: make gumbel softmax
    one, hinge, zero = torch.ones(dim, device=device), 0.75*torch.ones(dim, device=device), torch.zeros(dim, device=device)
    if logic == 'IMPLY':
        assert len(args) == 2
        arg1,arg2 = args
        forward_logic = torch.min(hinge, one - arg1 + arg2)
        memoized_cmp = torch.stack((hinge, one - arg1 + arg2),dim=0)
        backward_logic = torch.sum(F.softmin(memoized_cmp, dim=0)*memoized_cmp,dim=0)
    elif logic == 'AND':
        assert len(args) == 2
        arg1,arg2 = args
        forward_logic = torch.max(zero, arg1 + arg2 - one)
        memoized_cmp = torch.stack((zero, arg1 + arg2 - one),dim=0)
        backward_logic = torch.sum(F.softmax(memoized_cmp, dim=0)*memoized_cmp,dim=0)
    elif logic == 'OR':
        assert len(args) == 2
        arg1,arg2 = args
        forward_logic = torch.min(one, arg1 + arg2)
        memoized_cmp = torch.stack((one, arg1 + arg2),dim=0)
        backward_logic = torch.sum(F.softmin(memoized_cmp, dim=0)*memoized_cmp,dim=0)
    elif logic == 'NOT':
        assert type(args) != list and type(args) != tuple
        arg = args
        forward_logic = one - arg
        backward_logic = one - arg
    else:
        assert False
    return (forward_logic - backward_logic).detach() + backward_logic
