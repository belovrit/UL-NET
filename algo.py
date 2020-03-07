import numpy as np
import random
import torch
import torch.nn.functional as F
from tqdm import tqdm
from time import time
from collections import defaultdict

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
    triplet2id = data_dict['triplet2id']
    id2triplet = data_dict['id2triplet']

    alpha_beta = main_args.alpha_beta
    iters_y_opt = main_args.iters_y_opt
    iters_e = main_args.iters_e
    learning_rate = main_args.lr
    batch_size = main_args.batch_size

    num_observed = len(O_ids)
    offset = num_observed

    # Optimize y*: Optimizefrom(rule_dict, o_triplets, h_triplets, o2conf)
    optimizer_y_opt = torch.optim.Adam([y_opt], lr=5000*learning_rate)
    optimizer_y_opt.zero_grad()
    print('Optimizing y_opt')
    for _ in range(iters_y_opt):
        pw = get_loglikelihood(get_prob, rule2groundings, rule2weight_idx,
                               triplet2id, w.float(), y_opt, main_args)
        loss = -pw
        loss.backward()
        optimizer_y_opt.step()
        print('y-loss: {}'.format(pw))
        print('w: {}'.format(w))
        print('y_opt: {}'.format(get_prob(y_opt, alpha_beta)))
        print('--------------')

    optimizer_E_step = torch.optim.Adam(
        filter(lambda p: p.requires_grad, kge_model.parameters()),
        lr=learning_rate)

    optimizer_E_step.zero_grad()

    O_y_pred = torch.zeros(len(O_ids), requires_grad=True, device=main_args.device) # TODO: batch this!
    H_y_pred = torch.zeros(len(H_ids), requires_grad=True, device=main_args.device) # TODO: batch this!

    # compute beta of hidden triplets from KGE model
    cur_batch = 0
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
            y_true = torch.tensor([float(id2conf[0])], dtype=torch.float32,device=main_args.device)
            loss += F.mse_loss(y_pred, y_true) / len(O_ids_local)
            O_y_pred[tid] = y_pred.detach()
        try:
            loss.backward()
            optimizer_E_step.step()
            print('loss applied')
        except:
            print('loss skipped')
        cur_batch += batch_size
        if cur_batch >= len(O_ids):
            cur_batch = 0
        print('E-loss (observed): {}'.format(loss))
        print('--------------')

    cur_batch = 0
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
        try:
            loss.backward()
            optimizer_E_step.step()
            print('loss applied')
        except:
            print('loss skipped')
        cur_batch += batch_size
        if cur_batch >= len(H_ids):
            cur_batch = 0
        print('E-loss (hidden): {}'.format(loss))
        print('--------------')

    # save a data structure of id2beta dictionary
    id2betas = {}
    id2ystars = {}
    print('linking y_opts')
    t = time()
    # for tid in tqdm(O_ids+H_ids):
    #     id2ystars[tid] = get_prob(y_opt[tid].detach(), alpha_beta) #TODO: optimize away this code
    all_ids = O_ids + H_ids
    id2ystars = {tid: get_prob(y_opt[tid].detach(), alpha_beta) for tid in all_ids}
    print("linking y_opts: {} seconds".format(time() - t))
    print('linking betas')
    t = time()
    # for tid in tqdm(H_ids):
    #     id2betas[tid] = get_beta(H_y_pred[tid-offset].detach(), alpha_beta) #TODO: optimize away this code

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
                    # print(pred_beta - ystar)
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
                # try:
                #     conf_true = id2conf[head_id]
                #     ystar = id2ystars[head_id]
                #     accu_w_grad[w_idx] += (conf_true - ystar) / ground_size
                #     loss += (conf_true - ystar) * (conf_true - ystar)
                # except KeyError:
                #     hidden = True
                # if hidden:
                #     beta = id2betas[head_id]
                #     ystar = id2ystars[head_id]
                #     pred_beta = (alpha_beta - beta - 1) / (alpha_beta - 2)
                #     accu_w_grad[w_idx] += (pred_beta - ystar) / ground_size
                #     loss += (pred_beta - ystar) * (pred_beta - ystar)

            # update weights
            #w[w_idx] += lr * accu_w_grad[w_idx]
        # print("M_step iteration {}: Loss = {}".format(i, loss))
        for k, v in id2weights.items():
            id2weights_n[k] = torch.sigmoid(torch.tensor(np.mean([x.detach().item() for x in v]), device=main_args.device))

        for hid, pred in id2weights_n.items():
            conf_true = id2conf.get(hid)
            y_star = id2ystars[hid]
            if conf_true is not None:
                loss += (conf_true - y_star) * (conf_true - y_star)
            else:
                beta = id2betas[hid]
                pred_beta = (alpha_beta - beta - 1) / (alpha_beta - 2)
                loss += (pred_beta - y_star) * (pred_beta - y_star)
        loss.backward()
        optimizer_M_step.step()
        print('M-step loss: {}'.format(loss))
        print('--------------')


def get_loglikelihood(get_prob, rule2groundings, rule2weight_idx, triplet2id, w, y_opt, main_args):
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
                #id0, id1, id2 = triplet2id[head], triplet2id[t1],triplet2id[t2]
                ids0.append(int(id0))
                ids1.append(int(id1))
                ids2.append(int(id2))
            elif rule == 'AcausesB_and_BcausesC_imply_AcausesC':
                assert len(body) == 2
                t1, t2 = body
                id0, id1, id2 = head, t1, t2
                #id0, id1, id2 = triplet2id[head], triplet2id[t1],triplet2id[t2]
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
            '''
            l1 = torch.zeros_like(l0)
            l2 = torch.zeros_like(l0)
            for l, ids in zip([l1, l2], [ids0, ids1]):
                for j, id in enumerate(ids):
                    if id in id2conf:
                        l[j] = id2conf[id]
                    else:
                        l[j] = get_prob(kge_model(...), main_args.alpha_beta)
            '''
            l1 = get_prob(y_opt[ids1], main_args.alpha_beta)
            l2 = get_prob(y_opt[ids2], main_args.alpha_beta)
            grounding_confidence = soft_logic((soft_logic((l1,l2),'AND', len(l0), main_args.device),l0), 'IMPLY', len(l0), main_args.device)
        elif rule == 'AcausesB_and_BcausesC_imply_AcausesC':
            l0 = get_prob(y_opt[ids0], main_args.alpha_beta)
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
