import numpy as np
import random
import torch
import torch.nn.functional as F
from tqdm import tqdm

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
    def get_beta(x):
        beta = alpha_beta * F.sigmoid(x)
        return beta

    def get_prob(x):
        beta = get_beta(x)
        alpha = alpha_beta-beta
        return (alpha-1)/(alpha+beta-2)

    optimizer_y_opt = torch.optim.Adam([y_opt], lr=1000*learning_rate)
    optimizer_y_opt.zero_grad()
    print('Optimizing y_opt')
    for _ in range(iters_y_opt):
        pw = get_loglikelihood(get_prob, rule2groundings, rule2weight_idx,
                               triplet2id, w, y_opt, main_args)
        loss = -pw
        loss.backward()
        optimizer_y_opt.step()
        print('y-loss: {}'.format(pw))
        print('w: {}'.format(w))
        print('y_opt: {}'.format(get_prob(y_opt)))
        print('--------------')

    optimizer_E_step = torch.optim.Adam(kge_model.parameters(), lr=learning_rate)
    optimizer_E_step.zero_grad()

    O_y_pred = torch.zeros(len(O_ids), requires_grad=True, device=main_args.device) # TODO: batch this!
    H_y_pred = torch.zeros(len(H_ids), requires_grad=True, device=main_args.device) # TODO: batch this!

    # compute beta of hidden triplets from KGE model
    cur_batch = 0
    for _ in range(iters_e):
        print('computing KGE embeddings (observed)')
        loss = torch.Tensor([0])
        O_ids_local = O_ids[cur_batch:cur_batch+batch_size]
        for tid in tqdm(O_ids_local):
            triplet = id2triplet[tid]
            y_pred = torch.squeeze(kge_model(triplet.h,triplet.r,triplet.t, main_args)).type('torch.FloatTensor')
            y_true = torch.FloatTensor([float(id2conf[0])], device=main_args.device)
            loss += F.mse_loss(y_pred, y_true) / len(O_ids_local)
            O_y_pred[tid] = y_pred.detach()
        try:
            loss.backward()
            optimizer_E_step.step()
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
        loss = torch.Tensor([0])
        H_ids_local = H_ids[cur_batch:cur_batch+batch_size]
        for tid in tqdm(H_ids_local):
            triplet = id2triplet[tid]
            y_pred = torch.squeeze(kge_model(triplet.h,triplet.r,triplet.t, main_args))
            loss += F.mse_loss(y_pred, y_opt[tid]) / len(H_ids_local)
            H_y_pred[tid-offset] = y_pred.detach()
        try:
            loss.backward()
            optimizer_E_step.step()
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
    print('linking the beta and y_opts')
    for tid in tqdm(O_ids+H_ids):
        id2ystars[tid] = get_prob(y_opt[tid].detach()) #TODO: optimize away this code

    for tid in tqdm(H_ids):
        id2betas[tid] = get_beta(H_y_pred[tid-offset].detach()) #TODO: optimize away this code

    return id2betas, id2ystars

def m_step(data_dict, id2betas, id2ystars, w, lr, alpha_beta, iters):
    rule2groundings = data_dict['rule2groundings']
    id2conf = data_dict['id2conf']
    rule2weight_idx = data_dict['rule2weight_idx']
    accu_w_grad = np.zeros_like(w.detach().numpy())
    # calculate delta_w
    for i in range(iters):
        for rule, allgroundings in rule2groundings.items():
            ground_size = len(allgroundings)
            w_idx = rule2weight_idx[rule]
            for ground in tqdm(allgroundings):
                hidden = False
                head_id = ground[0]
                try:
                    conf_true = id2conf[head_id]
                    ystar = id2ystars[head_id]
                    accu_w_grad[w_idx] += (conf_true - ystar) / ground_size
                except KeyError:
                    hidden = True
                if hidden:
                    beta = id2betas[head_id]
                    ystar = id2ystars[head_id]
                    pred_beta = (alpha_beta - beta - 1) / (alpha_beta - 2)

                    accu_w_grad[w_idx] += (pred_beta - ystar) / ground_size
            # update weights
            w[w_idx] += lr * accu_w_grad[w_idx]



def get_loglikelihood(get_prob, rule2groundings, rule2weight_idx, triplet2id, w, y_opt, main_args):
    pw_memoized = torch.zeros_like(w, device=main_args.device)
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
            l0 = get_prob(y_opt[ids0])
            l1 = get_prob(y_opt[ids1])
            l2 = get_prob(y_opt[ids2])
            grounding_confidence = soft_logic((soft_logic((l1,l2),'AND'),l0), 'IMPLY')
        elif rule == 'AcausesB_and_BcausesC_imply_AcausesC':
            l0 = get_prob(y_opt[ids0])
            l1 = get_prob(y_opt[ids1])
            l2 = get_prob(y_opt[ids2])
            grounding_confidence = soft_logic((soft_logic((l1,l2),'AND'),l0), 'IMPLY')
        elif rule == 'notHidden':
            l0 = get_prob(y_opt[ids0])
            grounding_confidence = soft_logic(l0, 'NOT')
        else:
            assert False
        pw_memoized[widx] = w[widx] * torch.sum(grounding_confidence)

    pw = torch.sum(pw_memoized)
    return pw

# soft logic and EM functions
def soft_logic(args, logic): # TODO: make gumbel softmax
    one, hinge, zero = torch.Tensor([1]), torch.Tensor([0.75]), torch.Tensor([0])
    if logic == 'IMPLY':
        assert len(args) == 2
        arg1,arg2 = args
        return torch.min(hinge, one - arg1 + arg2)
    elif logic == 'AND':
        assert len(args) == 2
        arg1,arg2 = args
        return torch.max(zero, arg1 + arg2 - one)
    elif logic == 'OR':
        assert len(args) == 2
        arg1,arg2 = args
        return torch.min(one, arg1 + arg2)
    elif logic == 'NOT':
        assert type(args) != list and type(args) != tuple
        arg = args
        return one-arg
