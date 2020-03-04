import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

def e_step(data_dict, main_args, w, y_opt, kge_model):
    rule2groundings = data_dict['rule2groundings']
    id2conf = data_dict['id2conf']
    O_ids = sorted(list(data_dict['O_ids'])) #TODO: optimize away this code
    H_ids = sorted(list(data_dict['H_ids'])) #TODO: optimize away this code
    rule2weight_idx = data_dict['rule2weight_idx']
    triplet2id = data_dict['triplet2id']
    id2triplet = data_dict['id2triplet']

    alpha_beta = main_args.alpha_beta
    iters_y_opt = main_args.iters_y_opt
    iters_e = main_args.iters_e
    learning_rate = main_args.lr

    # Finding y_true
    print('Finding y_true')
    num_observed = len(O_ids)
    offset = num_observed
    y_true = torch.FloatTensor([0]*num_observed, device=main_args.device)
    for tid in tqdm(O_ids):
        y_true[tid] += id2conf[0] #TODO: optimize away this code

    # Optimize y*: Optimizefrom(rule_dict, o_triplets, h_triplets, o2conf)
    def get_prob(x):
        beta = alpha_beta * F.sigmoid(x)
        alpha = alpha_beta-beta
        return (alpha-1)/(alpha+beta-2)

    optimizer_y_opt = torch.optim.Adam([y_opt], lr=1000*learning_rate)
    optimizer_y_opt.zero_grad()
    print('Optimizing y_opt')
    for _ in range(iters_y_opt):
        pw = get_loglikelihood(get_prob, rule2groundings, rule2weight_idx,
                               triplet2id, w, y_opt, y_true, offset, main_args)
        loss = -pw
        loss.backward()
        optimizer_y_opt.step()
        print('y-loss: {}'.format(pw))
        print('w: {}'.format(w))
        print('y_opt: {}'.format(get_prob(y_opt)))
        print('y_true: {}'.format(get_prob(y_true)))
        print('--------------')

    optimizer_E_step = torch.optim.Adam(kge_model.parameters(), lr=learning_rate)
    optimizer_E_step.zero_grad()

    #H_ids = H_ids[:5]
    #O_ids = O_ids[:5]
    for _ in range(main_args.iters_e):
        # compute beta of hidden triplets from KGE model
        loss = torch.Tensor([0])
        print('computing KGE embeddings (hidden)')
        H_y_pred = torch.zeros(len(H_ids), requires_grad=True, device=main_args.device) # TODO: batch this!
        for tid in tqdm(H_ids):
            triplet = id2triplet[tid]
            y_pred = torch.squeeze(kge_model(triplet.h,triplet.r,triplet.t, main_args))
            loss += F.mse_loss(y_pred, y_opt[tid-offset])
            H_y_pred[tid-offset] += y_pred.detach()
        print('computing KGE embeddings (observed)')
        O_y_pred = torch.zeros(len(O_ids), requires_grad=True, device=main_args.device) # TODO: batch this!
        for tid in tqdm(O_ids):
            triplet = id2triplet[tid]
            y_pred = torch.squeeze(kge_model(triplet.h,triplet.r,triplet.t, main_args))
            loss += F.mse_loss(y_pred.type('torch.FloatTensor'), y_true[tid].type('torch.FloatTensor'))
            O_y_pred[tid] = y_pred.detach()

        loss.backward()
        optimizer_E_step.step()
        print('E-loss: {}'.format(loss))
        print('y_opt: {}'.format(y_opt))
        print('--------------')

    # save a data structure of id2beta dictionary
    id2betas = {}
    id2ystars = {}
    print('linking the beta and y_opts')
    for tid in tqdm(O_ids):
        id2betas[tid] = O_y_pred #TODO: optimize away this code
    for tid in tqdm(H_ids):
        id2ystars[tid] = y_opt[tid] #TODO: optimize away this code
        id2betas[tid] = H_y_pred #TODO: optimize away this code

    return id2betas, id2ystars

def m_step(id2betas, id2ystars, lr, alpha, iters):

    # calculate delta_w

    # update weights

    return [0, 0, 0]



def get_loglikelihood(get_prob, rule2groundings, rule2weight_idx, triplet2id, w, y_opt, y_true, offset, main_args):
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
                id0, id1, id2 = triplet2id[head], triplet2id[t1],triplet2id[t2]
                ids0.append(int(id0)-offset)
                ids1.append(int(id1))
                ids2.append(int(id2))
            elif rule == 'AcausesB_and_BcausesC_imply_AcausesC':
                assert len(body) == 2
                t1, t2 = body
                id0, id1, id2 = triplet2id[head], triplet2id[t1],triplet2id[t2]
                ids0.append(int(id0)-offset)
                ids1.append(int(id1))
                ids2.append(int(id2))
            elif rule == 'notHidden':
                assert len(body) == 0
                id0 = triplet2id[head]
                ids0.append(int(id0)-offset)
            else:
                assert False
        if rule == 'ArelatedToB_and_BrelatedToC_imply_ArelatedToC':
            l0 = get_prob(y_opt[ids0])
            l1 = get_prob(y_true[ids1])
            l2 = get_prob(y_true[ids2])
            grounding_confidence = soft_logic((soft_logic((l1,l2),'AND'),l0), 'IMPLY')
        elif rule == 'AcausesB_and_BcausesC_imply_AcausesC':
            l0 = get_prob(y_opt[ids0])
            l1 = get_prob(y_true[ids1])
            l2 = get_prob(y_true[ids2])
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
