import numpy as np

def e_step(data_dict, alpha):
    rule2groundings = data_dict['rule2groundings']
    id2conf = data_dict['id2conf']
    O_triplets = data_dict['O_triplets']
    H_triplets = data_dict['H_triplets'],

    # Optimize y*: Optimizefrom(rule_dict, o_triplets, h_triplets, o2conf)

    # compute beta of hidden triplets from KGE model

    # Loss (y*, beta, O_tripleets, H_triplets

    # Backprop(Loss)

    # save a data structure of id2beta dictionary
    id2betas = {}
    id2ystars = {}

    return id2betas, id2ystars

def m_step(id2betas, id2ystars, lr, alpha):

    # calculate delta_w

    # update weights

    return [0, 0, 0]