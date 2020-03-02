from collections import namedtuple
from tqdm import tqdm
import csv, os

mkstrlist = lambda x_list: [str(x) for x in x_list]
Triplet = namedtuple('Triplet', ['h','r','t'])
Grounding = namedtuple('Grounding', ['head', 'body'])

tid = 0
O2conf = {}
triplet2id = {}
id2triplet = {}
hr2t = {}
O_triplets = set()
H_triplets = set()
entities = set()
relations = set()

rule2groundings = {}
rule2weight_idx = {}
rules = set()

def generate_ArelatedToB_and_BrelatedToC_imply_ArelatedToC():
    global tid, rule2groundings, H_triplets, id2triplet, triplet2id
    rule_name = 'ArelatedToB_and_BrelatedToC_imply_ArelatedToC'
    num_rules = 0
    r_b0 = '0'
    r_b1 = '0'
    r_h = '0'
    rule2groundings[rule_name] = []
    print('Loading {} Rules'.format(rule_name))
    for tid_O in tqdm(O_triplets):
        triplet_b0 = id2triplet[tid_O]
        if triplet_b0.r == r_b0 and (triplet_b0.t,r_b1) in hr2t:
            for t in hr2t[(triplet_b0.t,r_b1)]:
                triplet_b1 = Triplet(triplet_b0.t, r_b1, t)
                assert triplet2id[triplet_b1] in O_triplets
                triplet_h = Triplet(triplet_b0.h, r_h, t)
                if triplet_h not in triplet2id: # i.e. not in O_triplets.union(H_triplets)
                    H_triplets.add(tid)
                    id2triplet[tid] = triplet_h
                    triplet2id[triplet_h] = tid
                    assert len(id2triplet) == len(triplet2id)
                    tid += 1
                grounding = Grounding(triplet_h, [triplet_b0, triplet_b1])
                rule2groundings[rule_name].append(grounding)
                num_rules += 1
    print('Loaded {} groundings'.format(num_rules))
    return rule_name

def generate_AcausesB_and_BcausesC_imply_AcausesC():
    global tid, rule2groundings, H_triplets, id2triplet, triplet2id
    rule_name = 'AcausesB_and_BcausesC_imply_AcausesC'
    num_rules = 0
    r_b0 = '22'
    r_b1 = '22'
    r_h = '22'
    rule2groundings[rule_name] = []
    print('Loading {} Rules'.format(rule_name))
    for tid_O in tqdm(O_triplets):
        triplet_b0 = id2triplet[tid_O]
        if triplet_b0.r == r_b0 and (triplet_b0.t,r_b1) in hr2t:
            for t in hr2t[(triplet_b0.t,r_b1)]:
                triplet_b1 = Triplet(triplet_b0.t, r_b1, t)
                assert triplet2id[triplet_b1] in O_triplets
                triplet_h = Triplet(triplet_b0.h, r_h, t)
                if triplet_h not in triplet2id: # i.e. not in O_triplets.union(H_triplets)
                    H_triplets.add(tid)
                    id2triplet[tid] = triplet_h
                    triplet2id[triplet_h] = tid
                    assert len(id2triplet) == len(triplet2id)
                    tid += 1
                grounding = Grounding(triplet_h, [triplet_b0, triplet_b1])
                rule2groundings[rule_name].append(grounding)
                num_rules += 1
    print('Loaded {} groundings'.format(num_rules))
    return rule_name

def generate_notHidden():
    global rule2groundings, H_triplets
    # assert that this is the last generate function!!!
    rule_name = 'notHidden'
    num_rules = 0
    rule2groundings[rule_name] = []
    print('Loading {} Rules'.format(rule_name))
    for tid_H in tqdm(H_triplets):
        triplet_h = id2triplet[tid_H]
        grounding = Grounding(triplet_h, [])
        rule2groundings[rule_name].append(grounding)
        num_rules += 1
    print('Loaded {} groundings'.format(num_rules))
    return rule_name

def init_globals(fi_name_list_shirley):
    assert len(fi_name_list_shirley) # do we want to include valid and test data??
    global tid,triplet2id,id2triplet,hr2t,O_triplets,H_triplets,\
            entities,relations,rule2groundings,rule2weight_idx,rules

    for fi_name in fi_name_list_shirley:
        fi = open(fi_name)
        rd = csv.reader(fi, delimiter = '\t')

        for ln in rd:
            h,r,t,c = ln
            triplet = Triplet(h,r,t)
            if triplet not in triplet2id:
                id2triplet[tid] = triplet
                triplet2id[triplet] = tid
                O2conf[tid] = c
                assert len(id2triplet) == len(triplet2id)
                tid += 1
            if (h,r) not in hr2t:
                hr2t[(h,r)] = []
            hr2t[(h,r)].append(t)

            O_triplets.add(tid)
            entities.add(h)
            relations.add(r)
            entities.add(t)
        fi.close()

    rule_generators = [\
                generate_ArelatedToB_and_BrelatedToC_imply_ArelatedToC,\
                generate_AcausesB_and_BcausesC_imply_AcausesC,\
                generate_notHidden\
            ]

    weight_idx = 0
    assert generate_notHidden == rule_generators[-1]
    for generate_rule in rule_generators:
        rule_name = generate_rule()
        rules.add(rule_name)
        rule2weight_idx[rule_name] = weight_idx
        weight_idx += 1

    # TODO: initialize weight tensor of dimension len(rules)

def prepare_data(data_path):
    data_file_shirley_list = [os.path.join(data_path, 'train.tsv')] #['data/cn15k/train.tsv']
    init_globals(data_file_shirley_list)

    # this line takes a long time
    global tid,triplet2id,id2triplet,hr2t,O_triplets,H_triplets,\
            entities,relations,rule2groundings,rule2weight_idx,rules
    assert len(rules) == len(rule2weight_idx) == len(rule2groundings)
    assert len(id2triplet) == len(triplet2id)
    print('===========================')
    print('number of triplet2id:')
    print(len(triplet2id))
    print('---------------------------')
    print('number of O_triplets:')
    print(len(O_triplets))
    print('---------------------------')
    print('number of H_triplets:')
    print(len(H_triplets))
    print('---------------------------')
    print('number of entities:')
    print(len(entities))
    print('---------------------------')
    print('number of relations:')
    print(len(relations))
    print('---------------------------')
    print('number of rules:')
    print(len(rules))
    print('---------------------------')
    for rule_name in rule2groundings:
        print('number of groundings for rule {}:'.format(rule_name))
        print(len(rule2groundings[rule_name]))

    return {'triplet2id': triplet2id, 'id2triplet': id2triplet,
            'O_triplets': O_triplets, 'H_triplets': H_triplets,
            'entities': entities, 'relations': relations,
            'rules': rules, 'rule2groundings': rule2groundings,
            'rule2weight_idx': rule2weight_idx, 'id2conf': O2conf}

    # cleanup takes a long time too