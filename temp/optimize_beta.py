import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
from data_load import load_data
from tqdm import tqdm

# class definitions
Triplet = namedtuple('Triplet', ['h','r','t'])
Grounding = namedtuple('Grounding', ['head', 'body'])
class KGEModel(nn.Module):
    def __init__(self, model_name, nentity, nrelation, hidden_dim, gamma, 
                 double_entity_embedding=False, double_relation_embedding=False):
        super(KGEModel, self).__init__()
        self.model_name = model_name
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0
        
        self.gamma = nn.Parameter(
            torch.Tensor([gamma]), 
            requires_grad=False
        )
        
        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]), 
            requires_grad=False
        )
        
        self.entity_dim = hidden_dim*2 if double_entity_embedding else hidden_dim
        self.relation_dim = hidden_dim*2 if double_relation_embedding else hidden_dim
        
        self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim))
        nn.init.uniform_(
            tensor=self.entity_embedding, 
            a=-self.embedding_range.item(), 
            b=self.embedding_range.item()
        )
        
        self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
        nn.init.uniform_(
            tensor=self.relation_embedding, 
            a=-self.embedding_range.item(), 
            b=self.embedding_range.item()
        )
        
        #Do not forget to modify this line when you add a new model in the "forward" function
        if model_name not in ['TransE', 'DistMult', 'ComplEx', 'RotatE']:
            raise ValueError('model %s not supported' % model_name)
            
        if model_name == 'RotatE' and (not double_entity_embedding or double_relation_embedding):
            raise ValueError('RotatE should use --double_entity_embedding')

        if model_name == 'ComplEx' and (not double_entity_embedding or not double_relation_embedding):
            raise ValueError('ComplEx should use --double_entity_embedding and --double_relation_embedding')
    
    def forward(self,h,r,t,mode='simple'):
        global dev
        cvtTensor = lambda x: torch.LongTensor([int(x)], device=dev)
        h,r,t = cvtTensor(h),cvtTensor(r),cvtTensor(t)
        model_func = {\
            'TransE': self.TransE,\
            'DistMult': self.DistMult,\
            'ComplEx': self.ComplEx,\
            'RotatE': self.RotatE,\
        }
        assert self.model_name in model_func
        hd = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=h).unsqueeze(1)
        rn = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=r).unsqueeze(1)
        tl = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=t).unsqueeze(1)
        score = model_func[self.model_name](hd, rn, tl, mode)
        return score

    def TransE(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head + (relation - tail)
        else:
            score = (head + relation) - tail

        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score

    def DistMult(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head * (relation * tail)
        else:
            score = (head * relation) * tail

        score = score.sum(dim = 2)
        return score

    def ComplEx(self, head, relation, tail, mode):
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_relation, im_relation = torch.chunk(relation, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            score = re_head * re_score + im_head * im_score
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            score = re_score * re_tail + im_score * im_tail

        score = score.sum(dim = 2)
        return score

    def RotatE(self, head, relation, tail, mode):
        pi = 3.14159265358979323846
        
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        #Make phases of relations uniformly distributed in [-pi, pi]

        phase_relation = relation/(self.embedding_range.item()/pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail

# soft logic and EM functions
def soft_logic(args, logic):
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

def get_loglikelihood(fo, num_observed):
    global rule2groundings, rule2weight_idx, w, h

    get_h = lambda idxs: F.sigmoid(h[[(idx-num_observed) for idx in idxs]])
    pw_memoized = torch.zeros_like(w, device=dev)
    for rule in rule2groundings:
        groundings = rule2groundings[rule]
        widx = rule2weight_idx[rule]
        ids0, ids1, ids2 = [],[],[] #if we add new rules, change this
        for grounding in groundings:
            head = grounding.head
            body = grounding.body
            if rule == 'ArelatedToB_and_BrelatedToC_imply_ArelatedToC':
                assert len(body) == 2
                id0 = triplet2id[head]
                t1, t2 = body
                id1, id2 = triplet2id[t1],triplet2id[t2]
                ids0.append(id0)
                ids1.append(id1)
                ids2.append(id2)
            elif rule == 'AcausesB_and_BcausesC_imply_AcausesC':
                assert len(body) == 2
                id0 = triplet2id[head]
                t1, t2 = body
                id1, id2 = triplet2id[t1],triplet2id[t2]
                ids0.append(id0)
                ids1.append(id1)
                ids2.append(id2)
            elif rule == 'notHidden':
                assert len(body) == 0
                id0 = triplet2id[head]
                ids0.append(id0)
            else:
                assert False
        if rule == 'ArelatedToB_and_BrelatedToC_imply_ArelatedToC':
            l0 = get_h(ids0)
            l1 = fo[ids1].detach()
            l2 = fo[ids2].detach()
            grounding_confidence = soft_logic((soft_logic((l1,l2),'AND'),l0), 'IMPLY')
        elif rule == 'AcausesB_and_BcausesC_imply_AcausesC':
            l0 = get_h(ids0)
            l1 = fo[ids1].detach()
            l2 = fo[ids2].detach()
            grounding_confidence = soft_logic((soft_logic((l1,l2),'AND'),l0), 'IMPLY')
        elif rule == 'notHidden':
            l0 = get_h(ids0)
            grounding_confidence = soft_logic(l0, 'NOT')
        else:
            assert False
        pw_memoized[widx] = w[widx] * torch.sum(grounding_confidence)

    pw = torch.sum(pw_memoized)
    return pw 

def get_y_opt(fo):
    global h, iters_y_opt, learning_rate

    optimizer_y_opt = torch.optim.Adam([h], lr=learning_rate)
    optimizer_y_opt.zero_grad()
    print('Optimizing y_opt')
    for i in range(iters_y_opt):
        pw = get_loglikelihood(fo, len(O_ids))
        #if i % 3 == 0:
        print('y-loss: {}'.format(pw))
        print('w: {}'.format(w))
        print('h: {}'.format(F.sigmoid(h)))
        print('f: {}'.format(fo))
        print('--------------')
        loss = -pw
        loss.backward()
        optimizer_y_opt.step()
    
    return F.sigmoid(h)

def run_E_step():
    global iters_E_step, O_ids
    fo = torch.zeros(len(O_ids), requires_grad=True, device=dev)
    fh = torch.zeros(len(H_ids), requires_grad=True, device=dev)
    optimizer_E_step = torch.optim.Adam([h], lr=learning_rate)
    optimizer_E_step.zero_grad()
    print('Optimizing E-step')
    for i in range(iters_E_step):
        print('computing KGE embeddings (observed)')
        for tid in tqdm(O_ids):
            triplet = id2triplet[tid]
            fo[tid] = kge_model(triplet.h,triplet.r,triplet.t)
        print('finding y_optimal/y_star')
        y_opt = get_y_opt(fo).detach()
        loss = torch.tensor([0], device=dev)

        O_ids_list = torch.LongTensor(list(O_ids), device=dev)[0:3]
        O_true = torch.FloatTensor([id2conf[int(tid)] for tid in O_ids_list],device=dev).detach()
        O_pred = fo[O_ids_list]
        O_loss = torch.sum((O_true-O_pred)**2)/len(O_ids_list)

        H_ids_list = torch.LongTensor([tid - len(O_ids) for tid in H_ids], device=dev)[0:3]
        H_true = y_opt[H_ids_list].detach()
        print('computing KGE embeddings (hidden)')
        for tid in tqdm(H_ids):
            triplet = id2triplet[tid]
            fh[tid-len(O_ids)] = kge_model(triplet.h,triplet.r,triplet.t)
        H_pred = fh[H_ids_list]
        H_loss = torch.sum((H_true-H_pred)**2)/len(H_ids_list)
        
        loss = (O_loss + H_loss)/2

        # if i % 100 == 0:
        print('E-loss: {}'.format(loss))
        print('y_opt: {}'.format(y_opt))
        print('--------------')
        loss.backward()
        print("*******")
        optimizer_E_step.step()
        print("========")

# device parameters
id2conf, triplet2id, id2triplet, hr2t, O_ids, H_ids, entities, relations, rule2groundings, rule2weight_idx, rules = load_data()
dev = 'cpu'

# Rule and Grounding Hyperparameters
w = torch.randn(len(rules), device=dev) # load this from M-step
h = torch.randn(len(H_ids), requires_grad=True, device=dev)

# EM Hyperparameters
iters_E_step = 1000
iters_y_opt = 3
learning_rate = 3

# KGE Hyperparameters
nentity = len(entities)
nrelation = len(relations)
model_name = 'TransE'
hidden_dim = 10
gamma = 0.05
kge_model = KGEModel(model_name, nentity, nrelation, hidden_dim, gamma)

input('Begin E-Step?')
run_E_step()
exit(0)
