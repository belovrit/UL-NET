from os.path import join, getctime
import pickle, csv
import torch
import glob
from KGEModel import *

def save_preprocessed(dict_obj, save_path):
    path = join(save_path, 'data_dict.pickle')
    print("Saving data pickle...")
    with open(path, 'wb') as fp:
        pickle.dump(dict_obj, fp)


def load_preprocessed(save_path):
    path = join(save_path, 'data_dict.pickle')
    print("Loading data pickle...")
    with open(path, 'rb') as fp:
        data_dict = pickle.load(fp)

    return data_dict


def save_trained_model(save_path, trained_model, iter=None):
    iter = "_iter_{}".format(iter) if iter is not None else ""
    p = join(save_path, 'trained_model{}.pt'.format(iter))
    torch.save(trained_model.state_dict(), p)
    print('Trained model saved to {}'.format(p))


def save_model_params(save_path, main_args, model):
    p = join(save_path, 'main_args.txt')
    with open(p, 'w') as fi:
        wr = csv.writer(fi, delimiter='\t')
        for arg in vars(main_args):
            wr.writerow([arg, getattr(main_args, arg)])
        wr.writerow(['n_entities', model.nentity])
        wr.writerow(['n_relations', model.nrelation])


def get_model_params(load_path):
    fi = open(load_path)
    rd = csv.reader(fi, delimiter='\t')

    for ln in rd:
        param, value = ln
        if param == 'model_name':
            model_name = value
        if param == 'n_entities':
            n_entities = value
        if param == 'n_relations':
            n_relations = value
        if param == 'hidden_dim':
            hidden_dim = value
        if param == 'gamma':
            gamma = value

    return model_name, int(n_entities), int(n_relations), int(hidden_dim), float(gamma)


def load_trained_model(load_path):
    p = join(load_path, 'trained_model*')
    f = join(load_path, 'main_args.txt')
    files = glob.glob(p)
    best_trained_model_path = max(files, key=getctime)
    model_name, n_entities, n_relations, hidden_dim, gamma = get_model_params(f)
    trained_model = KGEModel(model_name, n_entities, n_relations, hidden_dim, gamma)
    trained_model.load_state_dict(torch.load(best_trained_model_path))
    return trained_model
