from os.path import join
import pickle

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