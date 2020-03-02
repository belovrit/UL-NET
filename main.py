import os, gc
from src.prepare_data import prepare_data
from src.saver import *
#from config import FLAGS
import argparse
from src.utils import *
import datetime
from algo import e_step, m_step

# if './' not in sys.path:
#     sys.path.append('./')
# if './src' not in sys.path:
#     sys.path.append('./src')

if __name__ == '__main__':
    main_parser = argparse.ArgumentParser()
    main_parser.add_argument("--data", type=str, default='cn15k',
                             help="the dir path where you store data \
                            (train.tsv, val.tsv, test.tsv). Default: ppi5k")
    main_parser.add_argument("--preprocess", action="store_true")
    main_parser.add_argument("--iters", type=int, default=2)
    main_parser.add_argument("--alpha", type=float, default=1.0)
    main_parser.add_argument("--lr", type=float, default=1e-3)
    main_args = main_parser.parse_args()

    time = str(datetime.datetime.now()).replace(' ', '_')
    workpath = os.path.join(get_root_path(), 'record', time)
    data_path = join(get_data_path(), main_args.data)
    save_path = os.path.join(get_save_path(), main_args.data)
    ensure_dir(workpath)
    ensure_dir(save_path)

    if main_args.preprocess:
        data_dict = prepare_data(data_path)
        gc.collect()
        save_preprocessed(data_dict, save_path)
    else:
        data_dict = load_preprocessed(save_path)

    data_dict['weights'] = init_weights(data_dict['rules'])

    print("Start training...")
    for i in range(main_args.iters):
        print("EM iteration {}".format(i))
        id2betas, id2ystars = e_step(data_dict, main_args.alpha)
        new_weights = m_step(id2betas, id2ystars, main_args.lr, main_args.alpha)
        data_dict['weights'] = new_weights

    #print("Evaluating...")