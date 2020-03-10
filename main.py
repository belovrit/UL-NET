import os, gc, argparse, datetime
from src.prepare_data import prepare_data
from src.saver import *
from algo import e_step, m_step1, m_step2, m_step3
from eval import *


if __name__ == '__main__':
    main_parser = argparse.ArgumentParser()
    main_parser.add_argument("--data", type=str, default='cn15k',
                             help="the dir path where you store data")
    main_parser.add_argument("--preprocess", action="store_true")
    main_parser.add_argument("--iters_em", type=int, default=3)
    main_parser.add_argument("--model_name", type=str, default='TransE',
                             choices=['TransE', 'DistMult', 'ComplEx','RotatE'])
    main_parser.add_argument("--hidden_dim", type=int, default=300)
    main_parser.add_argument("--gamma", type=int, default=10)
    main_parser.add_argument("--iters_y_opt", type=int, default=30)
    main_parser.add_argument("--iters_e", type=int, default=50)
    main_parser.add_argument("--alpha_beta", type=float, default=1.0)
    main_parser.add_argument("--lr", type=float, default=1e-3)
    main_parser.add_argument("--iters_m", type=int, default=30)
    main_parser.add_argument("--device", type=str, default="cuda")
    main_parser.add_argument("--batch_size", type=int, default=1024)
    main_parser.add_argument("--load_model_path", type=str)
    main_parser.add_argument("--ranking", action="store_true")
    main_parser.add_argument("--zijies_update", type=bool, default=False)
    main_parser.add_argument("--include_mln", type=bool, default=False)

    main_args = main_parser.parse_args()


    if main_args.load_model_path is not None:
        workpath = main_args.load_model_path
        print("Loading trained model...")
        kge_model = load_trained_model(main_args.load_model_path)
        loaded_weights = load_rule_weights(main_args.load_model_path)
    else:
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

        # Initialize KGE Model
        n_entities = len(data_dict['entities'])
        n_relations = len(data_dict['relations'])
        kge_model = KGEModel(main_args.model_name, n_entities, n_relations,
                             main_args.hidden_dim, main_args.gamma)

        save_model_params(workpath, main_args, kge_model)


        # Initialize y_opt and w

        # w = torch.tensor(init_weights(data_dict["rule2weight_idx"]),
        #                  dtype=torch.float32,
        #                  device=main_args.device,
        #                  requires_grad=True)

        w = torch.tensor(np.random.uniform(0.0, 0.5, len(data_dict['rules'])),
                         dtype=torch.float32, device=main_args.device,
                         requires_grad=True)
        y_opt = torch.randn(len(data_dict['id2triplet']), dtype=torch.float32, requires_grad=True, device=main_args.device)

        print("Start training...")
        for i in range(main_args.iters_em):
            print("EM iteration {}".format(i))
            old_w = w
            print("Right Before e_step")
            id2betas, id2ystars = e_step(data_dict, main_args, w.detach(), y_opt, kge_model)
            # w = torch.tensor(init_weights(data_dict["rule2weight_idx"]),
            #                  dtype=torch.float32,
            #                  device=main_args.device,
            #                  requires_grad=True)
            w = torch.tensor(
                np.random.uniform(0.0, 0.5, len(data_dict['rules'])),
                dtype=torch.float32, device=main_args.device,
                requires_grad=True)

            gc.collect()
            m_step3(data_dict, id2betas, id2ystars, w, main_args)
            save_rule_weights(w.item(), workpath)

        save_trained_model(workpath, kge_model)

    print("Evaluating...")
    eval_result = {}
    tester = Tester(kge_model, main_args.mln_lambda)
    tester.load_test_triplets_conf_task(join(get_data_path(), main_args.data))
    eval_result["MSE"], eval_result["scores"] = tester.get_mse(main_args.alpha_beta)
    eval_result["MAE"] = tester.get_mae(main_args.alpha_beta)
    print("Mean Square Error: {}".format(eval_result["MSE"]))
    print("Mean Absolute Error: {}".format(eval_result["MAE"]))

    if main_args.ranking:
        tester.load_test_triplets_ranking_task(join(get_data_path(), main_args.data))
        eval_result["MnDCG"] = tester.mean_ndcg(tester.hr_map, main_args.alpha_beta)

    save_eval_result(eval_result, workpath)


