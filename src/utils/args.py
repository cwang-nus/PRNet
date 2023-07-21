import argparse

def get_public_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, help='train or test', default='train')
    parser.add_argument('--n_exp', type=int, default=0,
                        help='experiment index')
    parser.add_argument('--gpu', type=int, default=0, help='which gpu to run')
    parser.add_argument('--seed', type=int, default=2019)
    # data
    parser.add_argument('--dataset', type=str, default='TaxiBJ-P1',
                        choices=['TaxiBJ-P1', 'TaxiBJ-P2', 'TaxiBJ-P3',
                                 'TaxiBJ-P4', 'BikeNYC'])
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--prev_step', type=int, default=12)
    parser.add_argument('--pred_step', type=int, default=12)

    # training
    parser.add_argument('--max_epochs', type=int, default=250)
    parser.add_argument('--max_grad_norm', type=float, default=5.0)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--lr_decay_ratio', type=int, default=1)
    parser.add_argument('--lr_step_size', type=int, default=50)
    parser.add_argument('--datapath', type=str, default='./datasets',
                        help='datapath for dataset')
    parser.add_argument('--train_ratio', type=int, default=100,
                        help='how much ratio to to train')
    parser.add_argument('--log_dir', type=str, default='./log',
                        help='directory to save the log')
    parser.add_argument('--ext_flag', action='store_true',
                        help='whether to use external component')
    parser.add_argument('--exp_name', type=str, help='experiment name', default='')
    parser.add_argument('--save_iter', type=int, default=300)


    return parser