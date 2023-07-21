import os
from src.utils.helper import setup_seed, check_device, get_scaler, add_data_args, get_dataloader
from src.base.trainer import BaseTrainer
from src.utils.args import get_public_config

def get_config():
    parser = get_public_config()
    parser.add_argument('--model_name', type=str, default='prnet',
                        choices=['prnet'], help='which model to train')
    parser.add_argument('--n_layers', type=int, default=9)
    parser.add_argument('--n_filters', type=int, default=64)
    parser.add_argument('--base_lr', type=float, default=5e-4)
    parser.add_argument('--loss_fn', type=str, default='l1', choices=['l1', 'l2'])
    parser.add_argument('--s_flag', action='store_true',
                        help='whether to use spatial component')
    parser.add_argument('--c_flag', action='store_true',
                        help='whether to use closeness')
    parser.add_argument('--x_flag', action='store_true',
                        help='whether to the closeness segment')
    parser.add_argument('--scaler', type=int, default=50,
                        help='how much to scale the input')
    parser.add_argument('--weight_decay', '--wd', default=1e-4,
                        type=float, help='weight decay (default: 1e-4)')
    parser.add_argument('--s_r', default=8,
                        type=int, help='spatial remain dimension')
    parser.add_argument('--pred', type=str, default='conv',
                        help='which predictor to use')

    args = parser.parse_args()
    args = add_data_args(args, args.dataset)

    args.datapath = '{}/{}.npz'.format(args.datapath, args.dataset)

    setup_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    print(args)
    return args

def create_model(args):
    model_name = args.model_name
    if model_name == 'prnet':
        from src.models.prnet import PRNet
        model = PRNet(n_layers=args.n_layers,
                      n_filters=args.n_filters,
                      t_params=(args.prev_step, args.p, args.t),
                      s_flag=args.s_flag,
                      c_flag=args.c_flag,
                      x_flag=args.x_flag,
                      ext_flag=args.ext_flag,
                      s_r=args.s_r,
                      name=args.model_name,
                      dataset=args.dataset,
                      device=args.device,
                      height=args.height,
                      width=args.width,
                      prev_step=args.prev_step,
                      pred_step=args.pred_step,
                      input_dim=args.input_dim,
                      output_dim=args.output_dim)
    return model

def main():
    args = get_config()
    device = check_device()
    args.device = device
    model = create_model(args)
    data = get_dataloader(args.datapath,
                          get_scaler(args.scaler),
                          args.batch_size,
                          args.train_ratio,
                          args.mode,
                          args.ext_flag)

    trainer = BaseTrainer(model=model,
                          data=data,
                          dataset=args.dataset,
                          model_name=args.model_name,
                          exp_name=args.exp_name,
                          ext_flag=args.ext_flag,
                          base_lr=args.base_lr,
                          loss_fn=args.loss_fn,
                          lr_decay_ratio=args.lr_decay_ratio,
                          log_dir=args.log_dir,
                          n_exp=args.n_exp,
                          save_iter=args.save_iter,
                          clip_grad_value=args.max_grad_norm,
                          max_epochs=args.max_epochs,
                          patience=args.patience,
                          device=device)

    if args.mode == 'train':
        trainer.train()
    else:
        trainer.test(True, False, args.mode)

if __name__ == "__main__":
    main()