from torch.utils.data import DataLoader
import torch
import numpy as np
import random

from src.utils.scalers import Scaler

def get_scaler(scaler):
    return Scaler(scaler)

def add_data_args(args, dataset):
    if dataset == 'BikeNYC':
        args.height = 16
        args.width = 8
        args.input_dim = 2
        args.output_dim = 2
        args.save_iter = 10
        args.p = 3
        args.t = 3
    else:
        args.height = 32
        args.width = 32
        args.input_dim = 2
        args.output_dim = 2
        args.save_iter = 100
        args.p = 3
        args.t = 3
    return args

def get_dataloader(datapath, scaler, batch_size, train_ratio, mode='train', ext_flag=False):
    results = {}
    if mode == 'train':
        for category in ['train', 'val', 'test']:
            data = np.load(datapath, allow_pickle=True)
            Tensor = torch.FloatTensor
            XC = Tensor(scaler.transform(data['XC_'+category]))
            XP = Tensor(scaler.transform(data['XP_'+category]))
            XT = Tensor(scaler.transform(data['XT_'+category]))
            Y = Tensor(scaler.transform(data['Y_'+category]))
            YP = Tensor(scaler.transform(data['YP_'+category]))
            YT = Tensor(scaler.transform(data['YT_'+category]))
            if ext_flag:
                ext = Tensor(data['ext_'+category])

            gt = Y.unsqueeze(1) - YT

            if category == 'train':
                train_len = (train_ratio * len(XC)) // 100

                XC, XP, XT = XC[:train_len], XP[:train_len], XT[:train_len]
                Y, YP, YT = Y[:train_len], YP[:train_len], YT[:train_len]
                if ext_flag:
                    ext = ext[:train_len]
                gt = Y.unsqueeze(1) - YT

            assert len(XC) == len(Y)
            print('# {} samples: {}'.format(category, len(XC)))
            if ext_flag:
                data = torch.utils.data.TensorDataset(XC, XP, XT, Y, YP, YT, gt, ext)
            else:
                data = torch.utils.data.TensorDataset(XC, XP, XT, Y, YP, YT, gt)
            if category == 'test':
                shuffle_drop_flag = False
            else:
                shuffle_drop_flag = True

            results['{}_loader'.format(category)] = DataLoader(data, batch_size=batch_size, shuffle=shuffle_drop_flag, drop_last=shuffle_drop_flag)
    else:
        data = np.load(datapath, allow_pickle=True)
        Tensor = torch.FloatTensor
        XC = Tensor(scaler.transform(data['XC_'+mode]))
        XP = Tensor(scaler.transform(data['XP_'+mode]))
        XT = Tensor(scaler.transform(data['XT_'+mode]))
        Y = Tensor(scaler.transform(data['Y_'+mode]))
        YP = Tensor(scaler.transform(data['YP_'+mode]))
        YT = Tensor(scaler.transform(data['YT_'+mode]))

        if ext_flag:
            ext = Tensor(data['ext_'+mode])

        gt = Y.unsqueeze(1) - YT

        assert len(XC) == len(Y)
        print('# {} samples: {}'.format(mode, len(XC)))
        if ext_flag:
            data = torch.utils.data.TensorDataset(XC, XP, XT, Y, YP, YT, gt, ext)
        else:
            data = torch.utils.data.TensorDataset(XC, XP, XT, Y, YP, YT, gt)
        results['{}_loader'.format(mode)] = DataLoader(data, batch_size=batch_size, shuffle=False, drop_last=False)

    results['scaler'] = scaler
    return results


def check_device(device=None):
    if device is None:
        print("`device` is not set, will train and evaluate the model on default device.")
        if torch.cuda.is_available():
            print("cuda device is available, place the model on the device.")
            return torch.device("cuda")
        else:
            print("cuda device is not available, place the model on cpu.")
            return torch.device("cpu")
    else:
        if isinstance(device, torch.device):
            return device
        else:
            return torch.device(device)

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False