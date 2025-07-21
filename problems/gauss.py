from pathlib import Path

import numpy as np
import torch


def move_to_type_device(x, y, device):
    print('X:', x.shape)
    print('y:', y.shape)
    x = x.to(torch.get_default_dtype())
    y = y.to(torch.get_default_dtype())
    x, y = x.to(device), y.to(device)
    return x, y


def get_dataloader(args):
    args.input_dim = 28 * 28
    args.num_classes = 2
    args.output_dim = 1
    args.dataset = 'gauss'

    if args.run_mode == 'reconstruct' or args.run_mode == 'train_reconstruct':
        args.extraction_data_amount = args.extraction_data_amount_per_class * args.num_classes

    # for legacy:
    args.data_amount = args.data_per_class_train * args.num_classes
    args.data_use_test = True
    args.data_test_amount = 1000

    # Generate random samples directly
    train_x = np.random.randn(args.data_amount,
                              args.input_dim) * args.train_gauss_init_scale + args.train_gauss_init_bias
    train_y = (np.sign(train_x[:, 0]).astype(np.float32) + 1) / 2

    train_x, train_y = torch.from_numpy(train_x), torch.from_numpy(train_y)
    train_x, train_y = move_to_type_device(train_x, train_y, args.device)

    test_x = np.random.randn(args.data_test_amount,
                             args.input_dim) * args.train_gauss_init_scale + args.train_gauss_init_bias
    test_y = (np.sign(test_x[:, 0]).astype(np.float32) + 1) / 2

    test_x, test_y = torch.from_numpy(test_x), torch.from_numpy(test_y)
    test_x, test_y = move_to_type_device(test_x, test_y, args.device)

    parent_dir = Path(args.datasets_dir) / 'gauss'
    parent_dir.mkdir(parents=True, exist_ok=True)

    torch.save([train_x, train_y], parent_dir / 'train')
    torch.save([test_x, test_y], parent_dir / 'test')

    return [(train_x, train_y)], [(test_x, test_y)], None
