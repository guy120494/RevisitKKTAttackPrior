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

    if args.run_mode == 'reconstruct':
        args.extraction_data_amount = args.extraction_data_amount_per_class * args.num_classes

    # for legacy:
    args.data_amount = args.data_per_class_train * args.num_classes
    args.data_use_test = True
    args.data_test_amount = 1000

    # Generate random samples directly
    train_x = np.random.randn(args.data_amount, args.input_dim)
    test_x = np.random.randn(args.data_test_amount, args.input_dim)

    # Compute k values and add to first dimension
    k_values = np.where(np.arange(args.data_amount) % 2 == 0, 1, 0) * (args.input_dim ** 0.33)
    train_x[:, 0] += k_values
    train_y = np.sign(k_values).astype(np.float32)

    train_x, train_y = torch.from_numpy(train_x), torch.from_numpy(train_y)
    train_x, train_y = move_to_type_device(train_x, train_y, args.device)

    k_values = np.where(np.arange(args.data_test_amount) % 2 == 0, 1, 0) * (args.input_dim ** 0.33)
    test_x[:, 0] += k_values
    test_y = np.sign(k_values).astype(np.float32)

    test_x, test_y = torch.from_numpy(test_x), torch.from_numpy(test_y)
    test_x, test_y = move_to_type_device(test_x, test_y, args.device)

    return [(train_x, train_y)], [(test_x, test_y)], None
