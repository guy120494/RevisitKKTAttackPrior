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


def generate_dataset(basis, n, scale=1.0, seed=None, device='cuda:0'):
    """
    Sample n points from a fixed subspace with a given basis.

    Parameters
    ----------
    basis : (d, k) ndarray
        Orthonormal basis spanning the subspace.
    n : int
        Number of points to sample.
    scale : float
        Standard deviation of the sampling distribution.
    seed : int or None
        Random seed.

    Returns
    -------
    points : (n, d) ndarray
        Points lying in the subspace.
    """
    rng = np.random.default_rng(seed)
    d, k = basis.shape

    # Sample coordinates in R^k
    coeffs = rng.normal(scale=scale, size=(n, k))

    # Map to R^d
    train_x = coeffs @ basis.T
    train_y = (np.sign(train_x[:, 0]).astype(np.float32) + 1) / 2
    train_x, train_y = torch.from_numpy(train_x), torch.from_numpy(train_y)
    train_x, train_y = move_to_type_device(train_x, train_y, device)

    coeffs = rng.normal(scale=scale, size=(n, k))

    test_x = coeffs @ basis.T
    test_y = (np.sign(test_x[:, 0]).astype(np.float32) + 1) / 2
    test_x, test_y = torch.from_numpy(test_x), torch.from_numpy(test_y)
    test_x, test_y = move_to_type_device(test_x, test_y, device)

    return train_x, train_y, test_x, test_y


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

    parent_dir = Path(args.datasets_dir) / 'sphere'
    parent_dir.mkdir(parents=True, exist_ok=True)
    train_file = parent_dir / f'train_radius_{args.train_gauss_init_scale}_center_{args.train_gauss_init_bias}'
    test_file = parent_dir / f'test_radius_{args.train_gauss_init_scale}_center_{args.train_gauss_init_bias}'
    if train_file.exists() and test_file.exists():
        train_x, train_y = torch.load(train_file)
        test_x, test_y = torch.load(test_file)
    else:
        train_x, train_y, test_x, test_y = generate_dataset(basis=np.eye(args.input_dim)[:, :args.input_dim // 2],
                                                            n=args.data_amount)
        torch.save([train_x, train_y], train_file)
        torch.save([test_x, test_y], test_file)

    return [(train_x, train_y)], [(test_x, test_y)], None
