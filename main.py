""" Main file to run the code """
import argparse
from src.base import main_worker

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Clean Train')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--rank', default=0, type=int)
    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N')

    parser.add_argument('--root', default='./dataset', type=str)
    parser.add_argument('--dataset', default='cifar10', type=str)
    parser.add_argument('--arch', default='convnet', type=str)
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--inner_optim', default='Adam', type=str)
    parser.add_argument('--outer_optim', default='Adam', type=str)
    parser.add_argument('--decay', action='store_true')
    parser.add_argument('--inner_lr', default=0.01, type=float, help='inner learning rate')
    parser.add_argument('--label_lr_scale', default=1, type=float, help='scale the label lr')
    parser.add_argument('--distill_batch_size', default=10, type=int,
                        help='number of random patches to be selected each step for the distilled data')
    parser.add_argument('--window', default=60, type=int, help='Number of unrolling computing gradients')
    parser.add_argument('--minwindow', default=0, type=int, help='Start unrolling from steps x')
    parser.add_argument('--totwindow', default=200, type=int, help='Number of total unrolling computing gradients')
    parser.add_argument('--num_train_eval', default=10, type=int, help='Num of training of network for evaluation')
    parser.add_argument('--train_y', action='store_true')
    parser.add_argument('--train_lr', action='store_true')
    parser.add_argument('--batch_size', default=200, type=int)
    parser.add_argument('--eps', default=1e-8, type=float)
    parser.add_argument('--wd', default=0, type=float)
    parser.add_argument('--test_freq', default=5, type=int)
    parser.add_argument('--print_freq', default=20, type=int)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--ddtype', default='curriculum', type=str)
    parser.add_argument('--cctype', default=0, type=int)
    parser.add_argument('--zca', action='store_true')
    parser.add_argument('--wandb', action='store_false')
    parser.add_argument('--clip_coef', default=0.9, type=float)
    parser.add_argument('--name', default='test', type=str)
    parser.add_argument('--comp_aug', action='store_true')
    parser.add_argument('--comp_aug_real', action='store_true')
    parser.add_argument('--syn_strategy', default='flip_rotate', type=str)
    parser.add_argument('--real_strategy', default='flip_rotate', type=str)
    parser.add_argument('--steps_per_epoch', default=10, type=int)
    parser.add_argument('--update_steps', default=1, type=int)
    parser.add_argument('--batch_update_steps', default=1, type=int)

    parser.add_argument('--comp_ipc', type=int, default=10, help='number of images-per-class baseline compare')
    parser.add_argument('--class_area_width', type=int, default=120, help='width in pixels of the mosaic')
    parser.add_argument('--class_area_height', type=int, default=120, help='height in pixels of the mosaic')
    parser.add_argument('--load_poster_run_name', type=str, default='',
                        help='an image path to initialize the poster from')
    parser.add_argument('--poster_class_num_x', type=int, default=5,
                        help='number of classes in the poster in the X axis')
    parser.add_argument('--poster_class_num_y', type=int, default=2,
                        help='number of classes in the poster in the Y axis')
    parser.add_argument('--poster_width', type=int, default=475, help='number of classes in the poster in the Y axis')
    parser.add_argument('--poster_height', type=int, default=210, help='number of classes in the poster in the Y axis')
    parser.add_argument('--patch_num_x', type=int, default=16, help='number of patches in x direction')
    parser.add_argument('--patch_num_y', type=int, default=6, help='number of patches in y direction')

    args = parser.parse_args()

    main_worker(args)
