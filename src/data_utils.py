""" Utils for data processing """

import os
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
import torchvision.transforms as transforms

from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.datasets.utils import download_url
from torchvision.datasets.folder import default_loader
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder

from src.convnet import ConvNet, ConvNet2

# change paths as needed:
DATA_PATHS = {'tiny-imagenet-200': '../datasets/tiny-imagenet-200/tiny-imagenet-200',
              'cub-200': '../datasets/CUB200'}


# functions:
def get_arch(arch, num_classes, channel, im_size):
    """ Returns a network for the given architecture """
    if arch == 'convnet':
        net_width, net_depth, net_act, net_norm, net_pooling = 128, 3, 'relu', 'instancenorm', 'avgpooling'
        return ConvNet(channel, num_classes, net_width, net_depth, net_act, net_norm, net_pooling, im_size=im_size)
    if arch == 'convnet2':
        net_width, net_depth, net_act, net_norm, net_pooling = 128, 3, 'relu', 'none', 'avgpooling'
        return ConvNet(channel, num_classes, net_width, net_depth, net_act, net_norm, net_pooling, im_size=im_size)
    if arch == 'convnet4':
        net_width, net_depth, net_act, net_norm, net_pooling = 128, 4, 'relu', 'instancenorm', 'avgpooling'
        return ConvNet(channel, num_classes, net_width, net_depth, net_act, net_norm, net_pooling, im_size=im_size)
    if arch == 'convnetw':
        net_width, net_depth, net_act, net_norm, net_pooling = 128, 3, 'relu', 'instancenorm', 'avgpooling'
        return ConvNet2(channel, num_classes, net_width, net_depth, net_act, net_norm, net_pooling, im_size=im_size)
    if arch == 'convnetw4':
        net_width, net_depth, net_act, net_norm, net_pooling = 128, 4, 'relu', 'instancenorm', 'avgpooling'
        return ConvNet2(channel, num_classes, net_width, net_depth, net_act, net_norm, net_pooling, im_size=im_size)
    if arch == 'convnetw2':
        net_width, net_depth, net_act, net_norm, net_pooling = 128, 3, 'relu', 'batchnorm', 'avgpooling'
        return ConvNet2(channel, num_classes, net_width, net_depth, net_act, net_norm, net_pooling, im_size=im_size)

    raise NotImplementedError


def get_dataset(dataset, root, transform_train, transform_test, zca=False):
    """ Returns the dataset, the number of classes, and the shape of the images """
    process_config = None

    if zca:
        print('Using ZCA')

    if dataset == 'cifar10':
        num_classes = 10
        shape = [3, 32, 32]
        if zca:
            trainset = CIFAR10Dataset(root=root, download=True)
            trainset_test = CIFAR10Dataset(root=root, download=True)
            testset = CIFAR10Dataset(root=root, train=False, download=True)
            trainset.data, testset.data, process_config = preprocess(trainset.data, testset.data, regularization=0.1)
            trainset_test.data = trainset.data.clone()
        else:
            trainset = CIFAR10(root=root, download=True, transform=transform_train)
            trainset_test = CIFAR10(root=root, download=True, transform=transform_test)
            testset = CIFAR10(root=root, train=False, download=True, transform=transform_test)

    elif dataset == 'cifar100':
        num_classes = 100
        shape = [3, 32, 32]
        if zca:
            trainset = CIFAR100Dataset(root=root, download=True)
            testset = CIFAR100Dataset(root=root, train=False, download=True)
            trainset.data, testset.data, process_config = preprocess(trainset.data, testset.data, regularization=0.1)
            trainset_test = trainset
        else:
            trainset = CIFAR100(root=root, download=True, transform=transform_train)
            trainset_test = CIFAR100(root=root, download=True, transform=transform_test)
            testset = CIFAR100(root=root, train=False, download=True, transform=transform_test)

    elif dataset == 'tiny-imagenet-200':
        num_classes = 200
        shape = [3, 64, 64]

        root = DATA_PATHS[dataset]
        train_dir = os.path.join(root, 'train')
        val_dir = os.path.join(root, 'val')

        if zca:
            train_image_folder_set = ImageFolder(train_dir, transform=transforms.ToTensor())
            test_image_folder_set = ImageFolder(val_dir, transform=transforms.ToTensor())

            train_data = [(img, target) for img, target in tqdm(DataLoader(train_image_folder_set, batch_size=1024))]
            trainset = TensorDataset(torch.vstack([img for img, _ in train_data]).squeeze(),
                                     torch.hstack([target for _, target in train_data]).squeeze().long())

            test_data = [(img, target) for img, target in tqdm(DataLoader(test_image_folder_set, batch_size=1024))]
            testset = TensorDataset(torch.vstack([img for img, _ in test_data]).squeeze(),
                                    torch.hstack([target for _, target in test_data]).squeeze().long())

            trainset.data_tensor, testset.data_tensor, process_config = \
                preprocess(trainset.data_tensor, testset.data_tensor, regularization=0.1, permute=False)

            trainset_test = trainset

        else:
            raise NotImplementedError

    elif dataset == 'cub-200':
        num_classes = 200
        shape = [3, 32, 32]
        root = DATA_PATHS[dataset]

        transform = transforms.Compose([transforms.Resize(shape[1:]), transforms.ToTensor()])

        trainset = Cub200(root, transform=transform)
        testset = Cub200(root, train=False, transform=transform)

        if zca:
            trainset.data.data, testset.data.data, process_config = \
                preprocess(trainset.data.data, testset.data.data, regularization=0.1, permute=False)

        trainset_test = trainset

    else:
        raise NotImplementedError

    return trainset, trainset_test, testset, num_classes, shape, process_config


def get_transform(dataset):
    """ Returns the default transformation for the given dataset """
    print(dataset)
    if dataset == 'cifar10':
        default_transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        default_transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        print('the dataset is cifar10')

    elif dataset == 'cifar100':
        default_transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        default_transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        print('the dataset is cifar100')

    elif dataset == 'tiny-imagenet-200':
        default_transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        default_transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        print('the dataset is tiny-imagenet-200')

    elif dataset == 'cub-200':
        default_transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        default_transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        print('the dataset is cub-200-2011')

    else:
        raise NotImplementedError

    return default_transform_train, default_transform_test


def preprocess(train, test, zca_bias=0, regularization=0, permute=True):
    """ Preprocesses the data using ZCA whitening """
    if not permute:
        train = train.permute(0, 2, 3, 1).contiguous()
        test = test.permute(0, 2, 3, 1).contiguous()

    origTrainShape = train.shape
    origTestShape = test.shape

    train = np.ascontiguousarray(train, dtype=np.float32).reshape(train.shape[0], -1).astype('float64')
    test = np.ascontiguousarray(test, dtype=np.float32).reshape(test.shape[0], -1).astype('float64')

    nTrain = train.shape[0]

    train_mean = np.mean(train, axis=1)[:, np.newaxis]

    # Zero mean every feature
    train = train - np.mean(train, axis=1)[:, np.newaxis]
    test = test - np.mean(test, axis=1)[:, np.newaxis]

    # Normalize
    train_norms = np.linalg.norm(train, axis=1)
    test_norms = np.linalg.norm(test, axis=1)

    # Make features unit norm
    train = train / train_norms[:, np.newaxis]
    test = test / test_norms[:, np.newaxis]

    trainCovMat = 1.0 / nTrain * train.T.dot(train)

    print(f'Computing ZCA, cov shape{trainCovMat.shape}')
    (E, V) = np.linalg.eig(trainCovMat)

    E += zca_bias
    sqrt_zca_eigs = np.sqrt(E + regularization * np.sum(E) / E.shape[0])
    inv_sqrt_zca_eigs = np.diag(np.power(sqrt_zca_eigs, -1))
    global_ZCA = V.dot(inv_sqrt_zca_eigs).dot(V.T)
    inverse_ZCA = V.dot(np.diag(sqrt_zca_eigs)).dot(V.T)

    train = train.dot(global_ZCA)
    test = test.dot(global_ZCA)

    train_tensor = torch.Tensor(train.reshape(origTrainShape).astype('float64'))
    test_tensor = torch.Tensor(test.reshape(origTestShape).astype('float64'))

    # if permute:
    train_tensor = train_tensor.permute(0, 3, 1, 2).contiguous()
    test_tensor = test_tensor.permute(0, 3, 1, 2).contiguous()

    return train_tensor, test_tensor, (inverse_ZCA, global_ZCA, train_norms, train_mean)


def init_gaussian(num_classes, ipc, tensor_length):
    """ directly initialize the tensors with Gaussian distribution """
    # initialize the tensors
    tensors = torch.zeros(num_classes * ipc, tensor_length)

    # initialize the class means with the standard Gaussian distribution
    # the variance is just identity matrix
    class_means = torch.normal(torch.zeros(num_classes, tensor_length), torch.ones(tensor_length))

    # calculate the minimum distance between the class means
    min_dist = float('inf')
    for i in range(num_classes):
        for j in range(i + 1, num_classes):
            dist = torch.dist(class_means[i], class_means[j])
            if dist < min_dist:
                min_dist = dist
    # calculate the variance
    variance = min_dist / 4

    # initialize the tensors
    for i in range(num_classes):
        for j in range(ipc):
            tensors[i * ipc + j] = torch.normal(class_means[i], variance * torch.ones(tensor_length))

    return tensors


def project(data, pgd_coef=1):
    """ project the data to a unit ball """
    data_norm = torch.reshape(torch.norm(torch.flatten(data, start_dim=1, end_dim=-1), dim=-1),
                              [data.shape[0], *[1] * (data.dim() - 1)])
    return data / data_norm * pgd_coef


def format_tiny_imagenet_val(root):
    """ Formats the tiny-imagenet-200 validation set to match the train set structure """
    val_dir = os.path.join(root, 'val')
    print(f'Formatting: {val_dir}')
    val_annotations = os.path.join(val_dir, 'val_annotations.txt')

    val_dict = {}
    with open(val_annotations, 'r') as f:
        for line in tqdm(f):
            line = line.strip().split()
            assert (len(line) == 6)
            wnind = line[1]
            img_name = line[0]
            boxes = '\t'.join(line[2:])
            if wnind not in val_dict:
                val_dict[wnind] = []
            entries = val_dict[wnind]
            entries.append((img_name, boxes))

    assert (len(val_dict) == 200)

    for wnind, entries in val_dict.items():
        val_wnind_dir = os.path.join(val_dir, wnind)
        val_images_dir = os.path.join(val_dir, 'images')
        val_wnind_images_dir = os.path.join(val_wnind_dir, 'images')
        os.mkdir(val_wnind_dir)
        os.mkdir(val_wnind_images_dir)
        wnind_boxes = os.path.join(val_wnind_dir, f'{wnind}_boxes.txt')
        f = open(wnind_boxes, 'w')
        for img_name, box in entries:
            source = os.path.join(val_images_dir, img_name)
            dst = os.path.join(val_wnind_images_dir, img_name)
            os.system(f'cp {source} {dst}')
            f.write(f'{img_name}\t{box}\n')
        f.close()
    print(f'Cleaning up: {val_images_dir}')
    os.system(f'rm -rf {val_images_dir}')
    print('Formatting val done')


# aug:
class ImageIntervention(object):
    """ class for intervening the data """

    def __init__(self, name, strategy, phase, not_single=False):
        self.name = name
        self.phase = phase
        self.not_single = not_single
        self.flip = False
        self.color = False
        self.cutout = False
        if self.name in ['syn_aug', 'real_aug', 'pair_aug']:
            self.functions = {
                'scale': self.diff_scale,
                'flip': self.diff_flip,
                'rotate': self.diff_rotate,
                'crop': self.diff_crop,
                'color': [self.diff_brightness, self.diff_saturation, self.diff_contrast],
                'cutout': self.diff_cutout,
            }
            self.prob_flip = 0.5
            self.ratio_scale = 1.2
            self.ratio_rotate = 15.0
            self.ratio_crop_pad = 0.125
            self.ratio_cutout = 0.5  # the size would be 0.5x0.5
            self.ratio_noise = 0.05
            self.brightness = 1.0
            self.saturation = 2.0
            self.contrast = 0.5

            self.keys = list(strategy.split('_'))
            for key in self.keys:
                if key == 'flip' and not_single == True:
                    self.flip = True
                    self.keys.remove(key)
                elif key == 'color' and not_single == True:
                    self.color = True
                    self.keys.remove(key)
                elif key == 'cutout' and not_single == True:
                    self.cutout = True
                    self.keys.remove(key)

        elif self.name != 'none':
            raise NotImplementedError

    def __call__(self, x, dtype):
        if self.name == 'none':
            return x

        elif self.name == 'syn_aug':
            if dtype == 'real':
                return x
            elif dtype == 'syn':
                return self.do(x)
            else:
                raise NotImplementedError

        elif self.name == 'real_aug':
            if dtype == 'syn':
                return x
            elif dtype == 'real':
                return self.do(x)
            else:
                raise NotImplementedError

        elif self.name == 'pair_aug':
            return self.do(x)

    def do(self, x):
        if not self.not_single:
            intervention = self.keys[np.random.randint(0, len(self.keys), size=(1,))[0]]

            if intervention == 'color':
                function = self.functions['color'][np.random.randint(0, len(self.functions['color']), size=(1,))[0]]
            else:
                function = self.functions[intervention]

            x = function(x)
        else:
            if self.flip:
                x = self.diff_flip(x)
            if self.color:
                for f in self.functions['color']:
                    x = f(x)
            if len(self.keys) > 0:
                intervention = self.keys[np.random.randint(0, len(self.keys), size=(1,))[0]]
                function = self.functions[intervention]
                x = function(x)

            if self.cutout:
                x = self.diff_cutout(x)

        return x

    def diff_scale(self, x):
        # x>1, max scale
        # sx, sy: (0, +oo), 1: orignial size, 0.5: enlarge 2 times
        ratio = self.ratio_scale
        sx = torch.Tensor(np.random.rand(x.shape[0]) * (ratio - 1.0 / ratio) + 1.0 / ratio)
        sy = torch.Tensor(np.random.rand(x.shape[0]) * (ratio - 1.0 / ratio) + 1.0 / ratio)
        theta = [[[sx[i], 0, 0],
                  [0, sy[i], 0], ] for i in range(x.shape[0])]
        theta = torch.tensor(theta, dtype=torch.float)
        if self.phase == 'train' and self.name == 'pair_aug':
            theta[:] = theta[0]
        grid = F.affine_grid(theta, x.shape, align_corners=False).to(x.device)
        x = F.grid_sample(x, grid, align_corners=False)
        return x

    def diff_flip(self, x):
        prob = self.prob_flip
        randf = torch.Tensor(np.random.rand(x.size(0), 1, 1, 1)).to(x.device)
        if self.phase == 'train' and self.name == 'pair_aug':
            randf[:] = randf[0]
        return torch.where(randf < prob, x.flip(3), x)

    def diff_rotate(self, x):
        ratio = self.ratio_rotate
        theta = torch.Tensor(np.random.rand(x.shape[0]) - 0.5) * 2 * ratio / 180 * float(np.pi)
        theta = [[[torch.cos(theta[i]), torch.sin(-theta[i]), 0],
                  [torch.sin(theta[i]), torch.cos(theta[i]), 0], ] for i in range(x.shape[0])]
        theta = torch.tensor(theta, dtype=torch.float)
        if self.phase == 'train' and self.name == 'pair_aug':
            theta[:] = theta[0]
        grid = F.affine_grid(theta, x.shape, align_corners=False).to(x.device)
        x = F.grid_sample(x, grid, align_corners=False)
        return x

    def diff_crop(self, x):
        # The image is padded on its surrounding and then cropped.
        ratio = self.ratio_crop_pad
        shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
        translation_x = torch.Tensor(np.random.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1])).to(
            x.device).long()
        translation_y = torch.Tensor(np.random.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1])).to(
            x.device).long()
        if self.phase == 'train' and self.name == 'pair_aug':
            translation_x[:] = translation_x[0]
            translation_y[:] = translation_y[0]
        grid_batch, grid_x, grid_y = torch.meshgrid(
            torch.arange(x.size(0), dtype=torch.long, device=x.device),
            torch.arange(x.size(2), dtype=torch.long, device=x.device),
            torch.arange(x.size(3), dtype=torch.long, device=x.device),
        )
        grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
        grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
        x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
        x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2)
        return x

    def diff_brightness(self, x):
        ratio = self.brightness
        randb = torch.Tensor(np.random.rand(x.size(0), 1, 1, 1)).to(x.device)
        if self.phase == 'train' and self.name == 'pair_aug':
            randb[:] = randb[0]
        x = x + (randb - 0.5) * ratio
        return x

    def diff_saturation(self, x):
        ratio = self.saturation
        x_mean = x.mean(dim=1, keepdim=True)
        rands = torch.Tensor(np.random.rand(x.size(0), 1, 1, 1)).to(x.device)
        if self.phase == 'train' and self.name == 'pair_aug':
            rands[:] = rands[0]
        x = (x - x_mean) * (rands * ratio) + x_mean
        return x

    def diff_contrast(self, x):
        ratio = self.contrast
        x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
        randc = torch.Tensor(np.random.rand(x.size(0), 1, 1, 1)).to(x.device)
        if self.phase == 'train' and self.name == 'pair_aug':
            randc[:] = randc[0]
        x = (x - x_mean) * (randc + ratio) + x_mean
        return x

    def diff_cutout(self, x):
        ratio = self.ratio_cutout
        cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
        offset_x = torch.Tensor(
            np.random.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1])).to(x.device).long()
        offset_y = torch.Tensor(
            np.random.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1])).to(x.device).long()
        if self.phase == 'train' and self.name == 'pair_aug':
            offset_x[:] = offset_x[0]
            offset_y[:] = offset_y[0]
        grid_batch, grid_x, grid_y = torch.meshgrid(
            torch.arange(x.size(0), dtype=torch.long, device=x.device),
            torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
            torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
        )
        grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
        grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
        mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
        mask[grid_batch, grid_x, grid_y] = 0
        x = x * mask.unsqueeze(1)
        return x


# datasets:
class CIFAR10Dataset(CIFAR10):
    """ CIFAR10 dataset """
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


class CIFAR100Dataset(CIFAR100):
    """ CIFAR100 dataset """
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


class TensorDataset(torch.utils.data.Dataset):
    """ Tensor dataset """
    def __init__(self, data_tensor, target_tensor):
        assert data_tensor.size(0) == target_tensor.size(0), "Data and targets must have the same number of samples"
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor

    def __len__(self):
        return self.data_tensor.size(0)

    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor[index]


class Cub200(Dataset):
    """ CUB-200-2011 dataset """
    base_folder = 'CUB_200_2011/images'
    url = 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'
    filename = 'CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'

    def __init__(self, root, train=True, transform=None, download=True, load_to_mem=True):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.train = train
        self.loader = default_loader
        self.loaded_to_mem = False

        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')
        if load_to_mem:
            self._load_and_process_images()

    def _load_metadata(self):
        images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])

        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')

        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

        self.data.target -= 1  # Targets start at 1 by default, so shift to 0

    def _load_and_process_images(self):
        print('Loading and processing images...')
        self.targets = torch.tensor(self.data.target.values).long()
        self.data = torch.stack((self.data['filepath'].apply(lambda x:
                                 self.transform(self.loader(os.path.join(self.root, self.base_folder, x))))).to_list())

        self.loaded_to_mem = True

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False

        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, self.base_folder, row.filepath)
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True

    def _download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if not self.loaded_to_mem:
            sample = self.data.iloc[idx]
            target = sample.target
            path = os.path.join(self.root, self.base_folder, sample.filepath)
            img = self.loader(path)

            if self.transform is not None:
                img = self.transform(img)
        else:
            img = self.data[idx]
            target = self.targets[idx]

        return img, target


CIFAR10_LABELS_DICT = {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4,
                       'dog': 5, 'frog': 6, 'horse': 7, 'boat': 8, 'truck': 9}

CIFAR100_LABELS_DICT = {'apple': 0, 'aquarium_fish': 1, 'baby': 2, 'bear': 3, 'beaver': 4,
                        'bed': 5, 'bee': 6, 'beetle': 7, 'bicycle': 8, 'bottle': 9,
                        'bowl': 10, 'boy': 11, 'bridge': 12, 'bus': 13, 'butterfly': 14,
                        'camel': 15, 'can': 16, 'castle': 17, 'caterpillar': 18,
                        'cattle': 19, 'chair': 20, 'chimpanzee': 21, 'clock': 22,
                        'cloud': 23, 'cockroach': 24, 'computer_keyboard': 39,
                        'couch': 25, 'crab': 26, 'crocodile': 27, 'cup': 28,
                        'dinosaur': 29, 'dolphin': 30, 'elephant': 31, 'flatfish': 32,
                        'forest': 33, 'fox': 34, 'girl': 35, 'hamster': 36, 'house': 37,
                        'kangaroo': 38, 'lamp': 40, 'lawn mower': 41, 'leopard': 42,
                        'lion': 43, 'lizard': 44, 'lobster': 45, 'man': 46, 'maple tree': 47,
                        'motorcycle': 48, 'mountain': 49, 'mouse': 50, 'mushroom': 51,
                        'oak_tree': 52, 'orange': 53, 'orchid': 54, 'otter': 55,
                        'palm_tree': 56, 'pear': 57, 'pickup_truck': 58, 'pine_tree': 59,
                        'plain': 60, 'plate': 61, 'poppy': 62, 'porcupine': 63, 'possum': 64,
                        'rabbit': 65, 'raccoon': 66, 'ray': 67, 'road': 68, 'rocket': 69,
                        'rose': 70, 'sea': 71, 'seal': 72, 'shark': 73, 'shrew': 74,
                        'skunk': 75, 'skyscraper': 76, 'snail': 77, 'snake': 78, 'spider': 79,
                        'squirrel': 80, 'streetcar': 81, 'sunflower': 82, 'sweet_pepper': 83,
                        'table': 84, 'tank': 85, 'telephone': 86, 'television': 87, 'tiger': 88,
                        'tractor': 89, 'train': 90, 'trout': 91, 'tulip': 92, 'turtle': 93,
                        'wardrobe': 94, 'whale': 95, 'willow_tree': 96, 'wolf': 97, 'woman': 98,
                        'worm': 99}

CUB200_LABELS_DICT = {'Black_footed_Albatross': 1, 'Laysan_Albatross': 2, 'Sooty_Albatross': 3,
                      'Groove_billed_Ani': 4, 'Crested_Auklet': 5, 'Least_Auklet': 6, 'Parakeet_Auklet': 7,
                      'Rhinoceros_Auklet': 8, 'Brewer_Blackbird': 9, 'Red_winged_Blackbird': 10,  'Rusty_Blackbird': 11,
                      'Yellow_headed_Blackbird': 12,  'Bobolink': 13,  'Indigo_Bunting': 14,  'Lazuli_Bunting': 15,
                      'Painted_Bunting': 16,  'Cardinal': 17,  'Spotted_Catbird': 18,  'Gray_Catbird': 19,
                      'Yellow_breasted_Chat': 20,  'Eastern_Towhee': 21,  'Chuck_will_Widow': 22,
                      'Brandt_Cormorant': 23,  'Red_faced_Cormorant': 24,  'Pelagic_Cormorant': 25,
                      'Bronzed_Cowbird': 26,  'Shiny_Cowbird': 27,  'Brown_Creeper': 28,  'American_Crow': 29,
                      'Fish_Crow': 30,  'Black_billed_Cuckoo': 31,  'Mangrove_Cuckoo': 32,  'Yellow_billed_Cuckoo': 33,
                      'Gray_crowned_Rosy_Finch': 34,  'Purple_Finch': 35,  'Northern_Flicker': 36,
                      'Acadian_Flycatcher': 37,  'Great_Crested_Flycatcher': 38,  'Least_Flycatcher': 39,
                      'Olive_sided_Flycatcher': 40,  'Scissor_tailed_Flycatcher': 41,  'Vermilion_Flycatcher': 42,
                      'Yellow_bellied_Flycatcher': 43,  'Frigatebird': 44,  'Northern_Fulmar': 45,  'Gadwall': 46,
                      'American_Goldfinch': 47,  'European_Goldfinch': 48,  'Boat_tailed_Grackle': 49,
                      'Eared_Grebe': 50,  'Horned_Grebe': 51,  'Pied_billed_Grebe': 52,  'Western_Grebe': 53,
                      'Blue_Grosbeak': 54,  'Evening_Grosbeak': 55,  'Pine_Grosbeak': 56,  'Rose_breasted_Grosbeak': 57,
                      'Pigeon_Guillemot': 58,  'California_Gull': 59,  'Glaucous_winged_Gull': 60,  'Heermann_Gull': 61,
                      'Herring_Gull': 62,  'Ivory_Gull': 63,  'Ring_billed_Gull': 64,  'Slaty_backed_Gull': 65,
                      'Western_Gull': 66,  'Anna_Hummingbird': 67,  'Ruby_throated_Hummingbird': 68,
                      'Rufous_Hummingbird': 69,  'Green_Violetear': 70,  'Long_tailed_Jaeger': 71,
                      'Pomarine_Jaeger': 72,  'Blue_Jay': 73,  'Florida_Jay': 74,  'Green_Jay': 75,
                      'Dark_eyed_Junco': 76,  'Tropical_Kingbird': 77,  'Gray_Kingbird': 78,  'Belted_Kingfisher': 79,
                      'Green_Kingfisher': 80,  'Pied_Kingfisher': 81,  'Ringed_Kingfisher': 82,
                      'White_breasted_Kingfisher': 83,  'Red_legged_Kittiwake': 84,  'Horned_Lark': 85,
                      'Pacific_Loon': 86,  'Mallard': 87,  'Western_Meadowlark': 88,  'Hooded_Merganser': 89,
                      'Red_breasted_Merganser': 90,  'Mockingbird': 91,  'Nighthawk': 92,  'Clark_Nutcracker': 93,
                      'White_breasted_Nuthatch': 94,  'Baltimore_Oriole': 95,  'Hooded_Oriole': 96,
                      'Orchard_Oriole': 97,  'Scott_Oriole': 98,  'Ovenbird': 99,  'Brown_Pelican': 100,
                      'White_Pelican': 101, 'Western_Wood_Pewee': 102, 'Sayornis': 103, 'American_Pipit': 104,
                      'Whip_poor_Will': 105, 'Horned_Puffin': 106, 'Common_Raven': 107, 'White_necked_Raven': 108,
                      'American_Redstart': 109, 'Geococcyx': 110, 'Loggerhead_Shrike': 111, 'Great_Grey_Shrike': 112,
                      'Baird_Sparrow': 113, 'Black_throated_Sparrow': 114, 'Brewer_Sparrow': 115,
                      'Chipping_Sparrow': 116, 'Clay_colored_Sparrow': 117, 'House_Sparrow': 118,
                      'Field_Sparrow': 119, 'Fox_Sparrow': 120, 'Grasshopper_Sparrow': 121, 'Harris_Sparrow': 122,
                      'Henslow_Sparrow': 123, 'Le_Conte_Sparrow': 124, 'Lincoln_Sparrow': 125,
                      'Nelson_Sharp_tailed_Sparrow': 126, 'Savannah_Sparrow': 127, 'Seaside_Sparrow': 128,
                      'Song_Sparrow': 129, 'Tree_Sparrow': 130, 'Vesper_Sparrow': 131, 'White_crowned_Sparrow': 132,
                      'White_throated_Sparrow': 133, 'Cape_Glossy_Starling': 134, 'Bank_Swallow': 135,
                      'Barn_Swallow': 136, 'Cliff_Swallow': 137, 'Tree_Swallow': 138, 'Scarlet_Tanager': 139,
                      'Summer_Tanager': 140, 'Artic_Tern': 141, 'Black_Tern': 142, 'Caspian_Tern': 143,
                      'Common_Tern': 144, 'Elegant_Tern': 145, 'Forsters_Tern': 146, 'Least_Tern': 147,
                      'Green_tailed_Towhee': 148, 'Brown_Thrasher': 149, 'Sage_Thrasher': 150,
                      'Black_capped_Vireo': 151, 'Blue_headed_Vireo': 152, 'Philadelphia_Vireo': 153,
                      'Red_eyed_Vireo': 154, 'Warbling_Vireo': 155, 'White_eyed_Vireo': 156,
                      'Yellow_throated_Vireo': 157, 'Bay_breasted_Warbler': 158, 'Black_and_white_Warbler': 159,
                      'Black_throated_Blue_Warbler': 160, 'Blue_winged_Warbler': 161, 'Canada_Warbler': 162,
                      'Cape_May_Warbler': 163, 'Cerulean_Warbler': 164, 'Chestnut_sided_Warbler': 165,
                      'Golden_winged_Warbler': 166, 'Hooded_Warbler': 167, 'Kentucky_Warbler': 168,
                      'Magnolia_Warbler': 169, 'Mourning_Warbler': 170, 'Myrtle_Warbler': 171, 'Nashville_Warbler': 172,
                      'Orange_crowned_Warbler': 173, 'Palm_Warbler': 174, 'Pine_Warbler': 175, 'Prairie_Warbler': 176,
                      'Prothonotary_Warbler': 177, 'Swainson_Warbler': 178, 'Tennessee_Warbler': 179,
                      'Wilson_Warbler': 180, 'Worm_eating_Warbler': 181, 'Yellow_Warbler': 182,
                      'Northern_Waterthrush': 183, 'Louisiana_Waterthrush': 184, 'Bohemian_Waxwing': 185,
                      'Cedar_Waxwing': 186, 'American_Three_toed_Woodpecker': 187, 'Pileated_Woodpecker': 188,
                      'Red_bellied_Woodpecker': 189, 'Red_cockaded_Woodpecker': 190, 'Red_headed_Woodpecker': 191,
                      'Downy_Woodpecker': 192, 'Bewick_Wren': 193, 'Cactus_Wren': 194, 'Carolina_Wren': 195,
                      'House_Wren': 196, 'Marsh_Wren': 197, 'Rock_Wren': 198, 'Winter_Wren': 199,
                      'Common_Yellowthroat': 200}
CUB200_LABELS_DICT = {k: v - 1 for k, v in CUB200_LABELS_DICT.items()}

TINY_IMAGENET_200_LABELS_DICT = {'goldfish': 0, 'European_fire_salamander': 1, 'bullfrog': 2, 'tailed_frog': 3,
                                 'American_alligator': 4, 'boa_constrictor': 5, 'trilobite': 6, 'scorpion': 7,
                                 'black_widow': 8, 'tarantula': 9, 'centipede': 10, 'goose': 11, 'koala': 12,
                                 'jellyfish': 13, 'brain_coral': 14, 'snail': 15, 'slug': 16, 'sea_slug': 17,
                                 'American_lobster': 18, 'spiny_lobster': 19, 'black_stork': 20, 'king_penguin': 21,
                                 'albatross': 22, 'dugong': 23, 'Chihuahua': 24, 'Yorkshire_terrier': 25,
                                 'golden_retriever': 26, 'Labrador_retriever': 27, 'German_shepherd': 28,
                                 'standard_poodle': 29, 'tabby': 30, 'Persian_cat': 31, 'Egyptian_cat': 32,
                                 'cougar': 33, 'lion': 34, 'brown_bear': 35, 'ladybug': 36, 'fly': 37, 'bee': 38,
                                 'grasshopper': 39, 'walking_stick': 40, 'cockroach': 41, 'mantis': 42, 'dragonfly': 43,
                                 'monarch': 44, 'sulphur_butterfly': 45, 'sea_cucumber': 46, 'guinea_pig': 47,
                                 'hog': 48, 'ox': 49, 'bison': 50, 'bighorn': 51, 'gazelle': 52, 'Arabian_camel': 53,
                                 'orangutan': 54, 'chimpanzee': 55, 'baboon': 56, 'African_elephant': 57,
                                 'lesser_panda': 58, 'abacus': 59, 'academic_gown': 60, 'altar': 61, 'apron': 62,
                                 'backpack': 63, 'bannister': 64, 'barbershop': 65, 'barn': 66, 'barrel': 67,
                                 'basketball': 68, 'bathtub': 69, 'beach_wagon': 70, 'beacon': 71, 'beaker': 72,
                                 'beer_bottle': 73, 'bikini': 74, 'binoculars': 75, 'birdhouse': 76, 'bow_tie': 77,
                                 'brass': 78, 'broom': 79, 'bucket': 80, 'bullet_train': 81, 'butcher_shop': 82,
                                 'candle': 83, 'cannon': 84, 'cardigan': 85, 'cash_machine': 86, 'CD_player': 87,
                                 'chain': 88, 'chest': 89, 'Christmas_stocking': 90, 'cliff_dwelling': 91,
                                 'computer_keyboard': 92, 'confectionery': 93, 'convertible': 94, 'crane': 95,
                                 'dam': 96, 'desk': 97, 'dining_table': 98, 'drumstick': 99, 'dumbbell': 100,
                                 'flagpole': 101, 'fountain': 102, 'freight_car': 103, 'frying_pan': 104,
                                 'fur_coat': 105, 'gasmask': 106, 'go-kart': 107, 'gondola': 108, 'hourglass': 109,
                                 'iPod': 110, 'jinrikisha': 111, 'kimono': 112, 'lampshade': 113, 'lawn_mower': 114,
                                 'lifeboat': 115, 'limousine': 116, 'magnetic_compass': 117, 'maypole': 118,
                                 'military_uniform': 119, 'miniskirt': 120, 'moving_van': 121, 'nail': 122,
                                 'neck_brace': 123, 'obelisk': 124, 'oboe': 125, 'organ': 126, 'parking_meter': 127,
                                 'pay-phone': 128, 'picket_fence': 129, 'pill_bottle': 130, 'plunger': 131,
                                 'pole': 132, 'police_van': 133, 'poncho': 134, 'pop_bottle': 135,
                                 "potter's_wheel": 136, 'projectile': 137, 'punching_bag': 138, 'reel': 139,
                                 'refrigerator': 140, 'remote_control': 141, 'rocking_chair': 142, 'rugby_ball': 143,
                                 'sandal': 144, 'school_bus': 145, 'scoreboard': 146, 'sewing_machine': 147,
                                 'snorkel': 148, 'sock': 149, 'sombrero': 150, 'space_heater': 151,
                                 'spider_web': 152, 'sports_car': 153, 'steel_arch_bridge': 154, 'stopwatch': 155,
                                 'sunglasses': 156, 'suspension_bridge': 157, 'swimming_trunks': 158, 'syringe': 159,
                                 'teapot': 160, 'teddy': 161, 'thatch': 162, 'torch': 163, 'tractor': 164,
                                 'triumphal_arch': 165, 'trolleybus': 166, 'turnstile': 167, 'umbrella': 168,
                                 'vestment': 169, 'viaduct': 170, 'volleyball': 171, 'water_jug': 172,
                                 'water_tower': 173, 'wok': 174, 'wooden_spoon': 175, 'comic_book': 176,
                                 'plate': 177, 'guacamole': 178, 'ice_cream': 179, 'ice_lolly': 180, 'pretzel': 181,
                                 'mashed_potato': 182, 'cauliflower': 183, 'bell_pepper': 184, 'mushroom': 185,
                                 'orange': 186, 'lemon': 187, 'banana': 188, 'pomegranate': 189, 'meat_loaf': 190,
                                 'pizza': 191, 'potpie': 192, 'espresso': 193, 'alp': 194, 'cliff': 195,
                                 'coral_reef': 196, 'lakeside': 197, 'seashore': 198, 'acorn': 199
                                 }

IMAGE_NET_MAPPING = {
    'n02119789': "kit_fox",
    'n02100735': "English_setter",
    'n02110185': "Siberian_husky",
    'n02096294': "Australian_terrier",
    'n02102040': "English_springer",
    'n02066245': "grey_whale",
    'n02509815': "lesser_panda",
    'n02124075': "Egyptian_cat",
    'n02417914': "ibex",
    'n02123394': "Persian_cat",
    'n02125311': "cougar",
    'n02423022': "gazelle",
    'n02346627': "porcupine",
    'n02077923': "sea_lion",
    'n02110063': "malamute",
    'n02447366': "badger",
    'n02109047': "Great_Dane",
    'n02089867': "Walker_hound",
    'n02102177': "Welsh_springer_spaniel",
    'n02091134': "whippet",
    'n02092002': "Scottish_deerhound",
    'n02071294': "killer_whale",
    'n02442845': "mink",
    'n02504458': "African_elephant",
    'n02092339': "Weimaraner",
    'n02098105': "soft-coated_wheaten_terrier",
    'n02096437': "Dandie_Dinmont",
    'n02114712': "red_wolf",
    'n02105641': "Old_English_sheepdog",
    'n02128925': "jaguar",
    'n02091635': "otterhound",
    'n02088466': "bloodhound",
    'n02096051': "Airedale",
    'n02117135': "hyena",
    'n02138441': "meerkat",
    'n02097130': "giant_schnauzer",
    'n02493509': "titi",
    'n02457408': "three-toed_sloth",
    'n02389026': "sorrel",
    'n02443484': "black-footed_ferret",
    'n02110341': "dalmatian",
    'n02089078': "black-and-tan_coonhound",
    'n02086910': "papillon",
    'n02445715': "skunk",
    'n02093256': "Staffordshire_bullterrier",
    'n02113978': "Mexican_hairless",
    'n02106382': "Bouvier_des_Flandres",
    'n02441942': "weasel",
    'n02113712': "miniature_poodle",
    'n02113186': "Cardigan",
    'n02105162': "malinois",
    'n02415577': "bighorn",
    'n02356798': "fox_squirrel",
    'n02488702': "colobus",
    'n02123159': "tiger_cat",
    'n02098413': "Lhasa",
    'n02422699': "impala",
    'n02114855': "coyote",
    'n02094433': "Yorkshire_terrier",
    'n02111277': "Newfoundland",
    'n02132136': "brown_bear",
    'n02119022': "red_fox",
    'n02091467': "Norwegian_elkhound",
    'n02106550': "Rottweiler",
    'n02422106': "hartebeest",
    'n02091831': "Saluki",
    'n02120505': "grey_fox",
    'n02104365': "schipperke",
    'n02086079': "Pekinese",
    'n02112706': "Brabancon_griffon",
    'n02098286': "West_Highland_white_terrier",
    'n02095889': "Sealyham_terrier",
    'n02484975': "guenon",
    'n02137549': "mongoose",
    'n02500267': "indri",
    'n02129604': "tiger",
    'n02090721': "Irish_wolfhound",
    'n02396427': "wild_boar",
    'n02108000': "EntleBucher",
    'n02391049': "zebra",
    'n02412080': "ram",
    'n02108915': "French_bulldog",
    'n02480495': "orangutan",
    'n02110806': "basenji",
    'n02128385': "leopard",
    'n02107683': "Bernese_mountain_dog",
    'n02085936': "Maltese_dog",
    'n02094114': "Norfolk_terrier",
    'n02087046': "toy_terrier",
    'n02100583': "vizsla",
    'n02096177': "cairn",
    'n02494079': "squirrel_monkey",
    'n02105056': "groenendael",
    'n02101556': "clumber",
    'n02123597': "Siamese_cat",
    'n02481823': "chimpanzee",
    'n02105505': "komondor",
    'n02088094': "Afghan_hound",
    'n02085782': "Japanese_spaniel",
    'n02489166': "proboscis_monkey",
    'n02364673': "guinea_pig",
    'n02114548': "white_wolf",
    'n02134084': "ice_bear",
    'n02480855': "gorilla",
    'n02090622': "borzoi",
    'n02113624': "toy_poodle",
    'n02093859': "Kerry_blue_terrier",
    'n02403003': "ox",
    'n02097298': "Scotch_terrier",
    'n02108551': "Tibetan_mastiff",
    'n02493793': "spider_monkey",
    'n02107142': "Doberman",
    'n02096585': "Boston_bull",
    'n02107574': "Greater_Swiss_Mountain_dog",
    'n02107908': "Appenzeller",
    'n02086240': "Shih-Tzu",
    'n02102973': "Irish_water_spaniel",
    'n02112018': "Pomeranian",
    'n02093647': "Bedlington_terrier",
    'n02397096': "warthog",
    'n02437312': "Arabian_camel",
    'n02483708': "siamang",
    'n02097047': "miniature_schnauzer",
    'n02106030': "collie",
    'n02099601': "golden_retriever",
    'n02093991': "Irish_terrier",
    'n02110627': "affenpinscher",
    'n02106166': "Border_collie",
    'n02326432': "hare",
    'n02108089': "boxer",
    'n02097658': "silky_terrier",
    'n02088364': "beagle",
    'n02111129': "Leonberg",
    'n02100236': "German_short-haired_pointer",
    'n02486261': "patas",
    'n02115913': "dhole",
    'n02486410': "baboon",
    'n02487347': "macaque",
    'n02099849': "Chesapeake_Bay_retriever",
    'n02108422': "bull_mastiff",
    'n02104029': "kuvasz",
    'n02492035': "capuchin",
    'n02110958': "pug",
    'n02099429': "curly-coated_retriever",
    'n02094258': "Norwich_terrier",
    'n02099267': "flat-coated_retriever",
    'n02395406': "hog",
    'n02112350': "keeshond",
    'n02109961': "Eskimo_dog",
    'n02101388': "Brittany_spaniel",
    'n02113799': "standard_poodle",
    'n02095570': "Lakeland_terrier",
    'n02128757': "snow_leopard",
    'n02101006': "Gordon_setter",
    'n02115641': "dingo",
    'n02097209': "standard_schnauzer",
    'n02342885': "hamster",
    'n02097474': "Tibetan_terrier",
    'n02120079': "Arctic_fox",
    'n02095314': "wire-haired_fox_terrier",
    'n02088238': "basset",
    'n02408429': "water_buffalo",
    'n02133161': "American_black_bear",
    'n02328150': "Angora",
    'n02410509': "bison",
    'n02492660': "howler_monkey",
    'n02398521': "hippopotamus",
    'n02112137': "chow",
    'n02510455': "giant_panda",
    'n02093428': "American_Staffordshire_terrier",
    'n02105855': "Shetland_sheepdog",
    'n02111500': "Great_Pyrenees",
    'n02085620': "Chihuahua",
    'n02123045': "tabby",
    'n02490219': "marmoset",
    'n02099712': "Labrador_retriever",
    'n02109525': "Saint_Bernard",
    'n02454379': "armadillo",
    'n02111889': "Samoyed",
    'n02088632': "bluetick",
    'n02090379': "redbone",
    'n02443114': "polecat",
    'n02361337': "marmot",
    'n02105412': "kelpie",
    'n02483362': "gibbon",
    'n02437616': "llama",
    'n02107312': "miniature_pinscher",
    'n02325366': "wood_rabbit",
    'n02091032': "Italian_greyhound",
    'n02129165': "lion",
    'n02102318': "cocker_spaniel",
    'n02100877': "Irish_setter",
    'n02074367': "dugong",
    'n02504013': "Indian_elephant",
    'n02363005': "beaver",
    'n02102480': "Sussex_spaniel",
    'n02113023': "Pembroke",
    'n02086646': "Blenheim_spaniel",
    'n02497673': "Madagascar_cat",
    'n02087394': "Rhodesian_ridgeback",
    'n02127052': "lynx",
    'n02116738': "African_hunting_dog",
    'n02488291': "langur",
    'n02091244': "Ibizan_hound",
    'n02114367': "timber_wolf",
    'n02130308': "cheetah",
    'n02089973': "English_foxhound",
    'n02105251': "briard",
    'n02134418': "sloth_bear",
    'n02093754': "Border_terrier",
    'n02106662': "German_shepherd",
    'n02444819': "otter",
    'n01882714': "koala",
    'n01871265': "tusker",
    'n01872401': "echidna",
    'n01877812': "wallaby",
    'n01873310': "platypus",
    'n01883070': "wombat",
    'n04086273': "revolver",
    'n04507155': "umbrella",
    'n04147183': "schooner",
    'n04254680': "soccer_ball",
    'n02672831': "accordion",
    'n02219486': "ant",
    'n02317335': "starfish",
    'n01968897': "chambered_nautilus",
    'n03452741': "grand_piano",
    'n03642806': "laptop",
    'n07745940': "strawberry",
    'n02690373': "airliner",
    'n04552348': "warplane",
    'n02692877': "airship",
    'n02782093': "balloon",
    'n04266014': "space_shuttle",
    'n03344393': "fireboat",
    'n03447447': "gondola",
    'n04273569': "speedboat",
    'n03662601': "lifeboat",
    'n02951358': "canoe",
    'n04612504': "yawl",
    'n02981792': "catamaran",
    'n04483307': "trimaran",
    'n03095699': "container_ship",
    'n03673027': "liner",
    'n03947888': "pirate",
    'n02687172': "aircraft_carrier",
    'n04347754': "submarine",
    'n04606251': "wreck",
    'n03478589': "half_track",
    'n04389033': "tank",
    'n03773504': "missile",
    'n02860847': "bobsled",
    'n03218198': "dogsled",
    'n02835271': "bicycle-built-for-two",
    'n03792782': "mountain_bike",
    'n03393912': "freight_car",
    'n03895866': "passenger_car",
    'n02797295': "barrow",
    'n04204347': "shopping_cart",
    'n03791053': "motor_scooter",
    'n03384352': "forklift",
    'n03272562': "electric_locomotive",
    'n04310018': "steam_locomotive",
    'n02704792': "amphibian",
    'n02701002': "ambulance",
    'n02814533': "beach_wagon",
    'n02930766': "cab",
    'n03100240': "convertible",
    'n03594945': "jeep",
    'n03670208': "limousine",
    'n03770679': "minivan",
    'n03777568': "Model_T",
    'n04037443': "racer",
    'n04285008': "sports_car",
    'n03444034': "go-kart",
    'n03445924': "golfcart",
    'n03785016': "moped",
    'n04252225': "snowplow",
    'n03345487': "fire_engine",
    'n03417042': "garbage_truck",
    'n03930630': "pickup",
    'n04461696': "tow_truck",
    'n04467665': "trailer_truck",
    'n03796401': "moving_van",
    'n03977966': "police_van",
    'n04065272': "recreational_vehicle",
    'n04335435': "streetcar",
    'n04252077': "snowmobile",
    'n04465501': "tractor",
    'n03776460': "mobile_home",
    'n04482393': "tricycle",
    'n04509417': "unicycle",
    'n03538406': "horse_cart",
    'n03599486': "jinrikisha",
    'n03868242': "oxcart",
    'n02804414': "bassinet",
    'n03125729': "cradle",
    'n03131574': "crib",
    'n03388549': "four-poster",
    'n02870880': "bookcase",
    'n03018349': "china_cabinet",
    'n03742115': "medicine_chest",
    'n03016953': "chiffonier",
    'n04380533': "table_lamp",
    'n03337140': "file",
    'n03891251': "park_bench",
    'n02791124': "barber_chair",
    'n04429376': "throne",
    'n03376595': "folding_chair",
    'n04099969': "rocking_chair",
    'n04344873': "studio_couch",
    'n04447861': "toilet_seat",
    'n03179701': "desk",
    'n03982430': "pool_table",
    'n03201208': "dining_table",
    'n03290653': "entertainment_center",
    'n04550184': "wardrobe",
    'n07742313': "Granny_Smith",
    'n07747607': "orange",
    'n07749582': "lemon",
    'n07753113': "fig",
    'n07753275': "pineapple",
    'n07753592': "banana",
    'n07754684': "jackfruit",
    'n07760859': "custard_apple",
    'n07768694': "pomegranate",
    'n12267677': "acorn",
    'n12620546': "hip",
    'n13133613': "ear",
    'n11879895': "rapeseed",
    'n12144580': "corn",
    'n12768682': "buckeye",
    'n03854065': "organ",
    'n04515003': "upright",
    'n03017168': "chime",
    'n03249569': "drum",
    'n03447721': "gong",
    'n03720891': "maraca",
    'n03721384': "marimba",
    'n04311174': "steel_drum",
    'n02787622': "banjo",
    'n02992211': "cello",
    'n04536866': "violin",
    'n03495258': "harp",
    'n02676566': "acoustic_guitar",
    'n03272010': "electric_guitar",
    'n03110669': "cornet",
    'n03394916': "French_horn",
    'n04487394': "trombone",
    'n03494278': "harmonica",
    'n03840681': "ocarina",
    'n03884397': "panpipe",
    'n02804610': "bassoon",
    'n03838899': "oboe",
    'n04141076': "sax",
    'n03372029': "flute",
    'n11939491': "daisy",
    'n12057211': "yellow_lady's_slipper",
    'n09246464': "cliff",
    'n09468604': "valley",
    'n09193705': "alp",
    'n09472597': "volcano",
    'n09399592': "promontory",
    'n09421951': "sandbar",
    'n09256479': "coral_reef",
    'n09332890': "lakeside",
    'n09428293': "seashore",
    'n09288635': "geyser",
    'n03498962': "hatchet",
    'n03041632': "cleaver",
    'n03658185': "letter_opener",
    'n03954731': "plane",
    'n03995372': "power_drill",
    'n03649909': "lawn_mower",
    'n03481172': "hammer",
    'n03109150': "corkscrew",
    'n02951585': "can_opener",
    'n03970156': "plunger",
    'n04154565': "screwdriver",
    'n04208210': "shovel",
    'n03967562': "plow",
    'n03000684': "chain_saw",
    'n01514668': "cock",
    'n01514859': "hen",
    'n01518878': "ostrich",
    'n01530575': "brambling",
    'n01531178': "goldfinch",
    'n01532829': "house_finch",
    'n01534433': "junco",
    'n01537544': "indigo_bunting",
    'n01558993': "robin",
    'n01560419': "bulbul",
    'n01580077': "jay",
    'n01582220': "magpie",
    'n01592084': "chickadee",
    'n01601694': "water_ouzel",
    'n01608432': "kite",
    'n01614925': "bald_eagle",
    'n01616318': "vulture",
    'n01622779': "great_grey_owl",
    'n01795545': "black_grouse",
    'n01796340': "ptarmigan",
    'n01797886': "ruffed_grouse",
    'n01798484': "prairie_chicken",
    'n01806143': "peacock",
    'n01806567': "quail",
    'n01807496': "partridge",
    'n01817953': "African_grey",
    'n01818515': "macaw",
    'n01819313': "sulphur-crested_cockatoo",
    'n01820546': "lorikeet",
    'n01824575': "coucal",
    'n01828970': "bee_eater",
    'n01829413': "hornbill",
    'n01833805': "hummingbird",
    'n01843065': "jacamar",
    'n01843383': "toucan",
    'n01847000': "drake",
    'n01855032': "red-breasted_merganser",
    'n01855672': "goose",
    'n01860187': "black_swan",
    'n02002556': "white_stork",
    'n02002724': "black_stork",
    'n02006656': "spoonbill",
    'n02007558': "flamingo",
    'n02009912': "American_egret",
    'n02009229': "little_blue_heron",
    'n02011460': "bittern",
    'n02012849': "crane",
    'n02013706': "limpkin",
    'n02018207': "American_coot",
    'n02018795': "bustard",
    'n02025239': "ruddy_turnstone",
    'n02027492': "red-backed_sandpiper",
    'n02028035': "redshank",
    'n02033041': "dowitcher",
    'n02037110': "oystercatcher",
    'n02017213': "European_gallinule",
    'n02051845': "pelican",
    'n02056570': "king_penguin",
    'n02058221': "albatross",
    'n01484850': "great_white_shark",
    'n01491361': "tiger_shark",
    'n01494475': "hammerhead",
    'n01496331': "electric_ray",
    'n01498041': "stingray",
    'n02514041': "barracouta",
    'n02536864': "coho",
    'n01440764': "tench",
    'n01443537': "goldfish",
    'n02526121': "eel",
    'n02606052': "rock_beauty",
    'n02607072': "anemone_fish",
    'n02643566': "lionfish",
    'n02655020': "puffer",
    'n02640242': "sturgeon",
    'n02641379': "gar",
    'n01664065': "loggerhead",
    'n01665541': "leatherback_turtle",
    'n01667114': "mud_turtle",
    'n01667778': "terrapin",
    'n01669191': "box_turtle",
    'n01675722': "banded_gecko",
    'n01677366': "common_iguana",
    'n01682714': "American_chameleon",
    'n01685808': "whiptail",
    'n01687978': "agama",
    'n01688243': "frilled_lizard",
    'n01689811': "alligator_lizard",
    'n01692333': "Gila_monster",
    'n01693334': "green_lizard",
    'n01694178': "African_chameleon",
    'n01695060': "Komodo_dragon",
    'n01704323': "triceratops",
    'n01697457': "African_crocodile",
    'n01698640': "American_alligator",
    'n01728572': "thunder_snake",
    'n01728920': "ringneck_snake",
    'n01729322': "hognose_snake",
    'n01729977': "green_snake",
    'n01734418': "king_snake",
    'n01735189': "garter_snake",
    'n01737021': "water_snake",
    'n01739381': "vine_snake",
    'n01740131': "night_snake",
    'n01742172': "boa_constrictor",
    'n01744401': "rock_python",
    'n01748264': "Indian_cobra",
    'n01749939': "green_mamba",
    'n01751748': "sea_snake",
    'n01753488': "horned_viper",
    'n01755581': "diamondback",
    'n01756291': "sidewinder",
    'n01629819': "European_fire_salamander",
    'n01630670': "common_newt",
    'n01631663': "eft",
    'n01632458': "spotted_salamander",
    'n01632777': "axolotl",
    'n01641577': "bullfrog",
    'n01644373': "tree_frog",
    'n01644900': "tailed_frog",
    'n04579432': "whistle",
    'n04592741': "wing",
    'n03876231': "paintbrush",
    'n03483316': "hand_blower",
    'n03868863': "oxygen_mask",
    'n04251144': "snorkel",
    'n03691459': "loudspeaker",
    'n03759954': "microphone",
    'n04152593': "screen",
    'n03793489': "mouse",
    'n03271574': "electric_fan",
    'n03843555': "oil_filter",
    'n04332243': "strainer",
    'n04265275': "space_heater",
    'n04330267': "stove",
    'n03467068': "guillotine",
    'n02794156': "barometer",
    'n04118776': "rule",
    'n03841143': "odometer",
    'n04141975': "scale",
    'n02708093': "analog_clock",
    'n03196217': "digital_clock",
    'n04548280': "wall_clock",
    'n03544143': "hourglass",
    'n04355338': "sundial",
    'n03891332': "parking_meter",
    'n04328186': "stopwatch",
    'n03197337': "digital_watch",
    'n04317175': "stethoscope",
    'n04376876': "syringe",
    'n03706229': "magnetic_compass",
    'n02841315': "binoculars",
    'n04009552': "projector",
    'n04356056': "sunglasses",
    'n03692522': "loupe",
    'n04044716': "radio_telescope",
    'n02879718': "bow",
    'n02950826': "cannon",
    'n02749479': "assault_rifle",
    'n04090263': "rifle",
    'n04008634': "projectile",
    'n03085013': "computer_keyboard",
    'n04505470': "typewriter_keyboard",
    'n03126707': "crane",
    'n03666591': "lighter",
    'n02666196': "abacus",
    'n02977058': "cash_machine",
    'n04238763': "slide_rule",
    'n03180011': "desktop_computer",
    'n03485407': "hand-held_computer",
    'n03832673': "notebook",
    'n06359193': "web_site",
    'n03496892': "harvester",
    'n04428191': "thresher",
    'n04004767': "printer",
    'n04243546': "slot",
    'n04525305': "vending_machine",
    'n04179913': "sewing_machine",
    'n03602883': "joystick",
    'n04372370': "switch",
    'n03532672': "hook",
    'n02974003': "car_wheel",
    'n03874293': "paddlewheel",
    'n03944341': "pinwheel",
    'n03992509': "potter's_wheel",
    'n03425413': "gas_pump",
    'n02966193': "carousel",
    'n04371774': "swing",
    'n04067472': "reel",
    'n04040759': "radiator",
    'n04019541': "puck",
    'n03492542': "hard_disc",
    'n04355933': "sunglass",
    'n03929660': "pick",
    'n02965783': "car_mirror",
    'n04258138': "solar_dish",
    'n04074963': "remote_control",
    'n03208938': "disk_brake",
    'n02910353': "buckle",
    'n03476684': "hair_slide",
    'n03627232': "knot",
    'n03075370': "combination_lock",
    'n03874599': "padlock",
    'n03804744': "nail",
    'n04127249': "safety_pin",
    'n04153751': "screw",
    'n03803284': "muzzle",
    'n04162706': "seat_belt",
    'n04228054': "ski",
    'n02948072': "candle",
    'n03590841': "jack-o'-lantern",
    'n04286575': "spotlight",
    'n04456115': "torch",
    'n03814639': "neck_brace",
    'n03933933': "pier",
    'n04485082': "tripod",
    'n03733131': "maypole",
    'n03794056': "mousetrap",
    'n04275548': "spider_web",
    'n01768244': "trilobite",
    'n01770081': "harvestman",
    'n01770393': "scorpion",
    'n01773157': "black_and_gold_garden_spider",
    'n01773549': "barn_spider",
    'n01773797': "garden_spider",
    'n01774384': "black_widow",
    'n01774750': "tarantula",
    'n01775062': "wolf_spider",
    'n01776313': "tick",
    'n01784675': "centipede",
    'n01990800': "isopod",
    'n01978287': "Dungeness_crab",
    'n01978455': "rock_crab",
    'n01980166': "fiddler_crab",
    'n01981276': "king_crab",
    'n01983481': "American_lobster",
    'n01984695': "spiny_lobster",
    'n01985128': "crayfish",
    'n01986214': "hermit_crab",
    'n02165105': "tiger_beetle",
    'n02165456': "ladybug",
    'n02167151': "ground_beetle",
    'n02168699': "long-horned_beetle",
    'n02169497': "leaf_beetle",
    'n02172182': "dung_beetle",
    'n02174001': "rhinoceros_beetle",
    'n02177972': "weevil",
    'n02190166': "fly",
    'n02206856': "bee",
    'n02226429': "grasshopper",
    'n02229544': "cricket",
    'n02231487': "walking_stick",
    'n02233338': "cockroach",
    'n02236044': "mantis",
    'n02256656': "cicada",
    'n02259212': "leafhopper",
    'n02264363': "lacewing",
    'n02268443': "dragonfly",
    'n02268853': "damselfly",
    'n02276258': "admiral",
    'n02277742': "ringlet",
    'n02279972': "monarch",
    'n02280649': "cabbage_butterfly",
    'n02281406': "sulphur_butterfly",
    'n02281787': "lycaenid",
    'n01910747': "jellyfish",
    'n01914609': "sea_anemone",
    'n01917289': "brain_coral",
    'n01924916': "flatworm",
    'n01930112': "nematode",
    'n01943899': "conch",
    'n01944390': "snail",
    'n01945685': "slug",
    'n01950731': "sea_slug",
    'n01955084': "chiton",
    'n02319095': "sea_urchin",
    'n02321529': "sea_cucumber",
    'n03584829': "iron",
    'n03297495': "espresso_maker",
    'n03761084': "microwave",
    'n03259280': "Dutch_oven",
    'n04111531': "rotisserie",
    'n04442312': "toaster",
    'n04542943': "waffle_iron",
    'n04517823': "vacuum",
    'n03207941': "dishwasher",
    'n04070727': "refrigerator",
    'n04554684': "washer",
    'n03133878': "Crock_Pot",
    'n03400231': "frying_pan",
    'n04596742': "wok",
    'n02939185': "caldron",
    'n03063689': "coffeepot",
    'n04398044': "teapot",
    'n04270147': "spatula",
    'n02699494': "altar",
    'n04486054': "triumphal_arch",
    'n03899768': "patio",
    'n04311004': "steel_arch_bridge",
    'n04366367': "suspension_bridge",
    'n04532670': "viaduct",
    'n02793495': "barn",
    'n03457902': "greenhouse",
    'n03877845': "palace",
    'n03781244': "monastery",
    'n03661043': "library",
    'n02727426': "apiary",
    'n02859443': "boathouse",
    'n03028079': "church",
    'n03788195': "mosque",
    'n04346328': "stupa",
    'n03956157': "planetarium",
    'n04081281': "restaurant",
    'n03032252': "cinema",
    'n03529860': "home_theater",
    'n03697007': "lumbermill",
    'n03065424': "coil",
    'n03837869': "obelisk",
    'n04458633': "totem_pole",
    'n02980441': "castle",
    'n04005630': "prison",
    'n03461385': "grocery_store",
    'n02776631': "bakery",
    'n02791270': "barbershop",
    'n02871525': "bookshop",
    'n02927161': "butcher_shop",
    'n03089624': "confectionery",
    'n04200800': "shoe_shop",
    'n04443257': "tobacco_shop",
    'n04462240': "toyshop",
    'n03388043': "fountain",
    'n03042490': "cliff_dwelling",
    'n04613696': "yurt",
    'n03216828': "dock",
    'n02892201': "brass",
    'n03743016': "megalith",
    'n02788148': "bannister",
    'n02894605': "breakwater",
    'n03160309': "dam",
    'n03000134': "chainlink_fence",
    'n03930313': "picket_fence",
    'n04604644': "worm_fence",
    'n04326547': "stone_wall",
    'n03459775': "grille",
    'n04239074': "sliding_door",
    'n04501370': "turnstile",
    'n03792972': "mountain_tent",
    'n04149813': "scoreboard",
    'n03530642': "honeycomb",
    'n03961711': "plate_rack",
    'n03903868': "pedestal",
    'n02814860': "beacon",
    'n07711569': "mashed_potato",
    'n07720875': "bell_pepper",
    'n07714571': "head_cabbage",
    'n07714990': "broccoli",
    'n07715103': "cauliflower",
    'n07716358': "zucchini",
    'n07716906': "spaghetti_squash",
    'n07717410': "acorn_squash",
    'n07717556': "butternut_squash",
    'n07718472': "cucumber",
    'n07718747': "artichoke",
    'n07730033': "cardoon",
    'n07734744': "mushroom",
    'n04209239': "shower_curtain",
    'n03594734': "jean",
    'n02971356': "carton",
    'n03485794': "handkerchief",
    'n04133789': "sandal",
    'n02747177': "ashcan",
    'n04125021': "safe",
    'n07579787': "plate",
    'n03814906': "necklace",
    'n03134739': "croquet_ball",
    'n03404251': "fur_coat",
    'n04423845': "thimble",
    'n03877472': "pajama",
    'n04120489': "running_shoe",
    'n03062245': "cocktail_shaker",
    'n03014705': "chest",
    'n03717622': "manhole_cover",
    'n03777754': "modem",
    'n04493381': "tub",
    'n04476259': "tray",
    'n02777292': "balance_beam",
    'n07693725': "bagel",
    'n03998194': "prayer_rug",
    'n03617480': "kimono",
    'n07590611': "hot_pot",
    'n04579145': "whiskey_jug",
    'n03623198': "knee_pad",
    'n07248320': "book_jacket",
    'n04277352': "spindle",
    'n04229816': "ski_mask",
    'n02823428': "beer_bottle",
    'n03127747': "crash_helmet",
    'n02877765': "bottlecap",
    'n04435653': "tile_roof",
    'n03724870': "mask",
    'n03710637': "maillot",
    'n03920288': "Petri_dish",
    'n03379051': "football_helmet",
    'n02807133': "bathing_cap",
    'n04399382': "teddy",
    'n03527444': "holster",
    'n03983396': "pop_bottle",
    'n03924679': "photocopier",
    'n04532106': "vestment",
    'n06785654': "crossword_puzzle",
    'n03445777': "golf_ball",
    'n07613480': "trifle",
    'n04350905': "suit",
    'n04562935': "water_tower",
    'n03325584': "feather_boa",
    'n03045698': "cloak",
    'n07892512': "red_wine",
    'n03250847': "drumstick",
    'n04192698': "shield",
    'n03026506': "Christmas_stocking",
    'n03534580': "hoopskirt",
    'n07565083': "menu",
    'n04296562': "stage",
    'n02869837': "bonnet",
    'n07871810': "meat_loaf",
    'n02799071': "baseball",
    'n03314780': "face_powder",
    'n04141327': "scabbard",
    'n04357314': "sunscreen",
    'n02823750': "beer_glass",
    'n13052670': "hen-of-the-woods",
    'n07583066': "guacamole",
    'n03637318': "lampshade",
    'n04599235': "wool",
    'n07802026': "hay",
    'n02883205': "bow_tie",
    'n03709823': "mailbag",
    'n04560804': "water_jug",
    'n02909870': "bucket",
    'n03207743': "dishrag",
    'n04263257': "soup_bowl",
    'n07932039': "eggnog",
    'n03786901': "mortar",
    'n04479046': "trench_coat",
    'n03873416': "paddle",
    'n02999410': "chain",
    'n04367480': "swab",
    'n03775546': "mixing_bowl",
    'n07875152': "potpie",
    'n04591713': "wine_bottle",
    'n04201297': "shoji",
    'n02916936': "bulletproof_vest",
    'n03240683': "drilling_platform",
    'n02840245': "binder",
    'n02963159': "cardigan",
    'n04370456': "sweatshirt",
    'n03991062': "pot",
    'n02843684': "birdhouse",
    'n03482405': "hamper",
    'n03942813': "ping-pong_ball",
    'n03908618': "pencil_box",
    'n03902125': "pay-phone",
    'n07584110': "consomme",
    'n02730930': "apron",
    'n04023962': "punching_bag",
    'n02769748': "backpack",
    'n10148035': "groom",
    'n02817516': "bearskin",
    'n03908714': "pencil_sharpener",
    'n02906734': "broom",
    'n03788365': "mosquito_net",
    'n02667093': "abaya",
    'n03787032': "mortarboard",
    'n03980874': "poncho",
    'n03141823': "crutch",
    'n03976467': "Polaroid_camera",
    'n04264628': "space_bar",
    'n07930864': "cup",
    'n04039381': "racket",
    'n06874185': "traffic_light",
    'n04033901': "quill",
    'n04041544': "radio",
    'n07860988': "dough",
    'n03146219': "cuirass",
    'n03763968': "military_uniform",
    'n03676483': "lipstick",
    'n04209133': "shower_cap",
    'n03782006': "monitor",
    'n03857828': "oscilloscope",
    'n03775071': "mitten",
    'n02892767': "brassiere",
    'n07684084': "French_loaf",
    'n04522168': "vase",
    'n03764736': "milk_can",
    'n04118538': "rugby_ball",
    'n03887697': "paper_towel",
    'n13044778': "earthstar",
    'n03291819': "envelope",
    'n03770439': "miniskirt",
    'n03124170': "cowboy_hat",
    'n04487081': "trolleybus",
    'n03916031': "perfume",
    'n02808440': "bathtub",
    'n07697537': "hotdog",
    'n12985857': "coral_fungus",
    'n02917067': "bullet_train",
    'n03938244': "pillow",
    'n15075141': "toilet_tissue",
    'n02978881': "cassette",
    'n02966687': "carpenter's_kit",
    'n03633091': "ladle",
    'n13040303': "stinkhorn",
    'n03690938': "lotion",
    'n03476991': "hair_spray",
    'n02669723': "academic_gown",
    'n03220513': "dome",
    'n03127925': "crate",
    'n04584207': "wig",
    'n07880968': "burrito",
    'n03937543': "pill_bottle",
    'n03000247': "chain_mail",
    'n04418357': "theater_curtain",
    'n04590129': "window_shade",
    'n02795169': "barrel",
    'n04553703': "washbasin",
    'n02783161': "ballpoint",
    'n02802426': "basketball",
    'n02808304': "bath_towel",
    'n03124043': "cowboy_boot",
    'n03450230': "gown",
    'n04589890': "window_screen",
    'n12998815': "agaric",
    'n02992529': "cellular_telephone",
    'n03825788': "nipple",
    'n02790996': "barbell",
    'n03710193': "mailbox",
    'n03630383': "lab_coat",
    'n03347037': "fire_screen",
    'n03769881': "minibus",
    'n03871628': "packet",
    'n03733281': "maze",
    'n03976657': "pole",
    'n03535780': "horizontal_bar",
    'n04259630': "sombrero",
    'n03929855': "pickelhaube",
    'n04049303': "rain_barrel",
    'n04548362': "wallet",
    'n02979186': "cassette_player",
    'n06596364': "comic_book",
    'n03935335': "piggy_bank",
    'n06794110': "street_sign",
    'n02825657': "bell_cote",
    'n03388183': "fountain_pen",
    'n04591157': "Windsor_tie",
    'n04540053': "volleyball",
    'n03866082': "overskirt",
    'n04136333': "sarong",
    'n04026417': "purse",
    'n02865351': "bolo_tie",
    'n02834397': "bib",
    'n03888257': "parachute",
    'n04235860': "sleeping_bag",
    'n04404412': "television",
    'n04371430': "swimming_trunks",
    'n03733805': "measuring_cup",
    'n07920052': "espresso",
    'n07873807': "pizza",
    'n02895154': "breastplate",
    'n04204238': "shopping_basket",
    'n04597913': "wooden_spoon",
    'n04131690': "saltshaker",
    'n07836838': "chocolate_sauce",
    'n09835506': "ballplayer",
    'n03443371': "goblet",
    'n13037406': "gyromitra",
    'n04336792': "stretcher",
    'n04557648': "water_bottle",
    'n03187595': "dial_telephone",
    'n04254120': "soap_dispenser",
    'n03595614': "jersey",
    'n04146614': "school_bus",
    'n03598930': "jigsaw_puzzle",
    'n03958227': "plastic_bag",
    'n04069434': "reflex_camera",
    'n03188531': "diaper",
    'n02786058': "Band_Aid",
    'n07615774': "ice_lolly",
    'n04525038': "velvet",
    'n04409515': "tennis_ball",
    'n03424325': "gasmask",
    'n03223299': "doormat",
    'n03680355': "Loafer",
    'n07614500': "ice_cream",
    'n07695742': "pretzel",
    'n04033995': "quilt",
    'n03710721': "maillot",
    'n04392985': "tape_player",
    'n03047690': "clog",
    'n03584254': "iPod",
    'n13054560': "bolete",
    'n10565667': "scuba_diver",
    'n03950228': "pitcher",
    'n03729826': "matchstick",
    'n02837789': "bikini",
    'n04254777': "sock",
    'n02988304': "CD_player",
    'n03657121': "lens_cap",
    'n04417672': "thatch",
    'n04523525': "vault",
    'n02815834': "beaker",
    'n09229709': "bubble",
    'n07697313': "cheeseburger",
    'n03888605': "parallel_bars",
    'n03355925': "flagpole",
    'n03063599': "coffee_mug",
    'n04116512': "rubber_eraser",
    'n04325704': "stole",
    'n07831146': "carbonara",
    'n03255030': "dumbbell"
}
