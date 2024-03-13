""" This file contains the implementation of the PoDD model """

import torch
import higher
import numpy as np
import torch.nn as nn
import torch.optim as optim

from src.data_utils import get_arch


class PoDD(nn.Module):
    def __init__(self, distilled_data, y_init, cropping_function, arch, window, lr, num_train_eval, total_patch_num,
                 distill_batch_size, num_classes=2, train_y=False, train_lr=False, channel=3, im_size=(32, 32),
                 inner_optim='SGD', cctype=0, syn_intervention=None, real_intervention=None, decay=False,
                 label_cropping_function=None):
        super(PoDD, self).__init__()

        self.data = distilled_data
        self.samples_num = total_patch_num
        self.get_crops = cropping_function
        self.get_labels = label_cropping_function
        self.distill_batch_size = distill_batch_size

        self.label = y_init
        self.train_y = train_y
        if train_y:
            self.label = self.label.float().cuda().requires_grad_(True)

        self.arch = arch
        self.decay = decay
        self.cctype = cctype
        self.window = window
        self.im_size = im_size
        self.channel = channel
        self.curriculum = window
        self.num_classes = num_classes
        self.inner_optim = inner_optim
        self.num_train_eval = num_train_eval
        self.syn_intervention = syn_intervention
        self.real_intervention = real_intervention

        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr if not train_lr else torch.nn.Parameter(lr)
        self.net = get_arch(arch, self.num_classes, self.channel, self.im_size)

    def get_overlapping_patches_and_labels(self):
        perm = torch.randperm(self.samples_num, device='cpu')
        indices = perm[:self.distill_batch_size].sort()[0]
        imgs = self.get_crops(self.data, indices)
        if self.train_y:
            labels = self.get_labels(self.label, indices)
        else:
            labels = self.label[indices]
        return imgs, labels

    def forward(self, x):
        self.net = get_arch(self.arch, self.num_classes, self.channel, self.im_size).cuda()
        self.net.train()

        if self.inner_optim == 'SGD':
            self.optimizer = optim.SGD(self.net.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)
            self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[200],
                                                            gamma=0.2) if self.decay else None
        elif self.inner_optim == 'Adam':
            self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)

        else:
            raise ValueError(f'inner_optim={self.inner_optim} is not supported')

        if self.dd_type not in ['curriculum', 'standard']:
            print('The dataset distillation method is not implemented!')
            raise NotImplementedError()

        if self.dd_type == 'curriculum':
            for i in range(self.curriculum):
                self.optimizer.zero_grad()
                imgs, label = self.get_overlapping_patches_and_labels()
                imgs = self.syn_intervention(imgs, dtype='syn')

                out, pres = self.net(imgs)

                loss = self.criterion(out, label)
                loss.backward()
                self.optimizer.step()
                if self.inner_optim == 'SGD' and self.scheduler is not None:
                    self.scheduler.step()

        loss_coef = 1
        with higher.innerloop_ctx(self.net, self.optimizer, copy_initial_weights=True) as (fnet, diffopt):
            for i in range(self.window):
                imgs, label = self.get_overlapping_patches_and_labels()
                imgs = self.syn_intervention(imgs, dtype='syn')

                if i + self.curriculum == 150 or i + self.curriculum == 240:
                    if self.inner_optim == 'SGD':
                        loss_coef = loss_coef * 0.2

                out, pres = fnet(imgs)
                loss = self.criterion(out, label)

                diffopt.step(loss)
                if self.inner_optim == 'SGD' and self.scheduler is not None:
                    self.scheduler.step()

            x = self.real_intervention(x, dtype='real')
            return fnet(x)

    def init_train(self, epoch, init=False):
        if init:
            self.net = get_arch(self.arch, self.num_classes, self.channel, self.im_size).cuda()
            if self.inner_optim == 'SGD':
                self.optimizer = optim.SGD(self.net.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)
                self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[600],
                                                                gamma=0.2) if self.decay else None
            elif self.inner_optim == 'Adam':
                self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)

        for i in range(epoch):
            self.optimizer.zero_grad()
            imgs, label = self.get_overlapping_patches_and_labels()
            imgs = self.syn_intervention(imgs, dtype='syn')
            out, pres = self.net(imgs)
            loss = self.criterion(out, label)
            loss.backward()
            self.optimizer.step()
            if self.inner_optim == 'SGD' and self.scheduler is not None:
                self.scheduler.step()

    # initialize the EMA with the currect data value
    def ema_init(self, ema_coef):
        self.shadow = -1e5
        self.ema_coef = ema_coef

    # update the EMA value
    def ema_update(self, grad_norm):
        if self.shadow == -1e5:
            self.shadow = grad_norm
        else:
            self.shadow -= (1 - self.ema_coef) * (self.shadow - grad_norm)
        return self.shadow

    def test(self, x):
        with torch.no_grad():
            out = self.net(x)
        return out


def random_indices(y, nclass=10, intraclass=False, device='cuda'):
    n = len(y)
    if intraclass:
        index = torch.arange(n).to(device)
        for c in range(nclass):
            index_c = index[y == c]
            if len(index_c) > 0:
                randidx = torch.randperm(len(index_c))
                index[y == c] = index_c[randidx]
    else:
        index = torch.randperm(n)
    return index


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2
