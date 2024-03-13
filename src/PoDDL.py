""" PoDDL class for PoDD labeling strategy PoDDL """

import torch
import numpy as np

from src.PoDD_utils import _get_patch_index_lst, get_crops_from_poster


class PoDDL:
    """ Class for PoDD labeling strategy PoDDL """
    @staticmethod
    def get_labels_from_array(label_array: torch.Tensor,
                              shrink_factor_x: float, shrink_factor_y: float,
                              patch_size_x: int, patch_size_y: int,
                              patch_num_x: int, patch_num_y: int, indexes_subset=None):
        """ Returns a tensor of learnable labels from array """
        label_patch_size_x = max(1, np.floor(patch_size_x * shrink_factor_x).astype(int))
        label_patch_size_y = max(1, np.floor(patch_size_y * shrink_factor_y).astype(int))

        labels = get_crops_from_poster(label_array,
                                       label_patch_size_x, label_patch_size_y,
                                       patch_num_x, patch_num_y, indexes_subset=indexes_subset)
        labels = labels.sum(dim=(2, 3))
        labels /= labels.sum(1).unsqueeze(1)
        return labels

    @staticmethod
    def get_poster_labels(class_order: np.array,
                          patch_size_x: int, patch_size_y: int,
                          mosaic_size_x: int, mosaic_size_y: int,
                          poster_size_x: int, poster_size_y: int,
                          class_num_x: int, class_num_y: int,
                          patch_num_x: int, patch_num_y: int, use_softmax: bool = True):
        """ Returns a tensor of labels from a poster."""
        class_num = np.prod(class_order.shape)
        assert class_num == class_num_x * class_num_y, 'class_num_x * class_num_y must be equal to class_num'

        poster_labels_map = torch.zeros((class_num, poster_size_y, poster_size_x))
        with (torch.no_grad()):
            class_index_list = _get_patch_index_lst(mosaic_size_x, mosaic_size_y,
                                                    poster_size_x, poster_size_y,
                                                    class_num_x, class_num_y)

            for i, ((px1, px2), (py1, py2)) in zip(class_order.T.flatten(), class_index_list):
                poster_labels_map[i, py1:py2, px1:px2] += 1

            patch_indexes = _get_patch_index_lst(patch_size_x, patch_size_y,
                                                 poster_size_x, poster_size_y,
                                                 patch_num_x, patch_num_y)

            if use_softmax:
                labels = torch.stack([torch.nn.functional.softmax(
                    poster_labels_map[:, y1:y2, x1:x2].sum(dim=(1, 2)), dim=0) for ((x1, x2), (y1, y2)) in patch_indexes])
            else:
                labels = [poster_labels_map[:, y1:y2, x1:x2].sum(dim=(1, 2)) for ((x1, x2), (y1, y2)) in patch_indexes]
                labels = torch.stack([label / label.sum() for label in labels])

            return labels

    @staticmethod
    def init_label_array(poster_shape: np.array, class_order: np.array, comp_ipc: int):
        poster_shape = poster_shape[2:]
        class_num_y, class_num_x = class_order.shape
        class_num = class_num_x * class_num_y
        poster_2D_size = np.prod(poster_shape)
        max_labels_num = class_num * comp_ipc

        shrink_factor = 1 / np.sqrt(poster_2D_size / max_labels_num)
        s = np.array(poster_shape) * shrink_factor

        label_array_shape = np.array(
            min([(np.ceil(s[0]), np.ceil(s[1])),
                 (np.ceil(s[0]), np.floor(s[1])),
                 (np.floor(s[0]), np.ceil(s[1])),
                 (np.floor(s[0]), np.floor(s[1]))],
                key=lambda t: max_labels_num - np.prod(t) if max_labels_num - np.prod(t) > 0 else np.inf)).astype(int)

        x_splits = np.linspace(0, label_array_shape[1], class_num_x + 1).round().astype(int)
        y_splits = np.linspace(0, label_array_shape[0], class_num_y + 1).round().astype(int)

        labels = torch.zeros((1, class_num, *label_array_shape))
        for i, (x1, x2) in enumerate(zip(x_splits[:-1], x_splits[1:])):
            for j, (y1, y2) in enumerate(zip(y_splits[:-1], y_splits[1:])):
                labels[:, class_order[j, i], y1:y2, x1:x2] = 1

        return labels
