""" implementation of the PoCO algorithm for optimizing the class order in the poster """

import torch
import itertools
import numpy as np
import pandas as pd

from tqdm import tqdm
from typing import List
from torch.nn import CosineSimilarity
from transformers import CLIPTokenizer, CLIPTextModel

from src.data_utils import CIFAR10_LABELS_DICT, CIFAR100_LABELS_DICT, TINY_IMAGENET_200_LABELS_DICT, CUB200_LABELS_DICT


class PoCO:
    """ Class for optimizing the class order in the poster """
    @staticmethod
    def _get_curr_neighbors(poster, i, j, rows, cols):
        indices = np.array([[i - 1, j], [i, j - 1], [i + 1, j], [i, j + 1]])
        valid_indices = (indices[:, 0] >= 0) & (indices[:, 0] < rows) & (indices[:, 1] >= 0) & (indices[:, 1] < cols)
        return list(filter(bool, [poster[i_, j_] for i_, j_ in indices[valid_indices]]))

    @staticmethod
    def _diagonal_order(poster_shape_: tuple):
        rows_num, cols_num = poster_shape_
        indices = []
        for line in range(1, (rows_num + cols_num)):
            start_col = max(0, line - rows_num)
            count = min(line, (cols_num - start_col), rows_num)
            for j in range(0, count):
                row = min(rows_num, line) - j - 1
                col = start_col + j
                indices.append((row, col))
        return indices

    @staticmethod
    def _calc_CLIP_distance_matrix(classes: List[str], device) -> pd.DataFrame:
        """ calculates the distance matrix between classes using CLIP text embeddings """
        cossim = CosineSimilarity(dim=0, eps=1e-6)
        model_id = 'openai/clip-vit-base-patch32'

        prompts = [c.replace('_', ' ') for c in classes]
        tokenizer = CLIPTokenizer.from_pretrained(model_id)
        text_encoder = CLIPTextModel.from_pretrained(model_id).to(device)

        text_inputs = tokenizer(
            prompts,
            padding="max_length",
            return_tensors="pt",
        ).to(device)

        text_embeddings = torch.flatten(text_encoder(text_inputs.input_ids.to(device))['last_hidden_state'], 1, -1)
        distance_matrix = np.zeros((len(classes), len(classes)))
        for i, j in tqdm(list(itertools.combinations(np.arange(len(classes)), 2))):
            distance_matrix[i, j] = distance_matrix[j, i] = 1 - cossim(text_embeddings[i], text_embeddings[j])

        return pd.DataFrame(distance_matrix, index=classes, columns=classes)

    @staticmethod
    def optimize_poster_class_order(poster_shape, dataset: str, device, as_strings: bool = False):
        """ calculates the optimal order of classes in the poster
        using greedy algorithm on the class similarity by its name """
        if dataset == 'cifar10':
            class_dict = CIFAR10_LABELS_DICT
        elif dataset == 'cifar100':
            class_dict = CIFAR100_LABELS_DICT
        elif dataset == 'cub-200':
            class_dict = CUB200_LABELS_DICT
        elif dataset == 'tiny-imagenet-200':
            class_dict = TINY_IMAGENET_200_LABELS_DICT
        else:
            raise ValueError(f'unknown dataset: {dataset},'
                             f' in order to run you need to add the labels mapping to the dataset_utils.py file')

        print('Optimizing poster class order...')
        classes = list(class_dict.keys())
        assert len(classes) == np.prod(poster_shape), 'wrong number of classes'
        poster_order = np.full(poster_shape, '', dtype='<U32')

        distance_matrix = PoCO._calc_CLIP_distance_matrix(classes, device)
        for i, j in PoCO._diagonal_order(poster_shape):
            neighbors = PoCO._get_curr_neighbors(poster_order, i, j, *poster_shape)
            closest = classes[0] if not neighbors else distance_matrix.loc[classes, neighbors].sum(1).idxmin()
            poster_order[i, j] = closest
            classes.remove(closest)

        if as_strings:
            return poster_order

        poster_order = np.vectorize(class_dict.get)(poster_order)

        return poster_order
