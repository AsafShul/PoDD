""" This module contains utilities for PoDD """

import torch
import numpy as np


def _get_fade_mask(i1, i2, other_dim_size, horizontal=True):
    overlap_pixels = i1[1] - i2[0]
    first_image_size, second_image_size = i1[1], i2[1] - i2[0]
    mask_func = _get_horizontal_fade_mask if horizontal else _get_vertical_fade_mask
    mask = mask_func(first_image_size, overlap_pixels, second_image_size, other_dim_size)
    return mask


def _get_horizontal_fade_mask(left_image_x, overlap_pixels, right_image_x, y):
    left_mask = torch.ones(1, y, left_image_x - max(overlap_pixels, 0))
    right_mask = torch.zeros(1, y, right_image_x - max(overlap_pixels, 0))
    overlap_pixels = abs(overlap_pixels)

    if overlap_pixels != 0:
        fade = 1 - torch.linspace(0, 1, overlap_pixels, dtype=torch.float32)
        middle = fade.view(1, 1, -1).expand(1, y, overlap_pixels)
        stack = [left_mask, middle, right_mask]
    else:
        stack = [left_mask, right_mask]

    concatenated_mask = torch.cat(stack, dim=2)
    return concatenated_mask


def _get_vertical_fade_mask(curr_y, overlap_pixels, row_y, row_x):
    up_mask = torch.ones(1, curr_y - max(overlap_pixels, 0), row_x)
    down_mask = torch.zeros(1, row_y - max(overlap_pixels, 0), row_x)
    overlap_pixels = abs(overlap_pixels)

    if overlap_pixels != 0:
        fade = 1 - torch.linspace(0, 1, overlap_pixels, dtype=torch.float32)
        middle = fade.view(1, overlap_pixels, 1).expand(1, overlap_pixels, row_x)
        stack = [up_mask, middle, down_mask]
    else:
        stack = [up_mask, down_mask]

    concatenated_mask = torch.cat(stack, dim=1)
    return concatenated_mask


def _pad_horizontal_tensors(left_tensor, next_tensor, mask):
    y = left_tensor.size(1)
    new_image_x = mask.size(-1)

    left = torch.zeros(left_tensor.size(0), y, new_image_x, dtype=left_tensor.dtype)
    right = torch.zeros(next_tensor.size(0), y, new_image_x, dtype=next_tensor.dtype)

    left[:, :, :left_tensor.size(2)] = left_tensor
    right[:, :, -next_tensor.size(2):] = next_tensor

    return left, right


def _pad_vertical_tensors(up_row_tensor, down_row_tensor, mask):
    new_image_y = mask.size(1)
    x = up_row_tensor.size(2)

    up = torch.zeros(up_row_tensor.size(0), new_image_y, x, dtype=up_row_tensor.dtype)
    down = torch.zeros(down_row_tensor.size(0), new_image_y, x, dtype=down_row_tensor.dtype)

    up[:, :up_row_tensor.size(1), :] = up_row_tensor
    down[:, -down_row_tensor.size(1):, :] = down_row_tensor

    return up, down


def _blend_row(row_tensors, row_indexes):
    indexes = [i for i, _ in row_indexes]
    masks = [_get_fade_mask(i1, i2, row_tensors[0].size(1)) for i1, i2 in zip(indexes[:-1], indexes[1:])]

    curr_tensor = row_tensors[0]
    for next_tensor, mask in zip(row_tensors[1:], masks):
        left_tensor, right_tensor = _pad_horizontal_tensors(curr_tensor, next_tensor, mask)
        curr_tensor = left_tensor * mask + right_tensor * (1 - mask)

    return curr_tensor


def combine_images_with_fade(images, full_size_x, full_size_y, image_num_x, image_num_y):
    """ Combines images into a mosaic with fade effect between them. """
    assert len(images) == image_num_x * image_num_y, 'wrong number of images'

    image_size_x, image_size_y = images[0].shape[1:]
    indexes = _get_patch_index_lst(image_size_x, image_size_y, full_size_x, full_size_y, image_num_x, image_num_y)

    blended_rows = [_blend_row(images[i * image_num_x: (i + 1) * image_num_x], indexes[i::image_num_y])
                    for i in range(image_num_y)]

    masks = [_get_fade_mask(i1[1], i2[1], blended_rows[0].size(-1), horizontal=False)
             for i1, i2 in zip(indexes[:image_num_x - 1], indexes[1:image_num_y])]

    final_image = blended_rows[0]
    for next_row, mask in zip(blended_rows[1:], masks):
        final_image, next_row = _pad_vertical_tensors(final_image, next_row, mask)
        final_image = final_image * mask + next_row * (1 - mask)

    return final_image


def get_crops_from_poster(poster: torch.Tensor, patch_size_x: int,
                          patch_size_y: int, patch_num_x: int,
                          patch_num_y: int, return_index=False, indexes_subset=None):
    """ Returns a tensor of crops from a poster."""
    poster_size_y, poster_size_x = poster.shape[2:]
    index_list = _get_patch_index_lst(patch_size_x, patch_size_y,
                                      poster_size_x, poster_size_y,
                                      patch_num_x, patch_num_y)
    if indexes_subset is not None:
        index_list = [index_list[i] for i in indexes_subset]

    crops = torch.stack([poster[0, :, y1:y2, x1:x2] for ((x1, x2), (y1, y2)) in index_list])
    return crops if not return_index else (crops, index_list)


def _get_patch_index_lst(patch_size_x: int, patch_size_y: int,
                         full_size_x: int, full_size_y: int, number_of_patches_x: int, number_of_patches_y: int):
    """ Returns a list of tuples of patch indexes from either a mosaic or a poster. """
    x_patch_steps = np.linspace(0, full_size_x - patch_size_x, number_of_patches_x).round().astype(int)
    y_patch_steps = np.linspace(0, full_size_y - patch_size_y, number_of_patches_y).round().astype(int)
    patch_lst = [((i, i + patch_size_x), (j, j + patch_size_y)) for i in x_patch_steps for j in y_patch_steps]

    return patch_lst
