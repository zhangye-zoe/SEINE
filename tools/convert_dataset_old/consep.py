import argparse
import math
import os
import os.path as osp
from functools import partial

import mmcv
import numpy as np
from PIL import Image
from scipy.io import loadmat

from tiseg.datasets.utils import colorize_seg_map


def convert_mat_to_array(mat, array_key='inst_map', save_path=None):
    """Convert matlab format array file to numpy array file."""
    if isinstance(mat, str):
        mat = loadmat(mat)

    mat = mat[array_key]

    if save_path is not None:
        pass

    return mat


def pillow_save(save_path, array, palette=None):
    """storage image array by using pillow."""
    image = Image.fromarray(array.astype(np.uint8))
    if palette is not None:
        image = image.convert('P')
        image.putpalette(palette)
    image.save(save_path)


# NOTE: new style patch crop.
def crop_patches(image, c_size):
    h, w = image.shape[:2]
    patches = []

    if h % c_size == 0:
        h_overlap = 0
    else:
        div = h // c_size
        h_overlap = math.ceil(((div + 1) * c_size - h) / div)

    if w % c_size == 0:
        w_overlap = 0
    else:
        div = w // c_size
        w_overlap = math.ceil(((div + 1) * c_size - w) / div)

    for i in range(0, h - c_size + 1, c_size - h_overlap):
        for j in range(0, w - c_size + 1, c_size - w_overlap):
            patch = image[i:i + c_size, j:j + c_size]
            patches.append(patch)

    return patches


def parse_single_item(item, raw_image_folder, raw_label_folder, new_path, crop_size):
    """meta process of single item data."""

    image_path = osp.join(raw_image_folder, item + '.png')
    label_path = osp.join(raw_label_folder, item + '.mat')

    # image & label extraction
    image = np.array(Image.open(image_path))[:, :, :3]
    instance_label = convert_mat_to_array(label_path)
    semantic_label = (instance_label > 0).astype(np.uint8)

    # split map into patches
    if crop_size != 0:
        image_patches = crop_patches(image, crop_size)
        instance_patches = crop_patches(instance_label, crop_size)
        semantic_patches = crop_patches(semantic_label, crop_size)

        assert len(image_patches) == len(instance_patches) == len(semantic_patches)

        item_len = len(image_patches)
        # record patch item name
        sub_item_list = [f'{item}_{i}' for i in range(item_len)]
    else:
        image_patches = [image]
        instance_patches = [instance_label]
        semantic_patches = [semantic_label]
        # record patch item name
        sub_item_list = [item]

    # patch storage
    patch_batches = zip(image_patches, instance_patches, semantic_patches)
    for patch, sub_item in zip(patch_batches, sub_item_list):
        # jump when exists
        if osp.exists(osp.join(new_path, sub_item + '.png')):
            continue
        # save image
        pillow_save(osp.join(new_path, sub_item + '.png'), patch[0])
        # save instance level label
        np.save(osp.join(new_path, sub_item + '_instance.npy'), patch[1])
        pillow_save(osp.join(new_path, sub_item + '_instance_colorized.png'), colorize_seg_map(patch[1]))
        # save semantic level label
        palette = np.zeros((2, 3), dtype=np.uint8)
        palette[0, :] = (0, 0, 0)
        palette[1, :] = (255, 255, 2)
        pillow_save(osp.join(new_path, sub_item + '_semantic.png'), patch[2], palette)

    return {item: sub_item_list}


def convert_cohort(img_folder, lbl_folder, new_folder, item_list, c_size=0):
    if not osp.exists(new_folder):
        os.makedirs(new_folder, 0o775)

    fix_kwargs = {
        'raw_image_folder': img_folder,
        'raw_label_folder': lbl_folder,
        'new_path': new_folder,
        'crop_size': c_size,
    }

    meta_process = partial(parse_single_item, **fix_kwargs)

    real_item_dict = {}
    results = mmcv.track_parallel_progress(meta_process, item_list, 4)
    [real_item_dict.update(result) for result in results]

    return real_item_dict


def parse_args():
    parser = argparse.ArgumentParser('Convert cpm17 dataset.')
    parser.add_argument('root_path', help='dataset root path.')
    parser.add_argument(
        '-c', '--crop-size', type=int, default=0, help='the crop size of fix crop in dataset convertion operation')

    return parser.parse_args()


def main():
    args = parse_args()
    root_path = args.root_path
    crop_size = args.crop_size

    for split, name in [('train', 'Train'), ('test', 'Test')]:
        raw_root = osp.join(root_path, 'consep', name)

        raw_img_folder = osp.join(raw_root, 'Images')
        raw_lbl_folder = osp.join(raw_root, 'Labels')

        item_list = [x.rstrip('.png') for x in os.listdir(raw_img_folder) if '.png' in x]

        if split == 'test':
            new_root = osp.join(root_path, split, 'c0')
            convert_cohort(raw_img_folder, raw_lbl_folder, new_root, item_list, 0)
        else:
            new_root = osp.join(root_path, split, f'c{crop_size}')
            convert_cohort(raw_img_folder, raw_lbl_folder, new_root, item_list, c_size=crop_size)

        item_list = [x.rstrip('_instance.npy') for x in os.listdir(new_root) if '_instance.npy' in x]

        if split == 'train':
            name = f'train_c{crop_size}.txt'
        else:
            name = 'test_c0.txt'
        with open(osp.join(root_path, name), 'w') as fp:
            [fp.write(item + '\n') for item in item_list]


if __name__ == '__main__':
    main()
