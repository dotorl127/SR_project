import os
import numpy as np
import multiprocessing
from functools import partial
from tqdm import tqdm


def spherical_projection(proj_W, proj_H, points):
    range = np.linalg.norm(points, 2, axis=1)
    x_, y_, z_ = points[:, 0], points[:, 1], points[:, 2]

    yaw = np.arctan2(y_, x_)
    pitch = np.arccos(z_ / range)
    fov = np.deg2rad(40)
    up_pitch = np.deg2rad(90 - 15)

    # normalize angle
    proj_x = 0.5 * (1.0 - yaw / np.pi)
    proj_y = (pitch - up_pitch) / fov  # pitch normalization

    proj_x *= proj_W
    proj_y *= proj_H

    # image horizontal fov filtering
    proj_x = np.floor(proj_x)
    proj_x = np.minimum(proj_W - 1, proj_x)
    proj_x = np.maximum(0, proj_x).astype(np.int32)

    # image vertical fov filtering
    proj_y = np.floor(proj_y)
    proj_y = np.minimum(proj_H - 1, proj_y)
    proj_y = np.maximum(0, proj_y).astype(np.int32)

    # order in decreasing for make ascending range
    order = np.argsort(range)[::-1]
    range = range[order]
    proj_x = proj_x[order]
    proj_y = proj_y[order]

    spherical_array = np.zeros((proj_H, proj_W), dtype=np.float32)
    spherical_array[proj_y, proj_x] = range

    return spherical_array


def convert_RGB(spherical_array, proj_W, proj_H):
    # convert range to RGB normalized value
    rgb_sum_val = spherical_array / 1000 * (pow(256, 3) - 1)
    b = np.floor(rgb_sum_val / pow(256, 2))
    rg_sum_val = rgb_sum_val - (b * pow(256, 2))
    g = np.floor(rg_sum_val / 256)
    r = np.floor(rg_sum_val - (g * 256))

    range_array = np.zeros((proj_H, proj_W, 3), dtype=np.float32)
    range_array[:, :, 0] = r
    range_array[:, :, 1] = g
    range_array[:, :, 2] = b

    return range_array


def generate_range_image(file_name, src_loc, dst_loc=None, diff=None, proj_W=1024, proj_H=64):
    points = np.load(f'{src_loc}/{file_name}')[:, :3]  # for npy file

    x, y, z = 0, 0, 0
    if diff is not None:
        x, y, z = str(diff[0]).strip().split(', ')
        x, y, z = float(x), float(y), float(z)
    points -= x, y, z

    spherical_array = spherical_projection(proj_W, proj_H, points)

    img_file_name = str(file_name).rstrip('.npy')

    # just save 1 channel range npy data
    range_array = spherical_array.reshape((1, spherical_array.shape[0], spherical_array.shape[1]))

    if dst_loc is not None:
        np.save(f'{dst_loc}/{img_file_name}.npy', range_array)
    else:
        return range_array


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, default='/media/moon/extraDB/CARLA_DATASET',
                        help='specify location root dataset')
    parser.add_argument('--up_fov', type=float, default=15.0, help='specify up fov, degree')
    parser.add_argument('--down_fov', type=float, default=-25.0, help='specify down fov, degree')
    parser.add_argument('--w', type=int, default=1024, help='specify image size of width, pixel')
    parser.add_argument('--h', type=int, default=64, help='specify image size of height, pixel')
    args = parser.parse_args()

    raw_dir_list = sorted(os.listdir(args.root_path))

    for raw in raw_dir_list:
        raw_data_path = f'{args.root_path}/{raw}'

        if os.path.exists(f'{raw_data_path}/train') and os.path.exists(f'{raw_data_path}/GT'):
            print('#' * 10, raw_data_path, '#' * 10)
            print('convert range image has already completed, skip')
            continue

        src_data_path = raw_data_path + '/_out_src'
        dst_data_path = raw_data_path + '/_out_dst'

        src_pc_lst = os.listdir(src_data_path)
        dst_pc_lst = os.listdir(dst_data_path)
        if os.path.isfile(f'{raw_data_path}/diff.txt'):
            with open(f'{raw_data_path}/diff.txt', 'r') as f:
                diff = f.readlines()
        else:
            diff = None

        # generate S' from S data
        print('#'*10, raw_data_path, '#'*10)
        print('start generate train range image from PCD data.')

        os.makedirs(f'{raw_data_path}/train', exist_ok=True)
        # transform with diff, offset + spherical projection
        convert_fn = partial(generate_range_image, src_loc=src_data_path, dst_loc=f'{raw_data_path}/train', diff=diff,
                             proj_W=args.w, proj_H=args.h)

        with multiprocessing.Pool() as p:
            list(tqdm(p.imap(convert_fn, src_pc_lst), total=len(src_pc_lst)))

        # generate T' from T data
        print('start generate ground-truth range image from PCD data.')

        os.makedirs(f'{raw_data_path}/GT', exist_ok=True)
        # only spherical projection
        convert_fn = partial(generate_range_image, src_loc=dst_data_path, dst_loc=f'{raw_data_path}/GT',
                             proj_W=args.w, proj_H=args.h)

        with multiprocessing.Pool() as p:
            list(tqdm(p.imap(convert_fn, dst_pc_lst), total=len(dst_pc_lst)))

        print('done.')
