import os
import argparse
import random
import json
from shutil import copyfile
from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, default='/media/moon/extraDB/CARLA_DATASET',
                        help='specify location root dataset')
    parser.add_argument('--dst_path', type=str, default='/media/moon/extraDB/liif/load/merged_dataset',
                        help='specify location destination dataset')
    args = parser.parse_args()

    root_path = args.root_path
    dst_path = args.dst_path
    dir_list = sorted(os.listdir(root_path))

    os.makedirs(f'{dst_path}/_out_src', exist_ok=True)
    os.makedirs(f'{dst_path}/_out_dst', exist_ok=True)
    os.makedirs(f'{dst_path}/dst_lidar_label', exist_ok=True)
    os.makedirs(f'{dst_path}/train', exist_ok=True)
    os.makedirs(f'{dst_path}/GT', exist_ok=True)

    split = {
        'train': [],
        'val': []
    }
    cnt = 0
    exist = False

    if os.path.exists(f'{dst_path}/split.json'):
        exist = True
        with open(f'{dst_path}/split.json', 'r') as f:
            split = json.load(f)
            cnt = int(max(max(split['train']), max(split['val'])).split('.')[0]) + 1

    cnt_list = []

    for dir_name in dir_list:
        up_file_list = sorted(os.listdir(f'{root_path}/{dir_name}/_out_src'))

        for up_file in tqdm(up_file_list, desc=f'{dir_name}'):
            file_name = up_file.split('.')[0]
            copyfile(f'{root_path}/{dir_name}/_out_src/{file_name}.npy', f'{dst_path}/_out_src/{cnt:08d}.npy')
            copyfile(f'{root_path}/{dir_name}/_out_dst/{file_name}.npy', f'{dst_path}/_out_dst/{cnt:08d}.npy')
            copyfile(f'{root_path}/{dir_name}/dst_lidar_label/{file_name}.txt',
                     f'{dst_path}/dst_lidar_label/{cnt:08d}.txt')
            copyfile(f'{root_path}/{dir_name}/train/{file_name}.npy', f'{dst_path}/train/{cnt:08d}.npy')
            copyfile(f'{root_path}/{dir_name}/GT/{file_name}.npy', f'{dst_path}/GT/{cnt:08d}.npy')
            cnt_list.append(f'{cnt:08d}.npy')
            cnt += 1

    random.shuffle(cnt_list)
    train_num = int(len(cnt_list) * 0.8)

    if not exist:
        split['train'] = sorted(cnt_list[:train_num])
        split['val'] = sorted(cnt_list[train_num:])
        with open(f'{dst_path}/split.json', 'w') as f:
            json.dump(split, f, indent=2)
    else:
        with open(f'{dst_path}/split.json', 'r') as f:
            split = json.load(f)
            split['train'] += sorted(cnt_list[:train_num])
            split['val'] += sorted(cnt_list[train_num:])
        with open(f'{dst_path}/split.json', 'w') as f:
            json.dump(split, f, indent=2)
