import argparse
import os
from datetime import datetime
import random

import numpy as np
import yaml
import torch

import models

import vispy
from vispy.scene import visuals, SceneCanvas
from vispy.visuals.transforms import STTransform

from range2points import range2points
from points2range import generate_range_image


def demo(dataset_path, shuffle, model):
    model.eval()

    filenames = sorted(os.listdir(dataset_path + '/_out_src/'))

    if shuffle:
        random.shuffle(filenames)

    for filename in filenames:
        with open(dataset_path + '/diff.txt', 'r') as f:
            diff = f.readlines()
            x_diff = float(diff[0].split(',')[0])
            y_diff = float(diff[0].split(',')[1].lstrip())
            z_diff = float(diff[0].split(',')[2].lstrip())
            print(f'X diff : {round(x_diff, 2)}')
            print(f'Y diff : {round(y_diff, 2)}')
            print(f'Z diff : {round(z_diff, 2)}')

        start_t = datetime.now()
        s2g_img = generate_range_image(file_name=filename, src_loc=f'{dataset_path}/_out_src/', diff=diff)
        pred = model(torch.from_numpy(s2g_img.reshape(1, *s2g_img.shape) / 200).cuda())
        print(f'latency : {(datetime.now() - start_t).seconds}.{(datetime.now() - start_t).microseconds}sec')
        pred = pred.detach()

        gt_img = generate_range_image(file_name=filename, src_loc=f'{dataset_path}/_out_dst/')

        img_canvas = SceneCanvas(title='converted', keys='interactive', show=True, size=(1024, 194))
        img_grid = img_canvas.central_widget.add_grid()
        img_view = vispy.scene.widgets.ViewBox(parent=img_canvas.scene)

        img_grid.add_widget(img_view, 0, 0)
        img_vis = visuals.Image(cmap='viridis')
        img_view.add(img_vis)
        img_vis.set_data(s2g_img.transpose(1, 2, 0) / 200)
        img_vis.update()

        img_grid.add_widget(img_view, 0, 0)
        img_vis = visuals.Image(cmap='viridis')
        img_vis.transform = STTransform(translate=[0, 65])
        img_view.add(img_vis)
        img_vis.set_data(pred[0].permute(1, 2, 0).cpu().numpy())
        img_vis.update()

        img_grid.add_widget(img_view, 0, 0)
        img_vis = visuals.Image(cmap='viridis')
        img_vis.transform = STTransform(translate=[0, 130])
        img_view.add(img_vis)
        img_vis.set_data(gt_img.transpose(1, 2, 0) / 200)
        img_vis.update()

        # range image convert to point cloud
        pred_pt = range2points(pred.cpu().numpy())

        src_pt = np.load(dataset_path + '/_out_src/' + filename)[:, :-1]
        gt_pt = np.load(dataset_path + '/_out_dst/' + filename)[:, :-1]

        # point cloud visualization
        canvas = SceneCanvas(keys='interactive', show=True, title='source')
        scan_view = vispy.scene.widgets.ViewBox(parent=canvas.scene)
        grid = canvas.central_widget.add_grid()
        grid.add_widget(scan_view, 0, 0)
        scan_vis = visuals.Markers()
        scan_vis.set_data(src_pt, face_color=(1, 1, 1, 1), size=1, edge_width=.0)
        scan_view.add(scan_vis)
        scan_view.camera = 'turntable'  # SHIFT + LMB: translate the center point
        visuals.XYZAxis(parent=scan_view.scene)

        # point cloud visualization
        src_canvas = SceneCanvas(keys='interactive', show=True, title='predicted')
        src_scan_view = vispy.scene.widgets.ViewBox(parent=canvas.scene)
        src_grid = src_canvas.central_widget.add_grid()
        src_grid.add_widget(src_scan_view, 0, 0)
        src_scan_vis = visuals.Markers()
        src_scan_vis.set_data(pred_pt, face_color=(1, 1, 1, 1), size=1, edge_width=.0)
        src_scan_view.add(src_scan_vis)
        src_scan_view.camera = 'turntable'  # SHIFT + LMB: translate the center point
        visuals.XYZAxis(parent=src_scan_view.scene)

        gt_canvas = SceneCanvas(keys='interactive', show=True, title='target')
        gt_scan_view = vispy.scene.widgets.ViewBox(parent=gt_canvas.scene)
        gt_grid = gt_canvas.central_widget.add_grid()
        gt_grid.add_widget(gt_scan_view, 0, 0)
        gt_scan_vis = visuals.Markers()
        gt_scan_vis.set_data(gt_pt, face_color=(1, 1, 1, 1), size=1, edge_width=.0)
        gt_scan_view.add(gt_scan_vis)
        gt_scan_view.camera = 'turntable'  # SHIFT + LMB: translate the center point
        visuals.XYZAxis(parent=gt_scan_view.scene)

        vispy.app.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--model')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--dataset_path', default='/media/moon/extraDB/CARLA_TEST_DATASET')
    parser.add_argument('--shuffle', type=str, default='False')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    model_spec = torch.load(args.model)['model']
    model = models.make(model_spec, load_sd=True).cuda()
    print('Model load complete.')

    shuffle = False
    if str(args.shuffle).lower()[0] == 'y' or str(args.shuffle).lower()[0] == 't':
        shuffle = True

    dataset_path = args.dataset_path
    if args.dataset_path[-1] == '/':
        dataset_path = dataset_path[:-1]

    demo(dataset_path, shuffle, model)
