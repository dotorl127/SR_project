import os

import vispy
from vispy.scene import visuals, SceneCanvas
from vispy.visuals.transforms import STTransform

from points2range import generate_range_image


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, default=None, help='specify location root dataset')
    args = parser.parse_args()

    raw_data_path = args.root_path
    src_data_path = raw_data_path + '/_out_src'
    dst_data_path = raw_data_path + '/_out_dst'

    src_pc_lst = os.listdir(src_data_path)
    dst_pc_lst = os.listdir(dst_data_path)
    if os.path.isfile(f'{raw_data_path}/diff.txt'):
        with open(f'{raw_data_path}/diff.txt', 'r') as f:
            diff = f.readlines()
            x_diff = float(diff[0].split(',')[0])
            y_diff = float(diff[0].split(',')[1].lstrip())
            z_diff = float(diff[0].split(',')[2].lstrip())
            print(f'X diff : {round(x_diff, 2)}')
            print(f'Y diff : {round(y_diff, 2)}')
            print(f'Z diff : {round(z_diff, 2)}')
    else:
        diff = None

    for s_pc_n, d_pc_n in zip(src_pc_lst, dst_pc_lst):
        src_img = generate_range_image(file_name=s_pc_n, src_loc=f'{src_data_path}')
        s2d_img = generate_range_image(file_name=s_pc_n, src_loc=f'{src_data_path}', diff=diff)
        dst_img = generate_range_image(file_name=d_pc_n, src_loc=f'{dst_data_path}')

        img_canvas = SceneCanvas(title='converted', keys='interactive', show=True, size=(1024, 194))
        img_grid = img_canvas.central_widget.add_grid()
        img_view = vispy.scene.widgets.ViewBox(border_color='white', parent=img_canvas.scene)

        img_grid.add_widget(img_view, 0, 0)
        img_vis = visuals.Image(cmap='viridis')
        img_view.add(img_vis)
        img_vis.set_data(src_img.transpose(1, 2, 0) / 200)
        img_vis.update()

        img_grid.add_widget(img_view, 0, 0)
        img_vis = visuals.Image(cmap='viridis')
        img_vis.transform = STTransform(translate=[0, 65])
        img_view.add(img_vis)
        img_vis.set_data(s2d_img.transpose(1, 2, 0) / 200)
        img_vis.update()

        img_grid.add_widget(img_view, 0, 0)
        img_vis = visuals.Image(cmap='viridis')
        img_vis.transform = STTransform(translate=[0, 130])
        img_view.add(img_vis)
        img_vis.set_data(dst_img.transpose(1, 2, 0) / 200)
        img_vis.update()

        vispy.app.run()
