import octomap as oct
import numpy as np


def test(path):
    print(f'TEST START {path}')
    res = 0.1
    points = np.load(path + '/pred.npy')
    gt_points = np.load(path + '/gt.npy')

    tpr_arr, fpr_arr = [], []

    for _ in range(9):
        print(f'== {res} TEST START')
        p_oct = oct.OcTree(res)
        p_oct.insertPointCloud(
            pointcloud=points,
            origin=np.array([0, 0, 0], dtype=float),
        )
        p_occupied, p_empty = p_oct.extractPointCloud()

        g_oct = oct.OcTree(res)
        g_oct.insertPointCloud(
            pointcloud=gt_points,
            origin=np.array([0, 0, 0], dtype=float),
        )

        labels = g_oct.getLabels(p_occupied)
        tp = np.sum(labels == 1)
        fp = labels.shape[0] - tp
        labels = g_oct.getLabels(p_empty)
        tn = np.sum(labels == 0)
        fn = labels.shape[0] - tn

        print(f'TP : {tp}')
        print(f'FP : {fp}')
        print(f'TN : {tn}')
        print(f'FN : {fn}')

        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)
        print(f'TPR : {tpr}')
        print(f'FPR : {fpr}')
        tpr_arr.append(tpr)
        fpr_arr.append(fpr)
        print(f'== {res} TEST COMPLETE')
        res *= 2

    with open(path + 'test_ret.txt', 'w') as f:
        f.write('TPR : ')
        f.write(''.join(str(tpr_arr)))
        f.write('\n')
        f.write('FPR : ')
        f.write(''.join(str(fpr_arr)))

    print(f'{path} TEST COMPLETE SAVE DATA')


# test('/media/moon/extraDB/liif/test/edsr-lsr-pts')
# test('/media/moon/extraDB/liif/test/edsr-baseline-pts')