import numpy as np


def range2points(img):
    # preprocess yaw array shape : (64, 1024)
    yaw = np.linspace(np.pi, -np.pi, 1024)
    yaw = np.tile(yaw, [64, 1])

    # preprocess pitch array shape : (64, 1024)
    up_pitch = 90 - 15
    down_pitch = 90 + 25
    pitch = np.linspace(up_pitch, down_pitch, 64)
    pitch = np.deg2rad(pitch)
    pitch = np.tile(pitch, [1024, 1]).T

    # calculate range
    if img.shape[0] == 3:  # for 3ch image
        img *= 256
        range = img[0, :] + (img[1, :] * 256) + (img[2, :] * 256**2)
        range = range / (256**3 - 1)
        range = range * 1000
    else:  # for 1ch image
        range = img.squeeze()
        range *= 200

    # masking invalid range value
    mask = range > 0
    mask += range < 200
    mask = mask.reshape(mask.shape[0] * mask.shape[1])

    # calculate coordinates
    x = range * np.sin(pitch) * np.cos(yaw)
    x = x.reshape(x.shape[0] * x.shape[1])
    y = range * np.sin(pitch) * np.sin(yaw)
    y = y.reshape(y.shape[0] * y.shape[1])
    z = range * np.cos(pitch)
    z = z.reshape(z.shape[0] * z.shape[1])

    # concatenate each axis values to completion point cloud data
    points = np.concatenate((x, y, z))
    points = points.reshape(3, -1).T[mask]

    return points
