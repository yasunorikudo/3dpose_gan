import argparse
import json
import numpy as np
import pickle

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata', type=str, default='data/h36m/Release-v1.1/metadata.xml',
                        help='Path to metadata.xml')
    args = parser.parse_args()

    with open(args.metadata) as f:
        w0 = [float(v) for v in f.read().split('<w0>[')[1].split(']</w0>')[0].split()]
    cam_names = ['54138969', '55011271', '58860488', '60457274']

    camera_params = {}
    for s in range(1, 12):
        subject = 'S{}'.format(s)
        if not subject in camera_params.keys():
            camera_params[subject] = {}
        for c in range(1, 5):
            cam_name = cam_names[c - 1]
            s1 = 6 * ((c - 1) * 11 + (s - 1))
            s2 = 264 + (c - 1) * 9
            camera_params[subject][cam_name] = dict(
                a = np.array(w0[s1:s1+3])[None].T,
                T = np.array(w0[s1+3:s1+6])[None].T,
                c = np.array(w0[s2+2:s2+4])[None].T,
                f = np.array(w0[s2:s2+2])[None].T,
                k = np.array(w0[s2+4:s2+7])[None].T,
                p = np.array(w0[s2+7:s2+9])[None].T)

    with open('data/h36m/cameras.pkl', 'wb') as f:
        pickle.dump(camera_params, f)
