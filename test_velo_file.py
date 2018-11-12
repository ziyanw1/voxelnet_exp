import numpy as np
import os
import sys
import glob

data_dir = '../../data/KITTI/training/'
f_lidar = glob.glob(os.path.join(data_dir, 'velodyne', '*.bin'))

print(len(f_lidar))
proxy_idx = 1000
raw_lidar = np.fromfile(f_lidar[proxy_idx], dtype=np.float32).reshape((-1, 4))
print(raw_lidar.shape)
print(np.unique(raw_lidar[4, :]))

