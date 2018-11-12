import os
import sys

split_dir = '../../data/ImageSets'
base_path = '../../data/KITTI/'

with open(os.path.join(split_dir, 'val.txt'), 'r') as f:
    lines = f.readlines()

for line in lines:
    train_path = os.path.join(base_path, 'training')
    val_path = os.path.join(base_path, 'validation')

    command = 'mv {} {}'.format(os.path.join(train_path, 'calib', '{}.txt'.format(line[:-1])), \
        os.path.join(val_path, 'calib', '{}.txt'.format(line[:-1])))
    print(command)
    os.system(command)
    
    command = 'mv {} {}'.format(os.path.join(train_path, 'image_2', '{}.png'.format(line[:-1])), \
        os.path.join(val_path, 'image_2', '{}.png'.format(line[:-1])))
    print(command)
    os.system(command)
    
    command = 'mv {} {}'.format(os.path.join(train_path, 'label_2', '{}.txt'.format(line[:-1])), \
        os.path.join(val_path, 'label_2', '{}.txt'.format(line[:-1])))
    print(command)
    os.system(command)
    
    command = 'mv {} {}'.format(os.path.join(train_path, 'velodyne', '{}.bin'.format(line[:-1])), \
        os.path.join(val_path, 'velodyne', '{}.bin'.format(line[:-1])))
    print(command)
    os.system(command)
