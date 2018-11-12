import os
import sys

PATH = '../../data/toyset/training/calib'

for i in range(1,1000):
    ori_file = os.path.join(PATH, '001201_0.txt')
    dst_file = os.path.join(PATH, '001201_{}.txt'.format(i))
    command = 'cp {} {}'.format(ori_file, dst_file)
    os.system(command)
