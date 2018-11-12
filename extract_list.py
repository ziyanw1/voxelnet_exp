import os
import sys
import numpy as np
import glob
import random

OBJ_DIR = '/home/ziyanw1/PCDs/02958343'
LIST_DIR = 'lists'
f_car = glob.glob(os.path.join(OBJ_DIR, '*_4096.ply'))
print(len(f_car))

train_ratio = 0.7
test_ratio = 0.3
random.shuffle(f_car) 

sep_idx = int(train_ratio*len(f_car))
train_list = f_car[:sep_idx]
test_list = f_car[sep_idx:]

train_list = [l+'\n' for l in train_list] 
test_list = [l+'\n' for l in test_list]

train_list_name = 'train_car_list.txt'
test_list_name = 'test_car_list.txt'

with open(train_list_name, 'w') as f:
    f.writelines(train_list)

with open(test_list_name, 'w') as f:
    f.writelines(test_list)
