import os
import sys
import cv2
import numpy as np
import glob
from plyfile import PlyData, PlyElement
from util_pcds import *
import pickle
from utils_notf import *
from multiprocessing import Pool, TimeoutError

DEBUG = False 
VIS = True
GENERATE_PLY = True
POOL_NUM = 8
MODE = 'training'
if MODE == 'training':
    from config import cfg
else:
    from config_val import cfg
sample_ratios = [2, 4, 8]
DATA_DIR = os.path.join('../../data/KITTI', MODE)
OBJ_DIR = '/home/ziyanw1/PCDs/02958343'
SCENE_DIR = os.path.join(DATA_DIR, 'velodyne')
#OBJ_OUT_DIR = os.path.join(DATA_DIR, 'template')
#SCENE_OUT_DIR = os.path.join(DATA_DIR, 'syn_scene')
#LABEL_OUT_DIR = os.path.join(DATA_DIR, 'syn_label')
#VIS_OUT_DIR = os.path.join(DATA_DIR, 'vis')
#META_OUT_DIR = os.path.join(DATA_DIR, 'meta')
RECON_OUT_DIR = os.path.join(DATA_DIR, 'recon_pc')

if not os.path.exists(RECON_OUT_DIR):
    os.mkdir(RECON_OUT_DIR)


def create_syn_scene_obj(scene_file_name, obj_file_name, generate_ply=GENERATE_PLY, vis=VIS):

    ## parse file
    s_set = scene_file_name.split('/')
    scene_name = s_set[-1][:-4]
    s_set = obj_file_name.split('/')
    obj_name = s_set[-1][:-4]
    
    ## dump template model both .ply and .bin
    obj_outfile_pcd = os.path.join(RECON_OUT_DIR, scene_name+'.pcd')
    if generate_ply:
        command = './pcl_pcd2ply {} {}'.format(obj_outfile_pcd, obj_outfile_pcd[:-4]+'.ply')
        os.system(command)


if __name__ == "__main__":
    
    if DEBUG:
        ## for debug
        scene_list = ['001201.bin', '001830.bin', '002453.bin']
        obj_list = ['5801f9eb726b56448b9c28e7b121fdbc_4096.ply']

        scene_list = [os.path.join(DATA_DIR, 'velodyne', s) for s in scene_list]
        obj_list = [os.path.join(OBJ_DIR, o) for o in obj_list]

        for s in scene_list:
            o = np.random.choice(obj_list)
            create_syn_scene_obj(s, o, True, True)

    else:
        #pool = Pool(POOL_NUM)
        f_scene = glob.glob(os.path.join(SCENE_DIR, '*.bin'))
        f_obj = glob.glob(os.path.join(OBJ_DIR, '*_4096.ply'))
        
        if MODE is 'training':
            f_obj_use = np.random.choice(f_obj[:int(0.7*len(f_obj))], size=len(f_scene))
        else:
            f_obj_use = np.random.choice(f_obj[int(0.7*len(f_obj)):], size=len(f_scene))
            

        #pool.imap(create_syn_scene_obj, f_scene, f_obj_use)
        #pool.close()
        for idx, (s_path, o_path) in enumerate(zip(f_scene, f_obj_use)):
            create_syn_scene_obj(s_path, o_path)
            print('------- Scene {} is generated -------'.format(idx))
