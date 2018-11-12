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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DEBUG = False 
VIS = True
GENERATE_PLY = True
POOL_NUM = 8
MODE = 'validation'
if MODE == 'training':
    from config import cfg
else:
    from config_val import cfg
sample_ratios = [2, 4, 8]
DATA_DIR = os.path.join('../../data/KITTI', MODE)
OBJ_DIR = '/home/ziyanw1/PCDs/02958343'
SCENE_DIR = os.path.join(DATA_DIR, 'velodyne')
OBJ_OUT_DIR = os.path.join(DATA_DIR, 'template')
SCENE_OUT_DIR = os.path.join(DATA_DIR, 'syn_scene')
LABEL_OUT_DIR = os.path.join(DATA_DIR, 'syn_label')
VIS_OUT_DIR = os.path.join(DATA_DIR, 'vis')
META_OUT_DIR = os.path.join(DATA_DIR, 'meta')


def create_syn_scene_obj(scene_file_name, obj_file_name, generate_ply=GENERATE_PLY, vis=VIS):

    ## parse file
    s_set = scene_file_name.split('/')
    scene_name = s_set[-1][:-4]
    s_set = obj_file_name.split('/')
    obj_name = s_set[-1][:-4]
    
    ## load scene here
    #scene_name = os.path.join(kitti_data_dir, scene_file)
    scene_pc = np.fromfile(scene_file_name, dtype=np.float32).reshape(-1, 4)
    scene_pc_pcd = np.concatenate((scene_pc[:, :3], np.zeros_like(scene_pc[:, :3])), axis=1)
    scene_pc_pcd[:, 5] = 255 ## assign green for background
    
    ## sample azim here
    azim = np.random.choice([0, 90, 180, 270])
    azim += np.random.uniform(-5, 5)
    R = azim2rot_deg(azim)
    
    ## sample translation here
    scene_x_var = np.sqrt(np.var(scene_pc[:, 0]))
    scene_y_var = np.sqrt(np.var(scene_pc[:, 1]))
    t_obj = np.zeros((1,3), dtype=np.float32)
    t_obj[0, 0] = np.random.uniform(scene_x_var, scene_x_var)
    t_obj[0, 1] = np.random.uniform(scene_y_var, scene_y_var)
    #t_obj[0, 2] -= 1    ## make the car on the ground
    
    ## load car here
    ply_data = PlyData.read(obj_file_name)
    
    pc_coord = []
    pc_coord.append(ply_data.elements[0].data['x'])
    pc_coord.append(ply_data.elements[0].data['z'])
    pc_coord.append(ply_data.elements[0].data['y'])
    pc_coord = np.squeeze(np.dstack(pc_coord))
    template_coord = pc_coord.copy()
    template_coord_pcd = np.concatenate((template_coord, np.zeros_like(template_coord)), axis=1) 
    template_coord = np.concatenate((template_coord, np.zeros((pc_coord.shape[0], 1))), axis=1)
    
    pc_coord *= 4 ## scale to real car 
    
    ## for 3D bbox
    h = np.max(pc_coord[:, 0]) - np.min(pc_coord[:, 0])
    w = np.max(pc_coord[:, 1]) - np.min(pc_coord[:, 1])
    l = np.max(pc_coord[:, 2]) - np.min(pc_coord[:, 2])
    
    ## car in plane rotation happens here
    ## coordinates in order of [x, y, z] ??
    new_pc_coord = np.transpose(np.matmul(R, np.transpose(pc_coord)))
    pc_coord = new_pc_coord

    ## translation happens here
    pc_coord = pc_coord + np.tile(t_obj, [pc_coord.shape[0], 1])
    
    ## down sampleing happens here
    sample_ratio = np.random.choice(sample_ratios)
    pc_coord = pc_coord[::sample_ratio, ...] 
    
    pc_coord_pcd = np.concatenate((pc_coord, np.zeros_like(pc_coord)), axis=1)
    pc_coord_pcd[:, 4] = 255 ## assign red for car
    pc_coord = np.concatenate((pc_coord, np.zeros((pc_coord.shape[0], 1))), axis=1)


    ## create new pc file
    vertex = np.concatenate((pc_coord_pcd, scene_pc_pcd), axis=0)

    ## dump template model both .ply and .bin
    obj_outfile_pcd = os.path.join(OBJ_OUT_DIR, scene_name+'.pcd')
    pcd_from_array(obj_outfile_pcd, template_coord_pcd)
    obj_outfile_bin = os.path.join(OBJ_OUT_DIR, scene_name+'.bin')
    template_coord.astype(np.float32).tofile(obj_outfile_bin)
    if generate_ply:
        command = './pcl_pcd2ply {} {}'.format(obj_outfile_pcd, obj_outfile_pcd[:-4]+'.ply')
        os.system(command)

    ## dump scene with template both p.ly and .bin
    scene_outfile_pcd = os.path.join(SCENE_OUT_DIR, scene_name+'.pcd')
    pcd_from_array(scene_outfile_pcd, vertex)
    scene_outfile_bin = os.path.join(SCENE_OUT_DIR, scene_name+'.bin')
    scene_gen_pc = np.concatenate((pc_coord, scene_pc), axis=0)
    scene_gen_pc.astype(np.float32).tofile(scene_outfile_bin)
    if generate_ply:
        command = './pcl_pcd2ply {} {}'.format(scene_outfile_pcd, scene_outfile_pcd[:-4]+'.ply')
        os.system(command)

    ### for debug
    #load_scene = np.fromfile(scene_outfile_bin, dtype=np.float32).reshape(-1, 4)
    #err = np.abs(load_scene - scene_gen_pc)
    #print(np.max(err))
    #sys.exit()
    
    ## dump annotations: label 
    ## x y z -> z y x in label
    gt_bbox = [(np.min(pc_coord[:, 0])+np.max(pc_coord[:, 0]))/2.0, (np.min(pc_coord[:, 1])+np.max(pc_coord[:, 1]))/2.0 \
        , (np.min(pc_coord[:, 2])+np.max(pc_coord[:, 2]))/2.0, h, w, l, np.deg2rad(azim)]
    gt_bboxes = np.asarray([gt_bbox])
    label_line = 'Car -1 -1 -10 -1 -1 -1 -1 {:.02f} {:.02f} {:.02f} {:.02f} {:.02f} {:.02f} {:.02f}\n'.format(
        l, w, h, -gt_bbox[1], gt_bbox[2], gt_bbox[0], np.deg2rad(90-azim)) 
    label_file_name = os.path.join(LABEL_OUT_DIR, scene_name+'.txt') 
    with open(label_file_name, 'w') as f:
        f.writelines([label_line])

    ## dumpe meta data: obj name, scene, postion rotation scale
    meta_info = {}
    meta_info['obj_name'] = obj_name
    meta_info['scene_name'] = scene_name
    meta_info['sample_ratio'] = sample_ratio
    meta_info['azimuth'] = azim
    meta_info['translation'] = t_obj
    meta_file_name = os.path.join(META_OUT_DIR, scene_name+'.pkl')
    with open(meta_file_name, 'wb') as f:
        pickle.dump(meta_info, f)

    if vis:
        batch_gt_boxes3d = label_to_gt_box3d(
            [[label_line]], cls='Car', coordinate='lidar')
        P, Tr, R = load_calib( os.path.join( cfg.CALIB_DIR, scene_name + '.txt' ) )
        bird_view = lidar_to_bird_view_img(scene_gen_pc, factor=cfg.BV_LOG_FACTOR)
        bird_view = draw_lidar_box3d_on_birdview(bird_view, batch_gt_boxes3d[0], 1, batch_gt_boxes3d[0], factor=cfg.BV_LOG_FACTOR, P2=P, T_VELO_2_CAM=Tr, R_RECT_0=R)
        bv_file_name = os.path.join(VIS_OUT_DIR, scene_name+'_bv.png')
        cv2.imwrite(bv_file_name, bird_view)

if __name__ == "__main__":
    
    f_scene = glob.glob(os.path.join(SCENE_DIR, '*.bin'))
    azim_list = []
    ratio_list = []
    trans_x_list = []
    trans_y_list = []
    
    count = 0
    for f_s in f_scene:
        
        ## parse file
        s_set = f_s.split('/')
        scene_name = s_set[-1][:-4]
        meta_file_name = os.path.join(META_OUT_DIR, scene_name+'.pkl')

        with open(meta_file_name, 'rb') as f:
            meta_info = pickle.load(f)

        azim_list.append(meta_info['azimuth'])
        ratio_list.append(meta_info['sample_ratio'])
        trans_x_list.append(meta_info['translation'][0, 0])
        trans_y_list.append(meta_info['translation'][0, 1])
        count += 1
        print('loading {}/{} scenes'.format(count, len(f_scene)))
    
    fig = plt.figure()
    plt.hist(azim_list) 
    fig.suptitle('Rotation', fontsize=20)
    plt.xlabel('azimuth/deg', fontsize=18)
    plt.ylabel('count', fontsize=18)
    plt.savefig('hist_azim_{}.png'.format(MODE))
    plt.clf()

    fig = plt.figure()
    plt.hist(np.log(ratio_list)/np.log(2))
    fig.suptitle('Down Sampling Ratio', fontsize=20)
    plt.xlabel('log(ratio)', fontsize=18)
    plt.ylabel('count', fontsize=18)
    plt.savefig('hist_r_{}.png'.format(MODE))
    plt.clf()

    fig = plt.figure()
    plt.hist(trans_x_list, bins=100)
    fig.suptitle('Translation X', fontsize=20)
    plt.xlabel('x', fontsize=18)
    plt.ylabel('count', fontsize=18)
    plt.savefig('hist_tx_{}.png'.format(MODE))
    plt.clf()
    
    fig = plt.figure()
    plt.hist(trans_y_list, bins=100)
    fig.suptitle('Translation Y', fontsize=20)
    plt.xlabel('y', fontsize=18)
    plt.ylabel('count', fontsize=18)
    plt.savefig('hist_ty_{}.png'.format(MODE))
    plt.clf()
