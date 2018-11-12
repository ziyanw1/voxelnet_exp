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

np.random.seed(0)

DEBUG = False
VIS = True
GENERATE_PLY = False
POOL_NUM = 8
MODE = 'validation'
from config import cfg
#if MODE == 'training':
#    from config import cfg
#else:
#    from config_val import cfg
sample_ratios = [2, 4, 8]
DATA_DIR = os.path.join('../../data/KITTI', MODE)
TOY_DIR = os.path.join('../../data/toyset', MODE)
OBJ_DIR = '/home/ziyanw1/PCDs/02958343'
SCENE_DIR = os.path.join(DATA_DIR, 'velodyne')
OBJ_OUT_DIR = os.path.join(TOY_DIR, 'template')
SCENE_OUT_DIR = os.path.join(TOY_DIR, 'syn_scene')
LABEL_OUT_DIR = os.path.join(TOY_DIR, 'syn_label')
VIS_OUT_DIR = os.path.join(TOY_DIR, 'vis')
META_OUT_DIR = os.path.join(TOY_DIR, 'meta')

if not os.path.exists(TOY_DIR):
    os.mkdir(TOY_DIR)

if not os.path.exists(OBJ_OUT_DIR):
    os.mkdir(OBJ_OUT_DIR)

if not os.path.exists(SCENE_OUT_DIR):
    os.mkdir(SCENE_OUT_DIR)

if not os.path.exists(LABEL_OUT_DIR):
    os.mkdir(LABEL_OUT_DIR)

if not os.path.exists(VIS_OUT_DIR):
    os.mkdir(VIS_OUT_DIR)

if not os.path.exists(META_OUT_DIR):
    os.mkdir(META_OUT_DIR)

X = np.linspace(15, 55, 11)
Y = np.linspace(-20, 20, 11)
grid_x, grid_y = np.meshgrid(X, Y)
#print(grid_x.shape)
#print(grid_y.shape)
#sys.exit()
idx_map = np.random.randint(0, 2, grid_x.shape)
if MODE == 'validation':
    idx_map = 1 - idx_map

idx_list = np.argwhere(idx_map == 1)
#print(X.shape)
#print(Y.shape)
#print(np.max(idx_list, 0))
#print(idx_list[:20])
#sys.exit()

def create_syn_scene_obj(scene_file_name, obj_file_name, scene_idx, generate_ply=GENERATE_PLY, vis=VIS):
    
    scene_down_sample = 2
    ## parse file
    s_set = scene_file_name.split('/')
    scene_name_ori = s_set[-1][:-4]
    scene_name = s_set[-1][:-4] + '_{}'.format(scene_idx)
    s_set = obj_file_name.split('/')
    obj_name = s_set[-1][:-4]
    
    ## load scene here
    #scene_name = os.path.join(kitti_data_dir, scene_file)
    scene_pc = np.fromfile(scene_file_name, dtype=np.float32).reshape(-1, 4)
    idx = np.argwhere(scene_pc[:,2] >= -3)
    idx = np.squeeze(idx)
    #print(scene_pc[:10])
    scene_pc = scene_pc[idx, :]
    scene_pc = scene_pc[::scene_down_sample, :]
    #print(scene_pc.shape)
    #print(idx[:10])
    #print(scene_pc[:10])
    #sys.exit()
    #print(np.max(scene_pc[:,0]))
    #print(np.min(scene_pc[:,0]))
    #print(np.max(scene_pc[:,1]))
    #print(np.min(scene_pc[:,1]))
    #print(np.max(scene_pc[:,2]))
    #print(np.min(scene_pc[:,2]))
    #print(np.mean(scene_pc[:,2]))
    #scene_pc_pcd = np.concatenate((scene_pc[:, :3], np.zeros_like(scene_pc[:, :3])), axis=1)
    #scene_pc_pcd[:, 5] = 255 ## assign green for background
    #sys.exit()
    
    ## sample azim here
    azim = np.random.choice([0, 90, 180, 270])
    azim += np.random.uniform(-5, 5)
    R = azim2rot_deg(azim)
    
    ## sample translation here
    scene_x_var = np.sqrt(np.var(scene_pc[:, 0]))
    scene_y_var = np.sqrt(np.var(scene_pc[:, 1]))
    t_obj = np.zeros((1,3), dtype=np.float32)
    
    offset_idx = idx_list[scene_idx % idx_list.shape[0]]
    t_obj[0, 0] = X[offset_idx[1]] + np.random.uniform(-1,1) 
    t_obj[0, 1] = Y[offset_idx[0]] + np.random.uniform(-1,1)
    t_obj[0, 2] -= 1    ## make the car on the ground
    
    ## load car here
    ply_data = PlyData.read(obj_file_name)
    
    pc_coord = []
    pc_coord.append(ply_data.elements[0].data['x'])
    pc_coord.append(ply_data.elements[0].data['z'])
    pc_coord.append(ply_data.elements[0].data['y'])
    pc_coord = np.squeeze(np.dstack(pc_coord))
    template_coord = pc_coord.copy()
    #template_coord_pcd = np.concatenate((template_coord, np.zeros_like(template_coord)), axis=1) 
    template_coord = np.concatenate((template_coord, np.zeros((pc_coord.shape[0], 1))), axis=1)
    
    pc_coord *= 4.5 ## scale to real car 
    #pc_coord[:, 2] = pc_coord[:, 2] - 1
    
    ## for 3D bbox
    l = np.max(pc_coord[:, 0]) - np.min(pc_coord[:, 0])
    w = np.max(pc_coord[:, 1]) - np.min(pc_coord[:, 1])
    h = np.max(pc_coord[:, 2]) - np.min(pc_coord[:, 2])
    #z = (np.max(pc_coord[:, 2])+np.min(pc_coord[:,2]))/2
    #print('z:{}'.format(z))
    #print('max z:{}'.format(np.max(pc_coord[:, 2])))
    #print('min z:{}'.format(np.min(pc_coord[:, 2])))
    #print('h:{}'.format(h))
    #print('w:{}'.format(w))
    #print('l:{}'.format(l))
    #sys.exit()
    
    ## car in plane rotation happens here
    ## coordinates in order of [x, y, z] ??
    new_pc_coord = np.transpose(np.matmul(R, np.transpose(pc_coord)))
    pc_coord = new_pc_coord

    ## translation happens here
    pc_coord = pc_coord + np.tile(t_obj, [pc_coord.shape[0], 1])
    
    ## down sampleing happens here
    sample_ratio = np.random.choice(sample_ratios)
    pc_coord = pc_coord[::sample_ratio, ...] 
    
    #pc_coord_pcd = np.concatenate((pc_coord, np.zeros_like(pc_coord)), axis=1)
    #pc_coord_pcd[:, 4] = 255 ## assign red for car
    pc_coord = np.concatenate((pc_coord, np.zeros((pc_coord.shape[0], 1))), axis=1)


    ## create new pc file
    #vertex = np.concatenate((pc_coord_pcd, scene_pc_pcd), axis=0)

    ## dump template model both .ply and .bin
    #obj_outfile_pcd = os.path.join(OBJ_OUT_DIR, scene_name+'.pcd')
    #pcd_from_array(obj_outfile_pcd, template_coord_pcd)
    obj_outfile_bin = os.path.join(OBJ_OUT_DIR, scene_name+'.bin')
    template_coord.astype(np.float32).tofile(obj_outfile_bin)
    #if generate_ply:
    #    command = './pcl_pcd2ply {} {}'.format(obj_outfile_pcd, obj_outfile_pcd[:-4]+'.ply')
    #    os.system(command)

    ## dump scene with template both p.ly and .bin
    #scene_outfile_pcd = os.path.join(SCENE_OUT_DIR, scene_name+'.pcd')
    #pcd_from_array(scene_outfile_pcd, vertex)
    scene_outfile_bin = os.path.join(SCENE_OUT_DIR, scene_name+'.bin')
    scene_gen_pc = np.concatenate((pc_coord, scene_pc), axis=0)
    scene_gen_pc.astype(np.float32).tofile(scene_outfile_bin)
    #if generate_ply:
    #    command = './pcl_pcd2ply {} {}'.format(scene_outfile_pcd, scene_outfile_pcd[:-4]+'.ply')
    #    os.system(command)

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
    camera_boxes = lidar_to_camera_box(gt_bboxes)
    h = camera_boxes[0, 3]
    w = camera_boxes[0, 4]
    l = camera_boxes[0, 5]
    x = camera_boxes[0, 0]
    y = camera_boxes[0, 1]
    z = camera_boxes[0, 2]
    ry = camera_boxes[0, 6]
    label_line = 'Car 0.0 0.0 0.0 10.0 20.0 30.0 50.0 {:.02f} {:.02f} {:.02f} {:.02f} {:.02f} {:.02f} {:.02f}\n'.format(
        h, w, l, x, y, z, ry) 
    #print(label_line)
    #sys.exit()
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
        P, Tr, R = load_calib( os.path.join( cfg.CALIB_DIR, scene_name_ori + '.txt' ) )
        bird_view = lidar_to_bird_view_img(scene_gen_pc, factor=cfg.BV_LOG_FACTOR)
        bird_view = draw_lidar_box3d_on_birdview(bird_view, batch_gt_boxes3d[0], 1, batch_gt_boxes3d[0], factor=cfg.BV_LOG_FACTOR, P2=P, T_VELO_2_CAM=Tr, R_RECT_0=R)
        bv_file_name = os.path.join(VIS_OUT_DIR, scene_name+'_bv.png')
        cv2.imwrite(bv_file_name, bird_view)

if __name__ == "__main__":
    
    if DEBUG:
        ## for debug
        scene_list = ['001201.bin']
        obj_list = ['5801f9eb726b56448b9c28e7b121fdbc_4096.ply']

        scene_list = [os.path.join(DATA_DIR, 'velodyne', s) for s in scene_list]
        obj_list = [os.path.join(OBJ_DIR, o) for o in obj_list]

        for s in scene_list:
            o = np.random.choice(obj_list)
            create_syn_scene_obj(s, o, 0, True, True)

    else:
        ##pool = Pool(POOL_NUM)
        #f_scene = glob.glob(os.path.join(SCENE_DIR, '*.bin'))
        #f_obj = glob.glob(os.path.join(OBJ_DIR, '*_4096.ply'))
        #
        #if MODE is 'training':
        #    f_obj_use = np.random.choice(f_obj[:int(0.7*len(f_obj))], size=len(f_scene))
        #else:
        #    f_obj_use = np.random.choice(f_obj[int(0.7*len(f_obj)):], size=len(f_scene))
        #    

        ##pool.imap(create_syn_scene_obj, f_scene, f_obj_use)
        ##pool.close()
        #for idx, (s_path, o_path) in enumerate(zip(f_scene, f_obj_use)):
        #    create_syn_scene_obj(s_path, o_path)
        #    print('------- Scene {} is generated -------'.format(idx))
        SCENE_NUM = 1000
        scene_list = ['001201.bin']
        obj_list = ['5801f9eb726b56448b9c28e7b121fdbc_4096.ply']

        s = os.path.join('../../data/KITTI/training', 'velodyne', scene_list[0])
        o = os.path.join(OBJ_DIR, obj_list[0])

        for s_i in range(SCENE_NUM):
            print('generating scene {}'.format(s_i))
            create_syn_scene_obj(s, o, s_i, True, True)
