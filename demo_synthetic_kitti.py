import os
import sys
import numpy as np
import glob
from plyfile import PlyData, PlyElement
from util_pcds import *

kitti_data_dir = '../../data/KITTI/training/velodyne'
shapenet_data_dir = '../../PCDs/02958343'
obj_file = '5801f9eb726b56448b9c28e7b121fdbc_4096.ply'
scene_file = '001202.bin'
label_data_dir = '../../data/KITTI/training/label_2/'
label_file = '001202.txt'
azim = 0

#f_lidar = glob.glob(os.path.join(kitti_data_dir, 'velodyne', '*.bin'))
#shapenet_obj_dir = glob.glob(os.path.join(shapenet_data_dir, '*'))

#print(f_lidar[:10])
#print(shapenet_obj_dir[:10])

#for _, _, files in os.walk(shapenet_data_dir):
#    print(files[:10])

## scene loading up here
scene_name = os.path.join(kitti_data_dir, scene_file)
scene_pc = np.fromfile(scene_name, dtype=np.float32).reshape(-1, 4)
print(scene_pc.shape)
scene_pc_pcd = np.concatenate((scene_pc[:, :3], np.zeros_like(scene_pc[:, :3])), axis=1)
scene_pc_pcd[:, 5] = 255 ## assign green for background


obj_name = os.path.join(shapenet_data_dir, obj_file)
ply_data = PlyData.read(obj_name)
#print(ply_data.elements[0].data['x'].shape)
#print(min(ply_data.elements[0].data['x']))
#print(max(ply_data.elements[0].data['x']))
#print(min(ply_data.elements[0].data['y']))
#print(max(ply_data.elements[0].data['y']))
#print(min(ply_data.elements[0].data['z']))
#print(max(ply_data.elements[0].data['z']))
pc_coord = []
pc_coord.append(ply_data.elements[0].data['x'])
pc_coord.append(ply_data.elements[0].data['z'])
pc_coord.append(ply_data.elements[0].data['y'])
pc_coord = np.squeeze(np.dstack(pc_coord))
pc_coord *= 5 ## scale to real car 

## car in plane rotation happens here
## coordinates in order of [x, y, z] ??
R = azim2rot_deg(azim)
new_pc_coord = np.transpose(np.matmul(R, np.transpose(pc_coord)))
pc_coord = new_pc_coord

## car in plane translation happens here
scene_x_var = np.sqrt(np.var(scene_pc[:, 0]))
scene_y_var = np.sqrt(np.var(scene_pc[:, 1]))
t_obj = np.zeros((1,3), dtype=np.float32)
t_obj[0, 0] = np.random.uniform(scene_x_var, scene_x_var)
t_obj[0, 1] = np.random.uniform(scene_y_var, scene_y_var)
t_obj[0, 0] = 0 
t_obj[0, 1] = 3 
pc_coord = pc_coord + np.tile(t_obj, [pc_coord.shape[0], 1])

## put car on ground, -3 along axix 0
pc_coord[:, 2] -= 1

pc_coord_pcd = np.concatenate((pc_coord, np.zeros_like(pc_coord)), axis=1)
pc_coord_pcd[:, 4] = 255 ## assign red for car
pc_coord = np.concatenate((pc_coord, np.zeros((pc_coord.shape[0], 1))), axis=1)
print(pc_coord.shape)

## create new pc file
vertex = np.concatenate((pc_coord_pcd, scene_pc_pcd), axis=0)

pcd_from_array('demo.pcd', vertex)

label_name = os.path.join(label_data_dir, label_file)
with open(label_name, 'r') as f:
    labels = f.readlines()
print(labels)
#print(np.max(pc_coord[:,0])-np.min(pc_coord[:,0]))
#print(np.max(pc_coord[:,1])-np.min(pc_coord[:,1]))
#print(np.max(pc_coord[:,2])-np.min(pc_coord[:,2]))
#print(np.mean(pc_coord[:,2]))
#print(np.mean(pc_coord[:,1]))
#print(np.mean(pc_coord[:,0]))
print(np.mean(scene_pc[:,2]))
print(np.mean(scene_pc[:,1]))
print(np.mean(scene_pc[:,0]))
print(np.mean(scene_pc[:,2]))
print(np.mean(scene_pc[:,1]))
print(np.mean(scene_pc[:,0]))

pc_coord.tofile('debug.bin')
