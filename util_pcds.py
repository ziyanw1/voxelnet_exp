import os
import sys
import numpy as np

def pcd_from_array(pcd_file_name, array):
    with open(pcd_file_name, 'w') as of:
        # write header
        of.write('# .PCD v.7 - Point Cloud Data file format\r\n')
        of.write('VERSION .7\r\n')
        of.write('FIELDS x y z rgb\r\n')
        of.write('SIZE 4 4 4 4\r\n')
        of.write('TYPE F F F F\r\n')
        of.write('COUNT 1 1 1 1\r\n')
        of.write('WIDTH {}\r\n'.format(len(array)))
        of.write('HEIGHT 1\r\n')
        of.write('VIEWPOINT 0 0 0 1 0 0 0\r\n')
        of.write('POINTS {}\r\n'.format(len(array)))
        of.write('DATA ascii\r\n')
        for coor in array:
            rgb = np.uint8(coor[3:])
            line_out = '{:.3f} {:.3f} {:.3f} {}\r\n'.format(coor[0], coor[1], coor[2], rgb[2]<<16|rgb[1]<<8|rgb[0])
            of.write(line_out)

def azim2rot_deg(azim_deg):
    azim = np.deg2rad(azim_deg)
    
    return azim2rot(azim)

def azim2rot(azim):
    R = np.zeros((3,3), dtype=np.float32)
    R[0, 0] = np.cos(azim)
    R[0, 1] = -np.sin(azim)
    R[1, 0] = np.sin(azim)
    R[1, 1] = np.cos(azim)
    R[2, 2] = 1

    return R

#CAM = 2
#def load_calib(calib_dir):
#    # P2 * R0_rect * Tr_velo_to_cam * y
#    lines = open(calib_dir).readlines()
#    lines = [ line.split()[1:] for line in lines ][:-1]
#    #
#    P = np.array(lines[CAM]).reshape(3,4)
#    P = np.concatenate( (  P, np.array( [[0,0,0,0]] )  ), 0  )
#    #
#    Tr_velo_to_cam = np.array(lines[5]).reshape(3,4)
#    Tr_velo_to_cam = np.concatenate(  [ Tr_velo_to_cam, np.array([0,0,0,1]).reshape(1,4)  ]  , 0     )
#    #
#    R_cam_to_rect = np.eye(4)
#    R_cam_to_rect[:3,:3] = np.array(lines[4][:9]).reshape(3,3)
#    #
#    P = P.astype('float32')
#    Tr_velo_to_cam = Tr_velo_to_cam.astype('float32')
#    R_cam_to_rect = R_cam_to_rect.astype('float32')
#    return P, Tr_velo_to_cam, R_cam_to_rect
#
#def lidar_to_bird_view_img(lidar, factor=1):
#    # Input:
#    #   lidar: (N', 4)
#    # Output:
#    #   birdview: (w, l, 3)
#    birdview = np.zeros(
#        (cfg.INPUT_HEIGHT * factor, cfg.INPUT_WIDTH * factor, 1))
#    for point in lidar:
#        x, y = point[0:2]
#        if cfg.X_MIN < x < cfg.X_MAX and cfg.Y_MIN < y < cfg.Y_MAX:
#            x, y = int((x - cfg.X_MIN) / cfg.VOXEL_X_SIZE *
#                       factor), int((y - cfg.Y_MIN) / cfg.VOXEL_Y_SIZE * factor)
#            birdview[y, x] += 1
#    birdview = birdview - np.min(birdview)
#    divisor = np.max(birdview) - np.min(birdview)
#    # TODO: adjust this factor
#    birdview = np.clip((birdview / divisor * 255) *
#                       5 * factor, a_min=0, a_max=255)
#    birdview = np.tile(birdview, 3).astype(np.uint8)
#
#    return birdview
