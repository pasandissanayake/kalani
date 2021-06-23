import os
import time
import matplotlib.pyplot as plt
import numpy as np
from numpy.core.numeric import indices
from extraction_utils import *
import multiprocessing as mp

param_list = []
imgfov_pc_cam2 = None
imgfov_pc_pixel = None

def left_right( pixel_corr ):

    polygon = np.array(pixel_corr)
    left = np.min(polygon, axis=0)
    right = np.max(polygon, axis=0)

    return left , right

def index_extraction( i ):

    depth = imgfov_pc_cam2[2, i]

    for i_param in param_list :

        f1_l , f1_r = i_param['f1_l'] , i_param['f1_r']
        f2_l , f2_r = i_param['f2_l'] , i_param['f2_r']
        f3_l , f3_r = i_param['f3_l'] , i_param['f3_r']
        f4_l , f4_r = i_param['f4_l'] , i_param['f4_r']
        min_depth , max_depth = i_param['min_depth'] , i_param['max_depth']

        if (  ( (( f1_l[0] < imgfov_pc_pixel[0,i] < f1_r[0]) and ( f1_l[1] < imgfov_pc_pixel[1,i] < f1_r[1]  )) or \
            (( f2_l[0] < imgfov_pc_pixel[0,i] < f2_r[0]) and ( f2_l[1] < imgfov_pc_pixel[1,i] < f2_r[1]  )) or \
            (( f3_l[0] < imgfov_pc_pixel[0,i] < f3_r[0]) and ( f3_l[1] < imgfov_pc_pixel[1,i] < f3_r[1]  )) or \
            (( f4_l[0] < imgfov_pc_pixel[0,i] < f4_r[0]) and ( f4_l[1] < imgfov_pc_pixel[1,i] < f4_r[1]  ))) and \
            (  min_depth < depth < max_depth ) ):

            return i  

def draw_boxes( calib , fig , objects ):
    # Projection matrix
    proj_cam2_2_velo = project_cam2_to_velo(calib)

    # Draw objects on lidar
    for obj in objects:
        if obj.type == 'DontCare':
            continue
        # Project boxes from camera to lidar coordinate
        boxes3d_pts = project_camera_to_lidar(obj.in_camera_coordinate(), proj_cam2_2_velo)

        # Draw boxes
        fig = draw_gt_boxes3d(boxes3d_pts, fig=fig)
    
    return fig

def lidar_extraction(pts_velo, calib, img_width, img_height , tracklets , depth_box ):

    start_time = time.time()

    # projection matrix (project from velo2cam2)
    proj_velo2cam2 = project_velo_to_cam2(calib)
    # apply projection
    pts_2d  , depth = project_to_image(pts_velo.transpose(), proj_velo2cam2)

    # Filter lidar points to be within image FOV
    inds = np.where((pts_2d[0, :] < img_width) & (pts_2d[0, :] >= 0) &
                    (pts_2d[1, :] < img_height) & (pts_2d[1, :] >= 0) &
                    ( pts_velo[:, 0] < 50 ) & (0 < pts_velo[:,0] )
                    )[0]
    
    global imgfov_pc_pixel ,imgfov_pc_cam2
    # Filter out pixels points
    imgfov_pc_pixel = pts_2d[:, inds]

    # Retrieve depth from lidar
    imgfov_pc_velo = pts_velo[inds, :]
    imgfov_pc_velo = np.hstack((imgfov_pc_velo, np.ones((imgfov_pc_velo.shape[0], 1))))
    imgfov_pc_cam2 = proj_velo2cam2 @ imgfov_pc_velo.transpose()

    cmap = plt.cm.get_cmap('hsv', 256)
    cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255
    index_list = []

    # projection matrix
    P_rect2cam2 = calib['P2'].reshape((3, 4))

    global param_list
    param_list = []
    for idx , (bbox , box_depth) in enumerate( zip( tracklets , depth_box  ) ):
    
        #for idx , obj in enumerate( objects):
        #    if obj.type == 'DontCare' :
        #        continue
        #    bbox  , box_depth = map_box_to_image(obj, P_rect2cam2)
        #    bbox = bbox.transpose()

        f1_l , f1_r = left_right( [ bbox[5] , bbox[6] , bbox[2] , bbox[1] ]   )
        f2_l , f2_r = left_right( [ bbox[7] , bbox[6] , bbox[2] , bbox[3] ]   )
        f3_l , f3_r = left_right( [ bbox[7] , bbox[4] , bbox[3] , bbox[0] ]   )
        f4_l , f4_r = left_right( [ bbox[4] , bbox[5] , bbox[0] , bbox[1] ]   )

        min_depth  , max_depth = left_right( [ box_depth[3] , box_depth[0] , box_depth[1] , box_depth[2]] )

        params ={
            'f1_l':f1_l , 'f1_r':f1_r , 'f2_l':f2_l , 'f2_r':f2_r , 'f3_l':f3_l , 'f3_r':f3_r , 
            'f4_l':f4_l , 'f4_r':f4_r , 'min_depth':min_depth , 'max_depth':max_depth
        }
        param_list.append( params )

    for i in range(imgfov_pc_pixel.shape[1]):
        depth = imgfov_pc_cam2[2, i]

        for i_param in param_list :

            f1_l , f1_r = i_param['f1_l'] , i_param['f1_r'] 
            f2_l , f2_r = i_param['f2_l'] , i_param['f2_r'] 
            f3_l , f3_r = i_param['f3_l'] , i_param['f3_r'] 
            f4_l , f4_r = i_param['f4_l'] , i_param['f4_r'] 
            min_depth , max_depth = i_param['min_depth'] , i_param['max_depth'] 

            if (  ( (( f1_l[0] < imgfov_pc_pixel[0,i] < f1_r[0]) and ( f1_l[1] < imgfov_pc_pixel[1,i] < f1_r[1]  )) or \
                (( f2_l[0] < imgfov_pc_pixel[0,i] < f2_r[0]) and ( f2_l[1] < imgfov_pc_pixel[1,i] < f2_r[1]  )) or \
                (( f3_l[0] < imgfov_pc_pixel[0,i] < f3_r[0]) and ( f3_l[1] < imgfov_pc_pixel[1,i] < f3_r[1]  )) or \
                (( f4_l[0] < imgfov_pc_pixel[0,i] < f4_r[0]) and ( f4_l[1] < imgfov_pc_pixel[1,i] < f4_r[1]  ))) and \
                    (  min_depth < depth < max_depth ) ):

                index_list.append(i)
 
    #pool = mp.Pool(mp.cpu_count())
    #result = pool.map( index_extraction , np.arange( imgfov_pc_pixel.shape[1] )  )
    #results = [pool.apply_async( index_extraction , args=(x,)) for x in np.arange( imgfov_pc_pixel.shape[1] ) ]
    #output = [p.get() for p in results]
    #result = filter(None.__ne__, result )
    #pool.close()
    #pool.join()

    index_list = np.array(list( index_list ))
    
    if(len(index_list)!=0 ):
        A = np.arange(len(pts_velo))
        B = inds[ index_list ]
        A = A[~np.isin(A,B)]
        lidar_cloud = pts_velo[ A , :][:,:3]

        end_time = time.time()

        print("Total Execution Time : ",end_time - start_time)

        #return lidar_cloud
        #return index array
        return pts_velo[ A , :][:,:3] , A

    else :

        return pts_velo[:,:][:,:3] , np.arange(len(pts_velo))


