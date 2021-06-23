import numpy as np
import glob
import cv2
import parseTrackletXML as pt_XML

def load_tracklet( xml_path , num_frames ):
    """ extract tracklet's 3d box points and type """

    # read info from xml file
    tracklets = pt_XML.parseXML( xml_path )
    f_tracklet = {}
    f_type = {}
    # refered to parseTrackletXML.py's example function
    # loop over tracklets
    for tracklet in tracklets:

        # this part is inspired by kitti object development kit matlab code: computeBox3D
        h, w, l = tracklet.size
        trackletBox = np.array([  # in velodyne coordinates around zero point and without orientation yet\
                [-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2], \
                [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2], \
                [0.0, 0.0, 0.0, 0.0, h, h, h, h]])
        # loop over all data in tracklet
        for translation, rotation, state, occlusion, truncation, amtOcclusion, amtBorders, absoluteFrameNumber in tracklet:
            # determine if object is in the image; otherwise continue
            if truncation not in (pt_XML.TRUNC_IN_IMAGE, pt_XML.TRUNC_TRUNCATED):
                continue

            # re-create 3D bounding box in velodyne coordinate system
            yaw = rotation[2]  # other rotations are 0 in all xml files I checked
            assert np.abs(rotation[:2]).sum() == 0, 'object rotations other than yaw given!'
            rotMat = np.array([ \
                    [np.cos(yaw), -np.sin(yaw), 0.0], \
                    [np.sin(yaw), np.cos(yaw), 0.0], \
                    [0.0, 0.0, 1.0]])

            cornerPosInVelo = np.dot(rotMat, trackletBox) + np.tile(translation, (8, 1)).T

            if absoluteFrameNumber in f_tracklet:
                f_tracklet[absoluteFrameNumber] += [cornerPosInVelo]
                f_type[absoluteFrameNumber] += [tracklet.objectType]
            else:
                f_tracklet[absoluteFrameNumber] = [cornerPosInVelo]
                f_type[absoluteFrameNumber] = [tracklet.objectType]

    # fill none in non object frame
    if num_frames is not None:
        for i in range(num_frames):
            if i not in f_tracklet:
                f_tracklet[i] = None
                f_type[i] = None

    return f_tracklet, f_type

def calib_velo2cam( v2c_file ):
    """
    get Rotation(R : 3x3), Translation(T : 3x1) matrix info
    using R,T matrix, we can convert velodyne coordinates to camera coordinates
    """
    if v2c_file is None:
        raise NameError("calib_velo_to_cam file isn't loaded.")

    for line in v2c_file:
        (key, val) = line.split(':', 1)
        if key == 'R':
            R = np.fromstring(val, sep=' ')
            R = R.reshape(3, 3)
        if key == 'T':
            T = np.fromstring(val, sep=' ')
            T = T.reshape(3, 1)
    return R, T

def calib_cam2cam( c2c_file ):
    """
    If your image is 'rectified image' :
         get only Projection(P : 3x4) matrix is enough
    but if your image is 'distorted image'(not rectified image) :
        you need undistortion step using distortion coefficients(5 : D)

    In this code, only P matrix info is used for rectified image
    """
    if c2c_file is None:
        raise NameError("calib_velo_to_cam file isn't loaded.")

    mode = '02'

    for line in c2c_file:
        (key, val) = line.split(':', 1)
        if key == ('P_rect_' + mode):
            P_ = np.fromstring(val, sep=' ')
            P_ = P_.reshape(3, 4)
            # erase 4th column ([0,0,0])
            P_ = P_[:3, :3]
    return P_

def point_matrix(  points ):
    """ extract points corresponding to FOV setting """

    __x = points[:, 0]
    __y = points[:, 1]
    __z = points[:, 2]

    # Stack arrays in sequence horizontally
    xyz_ = np.hstack(( __x[:, None], __y[:, None], __z[:, None]))
    xyz_ = xyz_.T

    # stack (1,n) arrays filled with the number 1
    one_mat = np.full((1, xyz_.shape[1]), 1)
    xyz_ = np.concatenate((xyz_, one_mat), axis=0)


    return xyz_


def velo_2_img_projection( points , v2c_file , c2c_file):
    """ convert velodyne coordinates to camera image coordinates """

    # rough velodyne azimuth range corresponding to camera horizontal fov

    # R_vc = Rotation matrix ( velodyne -> camera )
    # T_vc = Translation matrix ( velodyne -> camera )
    R_vc, T_vc = calib_velo2cam( v2c_file )

    # P_ = Projection matrix ( camera coordinates 3d points -> image plane 2d points )
    P_ = calib_cam2cam( c2c_file )

    """
    xyz_v - 3D velodyne points corresponding to h, v FOV limit in the velodyne coordinates
    c_    - color value(HSV's Hue vaule) corresponding to distance(m)

                 [x_1 , x_2 , .. ]
    xyz_v =  [y_1 , y_2 , .. ]
                 [z_1 , z_2 , .. ]
                 [ 1  ,  1  , .. ]
    """
    xyz_v  = point_matrix(points)

    """
    RT_ - rotation matrix & translation matrix
            ( velodyne coordinates -> camera coordinates )

                [r_11 , r_12 , r_13 , t_x ]
    RT_  =  [r_21 , r_22 , r_23 , t_y ]
                [r_31 , r_32 , r_33 , t_z ]
    """
    RT_ = np.concatenate((R_vc, T_vc), axis=1)

    # convert velodyne coordinates(X_v, Y_v, Z_v) to camera coordinates(X_c, Y_c, Z_c)
    for i in range(xyz_v.shape[1]):
        xyz_v[:3, i] = np.matmul(RT_, xyz_v[:, i])

    """
    xyz_c - 3D velodyne points corresponding to h, v FOV in the camera coordinates
                 [x_1 , x_2 , .. ]
    xyz_c =  [y_1 , y_2 , .. ]
                 [z_1 , z_2 , .. ]
    """
    xyz_c = np.delete(xyz_v, 3, axis=0)

    # convert camera coordinates(X_c, Y_c, Z_c) image(pixel) coordinates(x,y)
    for i in range(xyz_c.shape[1]):
        xyz_c[:, i] = np.matmul(P_, xyz_c[:, i])

    """
    xy_i - 3D velodyne points corresponding to h, v FOV in the image(pixel) coordinates before scale adjustment
    ans  - 3D velodyne points corresponding to h, v FOV in the image(pixel) coordinates
                 [s_1*x_1 , s_2*x_2 , .. ]
    xy_i =   [s_1*y_1 , s_2*y_2 , .. ]        ans =   [x_1 , x_2 , .. ]
                 [  s_1   ,   s_2   , .. ]                [y_1 , y_2 , .. ]
    """
    xy_i = xyz_c[::] / xyz_c[::][2]
    ans = np.delete(xy_i, 2, axis=0)

    return ans , xyz_c[::][2]

def load_velo2cam( v2c_path ):
    """ load Velodyne to Camera calibration info file """
    with open( v2c_path, "r") as f:
        file = f.readlines()
    return file

def load_cam2cam( c2c_path ):
    """ load Camera to Camera calibration info file """
    with open( c2c_path, "r") as f:
        file = f.readlines()
    return file