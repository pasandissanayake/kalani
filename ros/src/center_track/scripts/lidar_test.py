from extraction_utils import Box3D , read_calib_file , map_box_to_image , draw_projected_box3d , draw_lidar
import _init_paths
import mayavi.mlab as mlab
import pickle

with open('test.pkl','rb') as f:
    x = pickle.load(f)

draw_lidar(x )
mlab.show()