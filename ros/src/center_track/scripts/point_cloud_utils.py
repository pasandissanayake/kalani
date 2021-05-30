import numpy as np
from sensor_msgs.msg import PointCloud2, PointField
import numpy as np
from matplotlib.cm import get_cmap

import rospy
import sensor_msgs.msg as sensor_msgs
import std_msgs.msg as std_msgs

# prefix to the names of dummy fields we add to get byte alignment correct. this needs to not
# clash with any actual field names

DUMMY_FIELD_PREFIX = '__'
 
# mappings between PointField types and numpy types
type_mappings = [(PointField.INT8, np.dtype('int8')), (PointField.UINT8, np.dtype('uint8')), (PointField.INT16, np.dtype('int16')),
                  (PointField.UINT16, np.dtype('uint16')), (PointField.INT32, np.dtype('int32')), (PointField.UINT32, np.dtype('uint32')),
                  (PointField.FLOAT32, np.dtype('float32')), (PointField.FLOAT64, np.dtype('float64'))]
pftype_to_nptype = dict(type_mappings)
nptype_to_pftype = dict((nptype, pftype) for pftype, nptype in type_mappings)
 
# sizes (in bytes) of PointField types
pftype_sizes = {PointField.INT8: 1, PointField.UINT8: 1, PointField.INT16: 2, PointField.UINT16: 2,
                 PointField.INT32: 4, PointField.UINT32: 4, PointField.FLOAT32: 4, PointField.FLOAT64: 8}
 

def fields_to_dtype(fields, point_step):
    '''Convert a list of PointFields to a numpy record datatype.
    '''
    offset = 0
    np_dtype_list = []
    for f in fields:
        while offset < f.offset:
             # might be extra padding between fields
             np_dtype_list.append(('%s%d' % (DUMMY_FIELD_PREFIX, offset), np.uint8))
             offset += 1

        dtype = pftype_to_nptype[f.datatype]
        if f.count != 1:
            dtype = np.dtype((dtype, f.count))
 
        np_dtype_list.append((f.name, dtype))
        offset += pftype_sizes[f.datatype] * f.count
 
    # might be extra padding between points
    while offset < point_step:
        np_dtype_list.append(('%s%d' % (DUMMY_FIELD_PREFIX, offset), np.uint8))
        offset += 1
         
    return np_dtype_list
 

def dtype_to_fields(dtype):
     '''Convert a numpy record datatype into a list of PointFields.
     '''
     fields = []
     for field_name in dtype.names:
         np_field_type, field_offset = dtype.fields[field_name]
         pf = PointField()
         pf.name = field_name
         if np_field_type.subdtype:
             item_dtype, shape = np_field_type.subdtype
             pf.count = np.prod(shape)
             np_field_type = item_dtype
         else:
             pf.count = 1
 
         pf.datatype = nptype_to_pftype[np_field_type]
         pf.offset = field_offset
         fields.append(pf)
     return fields
 

def pointcloud2_to_array(cloud_msg, squeeze=True):

     ''' Converts a rospy PointCloud2 message to a numpy recordarray 
     
     Reshapes the returned array to have shape (height, width), even if the height is 1.
     The reason for using np.fromstring rather than struct.unpack is speed... especially
     for large point clouds, this will be <much> faster.
     '''
     # construct a numpy record type equivalent to the point type of this cloud
     dtype_list = fields_to_dtype(cloud_msg.fields, cloud_msg.point_step)
 
     # parse the cloud into an array
     cloud_arr = np.fromstring(cloud_msg.data, dtype_list)
 
     # remove the dummy fields that were added
     cloud_arr = cloud_arr[
         [fname for fname, _type in dtype_list if not (fname[:len(DUMMY_FIELD_PREFIX)] == DUMMY_FIELD_PREFIX)]]
     
     if squeeze and cloud_msg.height == 1:
         return np.reshape(cloud_arr, (cloud_msg.width,))
     else:
         return np.reshape(cloud_arr, (cloud_msg.height, cloud_msg.width))

def array_to_pointcloud2(cloud_arr, stamp=None, frame_id=None):
     '''Converts a numpy record array to a sensor_msgs.msg.PointCloud2.
     '''
     # make it 2d (even if height will be 1)
     cloud_arr = np.atleast_2d(cloud_arr)
 
     cloud_msg = PointCloud2()
 
     if stamp is not None:
         cloud_msg.header.stamp = stamp
     if frame_id is not None:
         cloud_msg.header.frame_id = frame_id
     cloud_msg.height = cloud_arr.shape[0]
     cloud_msg.width = cloud_arr.shape[1]
     cloud_msg.fields = dtype_to_fields(cloud_arr.dtype)
     cloud_msg.is_bigendian = False # assumption
     cloud_msg.point_step = cloud_arr.dtype.itemsize
     cloud_msg.row_step = cloud_msg.point_step*cloud_arr.shape[1]
     cloud_msg.is_dense = all([np.isfinite(cloud_arr[fname]).all() for fname in cloud_arr.dtype.names])
     cloud_msg.data = cloud_arr.tostring()
     return cloud_msg
 
def get_xyz_points(cloud_array, remove_nans=True, dtype=np.float):
     '''Pulls out x, y, and z columns from the cloud recordarray, and returns
         a 3xN matrix.
     '''
     # remove crap points
     if remove_nans:
         mask = np.isfinite(cloud_array['x']) & np.isfinite(cloud_array['y']) & np.isfinite(cloud_array['z'])
         cloud_array = cloud_array[mask]
     
     # pull out x, y, and z values
     points = np.zeros(cloud_array.shape + (3,), dtype=dtype)
     points[...,0] = cloud_array['x']
     points[...,1] = cloud_array['y']
     points[...,2] = cloud_array['z']
 
     return points
 
def pointcloud2_to_xyz_array(cloud_msg, remove_nans=True):
     return get_xyz_points(pointcloud2_to_array(cloud_msg), remove_nans=remove_nans)


def point_cloud(points, parent_frame):
    """ Creates a point cloud message.
    Args:
        points: Nx7 array of xyz positions (m) and rgba colors (0..1)
        parent_frame: frame in which the point cloud is defined
    Returns:
        sensor_msgs/PointCloud2 message
    """
    ros_dtype = sensor_msgs.PointField.FLOAT32
    dtype = np.float32
    itemsize = np.dtype(dtype).itemsize

    data = points.astype(dtype).tobytes()

    fields = [sensor_msgs.PointField(
        name=n, offset=i*itemsize, datatype=ros_dtype, count=1)
        for i, n in enumerate('xyz')]

    header = std_msgs.Header(frame_id=parent_frame, stamp=rospy.Time.now())

    return sensor_msgs.PointCloud2(
        header=header,
        height=1,
        width=points.shape[0],
        is_dense=False,
        is_bigendian=False,
        fields=fields,
        point_step=(itemsize * 7),
        row_step=(itemsize * 7 * points.shape[0]),
        data=data
    )

def xyzrgb_array_to_pointcloud2(points, colors, stamp=None, frame_id=None, seq=None):
    '''
    Create a sensor_msgs.PointCloud2 from an array
    of points.
    '''
    msg = PointCloud2()
    assert(points.shape == colors.shape)

    buf = []

    if stamp:
        msg.header.stamp = stamp
    if frame_id:
        msg.header.frame_id = frame_id
    if seq:
        msg.header.seq = seq
    if len(points.shape) == 3:
        msg.height = points.shape[1]
        msg.width = points.shape[0]
    else:
        N = len(points)
        xyzrgb = np.array(np.hstack([points, colors]), dtype=np.float32)
        msg.height = 1
        msg.width = N

    msg.fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1),
        PointField('r', 12, PointField.FLOAT32, 1),
        PointField('g', 16, PointField.FLOAT32, 1),
        PointField('b', 20, PointField.FLOAT32, 1)
    ]
    msg.is_bigendian = False
    msg.point_step = 24
    msg.row_step = msg.point_step * N
    msg.is_dense = True;
    msg.data = xyzrgb.tostring()

    return msg 