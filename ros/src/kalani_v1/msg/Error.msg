Header header

float64[] trans                     # estimated translation [east, north, up]
float64[] rot                       # estimated rotation [roll, pitch, yaw] in degrees
float64[] gt_trans                  # ground truth translation [east, north, up]
float64[] gt_rot                    # ground truth rotation [roll, pitch, yaw] in degrees
float64[] gnss_trans                # gnss translation [east, north, up]
float64[] magnetometer_rot          # magnetometer rotation [roll, pitch, yaw] in degrees
float64[] trans_error               # translational error in estimate [east, north, up]
float64[] rot_error                 # rotational error in estimate [roll, pitch, yaw] in degrees
float64[] gnss_error                # translational error in gnss [east, north, up]
float64[] magnetometer_error        # rotational error in magnetometer [roll, pitch, yaw] in degrees
float64[] trans_euclidean_error     # euclidean translational error in estimate
float64[] gnss_euclidean_error      # euclidean translational error in gnss
float64[] trans_abs_error           # absolute translational error in estimate [north, east, up]
float64[] gnss_abs_error            # absolute translational error in gnss [north, east, up]
float64[] magnetometer_abs_error    # absolute rotational error in magnetometer [roll, pitch, yaw] in degrees
float64[] trans_error_upperbound    # estimated translational upperbounds (3-sigma) [east, north, up]
float64[] trans_error_lowerbound    # estimated translational lowerbounds (3-sigma) [east, north, up]
float64[] rot_error_upperbound      # estimated rotational upperbounds (3-sigma) [roll, pitch, yaw] in degrees
float64[] rot_error_lowerbound      # estimated rotational lowerbounds (3-sigma) [roll, pitch, yaw] in degrees

