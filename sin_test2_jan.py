from __future__ import division, print_function
from copy import deepcopy
# from moveit_commander import MoveGroupCommander, RobotCommander
from moveit_commander.move_group import MoveGroupCommander
from moveit_commander.robot import RobotCommander
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point, Polygon, PolygonStamped
from scipy.spatial.transform import Rotation

import collections
import numpy as np
from scipy import signal, sparse, interpolate
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D

import cv2
from cv_bridge import CvBridge

from sklearn.neighbors import NearestNeighbors

import rosbag
import rospy
import sensor_msgs.point_cloud2

import ipdb

def load_frm_bag(filename):
    """Loads specific topics from rosbag and stores data as np.arrays

    Parameters
    ----------
    filename		: string, required
			    Name of .bag file
    Outputs
    ---------- 

    """

    bag = rosbag.Bag(filename)

    # topic_name == '/XTION3/camera/depth_registered/points':
    # no_of_msgs = bag.get_message_count(topic_name)
    
    x_lst = []
    y_lst = []
    z_lst = []
    rgb_lst = []

    #no need to record time for these, assume static image
    pcl_flag = False
    depth_flag = False
    color_img_flag = False
    for topic, msg, t in bag.read_messages():

        # if topic == '/XTION3/camera/depth_registered/points' and pcl_flag == False:
            # for pt in sensor_msgs.point_cloud2.read_points(msg, skip_nans=True):
                # x_lst.append(pt[0])
                # y_lst.append(pt[1])
                # z_lst.append(pt[2])
                # rgb_lst.append(pt[3])
            # pcl_flag = True

        if (topic == '/XTION3/camera/depth/image_rect' or topic == '/XTION3/camera/depth/image_rect/') and depth_flag == False:
            depth_msg = msg
            depth_flag = True

        elif topic == '/XTION3/camera/rgb/image_rect_color/compressed' and color_img_flag == False:
            color_img_msg = msg
            color_img_flag = True

    bag.close()

    return (x_lst,y_lst,z_lst,rgb_lst,depth_msg,color_img_msg)

#need to offset abit to make sure most of the nans are cropped
def crop_and_fill_nans(cv_depth, cv_color, border_offset=10):
    ''' test'''
    #crop nans at depth image border & do similarly for color image
    no_nan_indx = np.argwhere(~np.isnan(cv_depth))
    top_left_coord = np.array([     np.min(no_nan_indx[:,1])+border_offset,
                                    np.min(no_nan_indx[:,0])+border_offset  ])
    bttm_right_coord = np.array([   np.max(no_nan_indx[:,1])-border_offset,
                                    np.max(no_nan_indx[:,0])-border_offset  ])
    cv_depth = cv_depth[top_left_coord[1]:bttm_right_coord[1],top_left_coord[0]:bttm_right_coord[0]]
    cv_color = cv_color[top_left_coord[1]:bttm_right_coord[1],top_left_coord[0]:bttm_right_coord[0]]

    #crop nans until table border & do similarly for color image
    nan_indx = np.argwhere(np.isnan(cv_depth))
    top_rightmost_nan_col = np.max([coord for index, coord in enumerate(nan_indx) if np.min(nan_indx[:,0]) in coord])-border_offset
    bttm_leftmost_nan_col = np.min([coord for index, coord in enumerate(nan_indx) if np.max(nan_indx[:,0]) in coord])+border_offset
    cv_depth = cv_depth[:,bttm_leftmost_nan_col:top_rightmost_nan_col]
    cv_color = cv_color[:,bttm_leftmost_nan_col:top_rightmost_nan_col]

    #replace any remaining nans in depth image with mean of neighbouring non-nan values
    nan_indx = np.argwhere(np.isnan(cv_depth))
    px_coord_list = np.array([[0,1],[1,1],[1,0],[1,-1],[0,-1],[-1,-1],[-1,0],[-1,1]])
    for index in nan_indx:

        count = 0
        for coord in px_coord_list:
            if ~np.isnan(cv_depth[index[0]+coord[0],index[1]+coord[1]]):
                if np.isnan(cv_depth[index[0],index[1]]):
                    cv_depth[index[0],index[1]] = cv_depth[index[0]+coord[0],index[1]+coord[1]]
                    count = 1
                else:
                    cv_depth[index[0],index[1]] += cv_depth[index[0]+coord[0],index[1]+coord[1]]
                    count += 1

        cv_depth[index[0],index[1]] /= count

    return cv_depth, cv_color

# # http://answers.opencv.org/question/44580/how-to-resize-a-contour/
def scale_contour(scaling_factor,contour):
    '''test'''
    m = cv2.moments(contour)
    cont_center = np.array([int(m['m10']/m['m00']),int(m['m01']/m['m00'])])
    cont_scaled = contour.reshape(-1,2).astype(np.float64)
    for coord in cont_scaled:
        coord -= cont_center
        coord *= scaling_factor
        coord += cont_center

    return cont_scaled

def smooth_contours_w_butterworth(contour, fltr_ordr, norm_cutoff_freq, fltr_type_str):
    '''test'''
    # # https://stackoverflow.com/questions/51259361/smooth-a-bumpy-circle/51267877
    #filtering contour with butterworth filter so its not so jagged
    cont_coord = np.array(contour).reshape((-1,2))
    x_coord = np.array([coord[0] for coord in cont_coord])
    y_coord = np.array([coord[1] for coord in cont_coord])
    cmplx_sgnl = x_coord + 1.0j*y_coord

    # # https://dsp.stackexchange.com/questions/49460/apply-low-pass-butterworth-filter-in-python
    b,a = signal.butter(fltr_ordr,norm_cutoff_freq,fltr_type_str)

    return signal.filtfilt(b,a,cmplx_sgnl)

def generate_bspline(no_of_pts,contour,smthnss,per_flag):
    '''test'''
    x = contour.reshape(-1,2)[:,0]
    y = contour.reshape(-1,2)[:,1]
    if per_flag is True:
        tck_params,xy_as_u = interpolate.splprep([x,y],k=5,s=smthnss,per=1)
    elif per_flag is False:
        tck_params,xy_as_u = interpolate.splprep([x,y],k=5,s=smthnss)
    u_interp = np.linspace(xy_as_u.min(),xy_as_u.max(), no_of_pts)
    
    return np.transpose(np.array(interpolate.splev(u_interp, tck_params)))

def contour_center_n_dists_frm_apprx(contour,apprx_contour):
    '''test'''
    cont_coord = contour.reshape(-1,2)
    #determine center of graphical input bspline
    cont_center = np.array([np.mean(cont_coord[:,0]),np.mean(cont_coord[:,1])])

    if apprx_contour is None:
        apprx_coord = contour.reshape(-1,2)
    else:
        apprx_coord = apprx_contour.reshape(-1,2)
    #determine L2 norm distance for every point on approximated bspline and center of bspline from graphical input
    dist_arr = np.zeros((apprx_coord.shape[0],1))
    for i in range(apprx_coord.shape[0]):
        dist_arr[i] = np.linalg.norm(cont_center - apprx_coord[i])

    return cont_center, dist_arr

# # https://gamedev.stackexchange.com/a/98111
def strghtnss_metric(line_coord):
    '''test'''
    x = np.transpose(line_coord)[0,:]
    y = np.transpose(line_coord)[1,:]
    
    return np.sum(np.gradient(np.gradient(x)))**2 + np.sum(np.gradient(np.gradient(y)))**2

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def set_up_plt(fig,label_font_size,tick_font_size,x_label,y_label,z_label=None):
    '''test'''
    if z_label is None:
        ax = fig.add_subplot(111)
        ax.set_xlabel(x_label,fontsize=label_font_size)
        ax.set_ylabel(y_label,fontsize=label_font_size)
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontsize(tick_font_size)
    else:
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel(x_label, fontsize=label_font_size)
        ax.set_ylabel(y_label, fontsize=label_font_size)
        ax.set_zlabel(z_label, fontsize=label_font_size)
        for label in (ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels()):
            label.set_fontsize(tick_font_size)

    return ax

def plt_traj_3d(fig,traj,z_val=None):
    '''test'''
    ax = set_up_plt(fig,36,20,'x','y','z')
    if z_val is None:
        ax.scatter(traj[:,0],traj[:,1],traj[:,2])
    else:
        ax.scatter(traj[:,0],traj[:,1],z_val*np.ones((traj.shape[0],)))
    set_axes_equal(ax)
    ax.set_aspect('equal', adjustable='box')
    ax.grid()
    # ax.view_init(elev=25.0,azim=-30.0)
    fig.set_tight_layout({'pad':1.0})
    fig.set_size_inches(10,12)

def set_up_2dplot(fig,x_label,y_label,label_font_size,tick_font_size):
    '''test'''
    ax = fig.add_subplot(111)
    ax.set_xlabel(x_label,fontsize=label_font_size)
    ax.set_ylabel(y_label,fontsize=label_font_size)
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(tick_font_size)
    
    return ax

def cont_pt_by_pt(cont_coord,img_name,img,color_arr,pause_time,debug_mode=False):
    '''test'''
    for coord in cont_coord.reshape(-1,2):
        if debug_mode is True:
            print(coord)
        img[coord[1].astype(np.int),coord[0].astype(np.int)] = color_arr
        cv2.imshow(img_name,img)
        cv2.waitKey(pause_time)

    if debug_mode is True:
        print('-----')
        print(cont_coord.reshape(-1,2)[0])
        print(cont_coord.reshape(-1,2)[-1])
        print(cont_coord.shape)
        print('-----')
        print('waitKey(0)')
        cv2.imshow(img_name,img)
        cv2.waitKey(0)

def main():
    #load
    exp_filename = 'depth&rgbOnly_circShape_1stFrame'
    # exp_filename = 'depth&rgbOnly_elFrame_1stFrame'
    # exp_filename = 'depth&rgbOnly_lowNonRegShape_plainBckgrnd_1stFrame'
    # exp_filename = 'depth&rgbOnly_sqFrame_1stFrame'
    data_dict = {exp_filename: 0}

    namedTup = collections.namedtuple('namedTup','x y z rgb depth_msg color_img_msg')

    for file_str in data_dict.keys():

        data = load_frm_bag('vision_datasets/first_frame/'+file_str+'.bag')
        data_dict[file_str] = namedTup(data[0],data[1],data[2],data[3],data[4],data[5])

    # # https://stackoverflow.com/questions/47751323/get-depth-image-in-grayscale-in-ros-with-imgmsg-to-cv2-python
    bridge = CvBridge()

    # cv_img = bridge.imgmsg_to_cv2(data_dict[exp_filename].depth_msg, '32FC1')
    #load as opencv images from ROS msg
    depth_img = bridge.imgmsg_to_cv2(data_dict[exp_filename].depth_msg, "passthrough").copy()
    color_img = bridge.compressed_imgmsg_to_cv2(data_dict[exp_filename].color_img_msg, "passthrough").copy()

    depth_crppd,color_crppd = crop_and_fill_nans(depth_img,color_img,border_offset=30)

    # # https://stackoverflow.com/questions/46689428/convert-np-array-of-type-float64-to-type-uint8-scaling-values
    #convert to grayscale image since Canny works only on uint8
    color_gray = cv2.cvtColor(color_crppd,cv2.COLOR_BGR2GRAY)
    #can't convert float32 to uint8 using cvtColor so normalize then scale to (0-255)
    depth_gray = cv2.normalize(depth_crppd,None,0.0,1.0,cv2.NORM_MINMAX)
    depth_gray = cv2.convertScaleAbs(depth_gray,None,np.iinfo(color_gray.dtype).max)

    # # https://stackoverflow.com/questions/4292249/automatic-calculation-of-low-and-high-thresholds-for-the-canny-operation-in-open
    #determine thresholds for canny edge detection automatically
    thresh_color,color_thrshd = cv2.threshold(color_gray,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)
    color_canny = cv2.Canny(color_gray,0.5*thresh_color,thresh_color)

    color_dltd = cv2.dilate(color_canny,None,iterations=1)
    color_cont_img, cont_color, hierachy_color = cv2.findContours(color_dltd,cv2.RETR_TREE,cv2.CHAIN_APPROX_TC89_KCOS)

    # cv2.imshow('Color canny',color_canny)

    if len(cont_color) == 2:

        cont = cont_color[0]
        cont_cnnctd = np.vstack((cont.reshape(-1,2),cont.reshape(-1,2)[0,:]))
        cont_scld = scale_contour(0.8,cont_cnnctd)

    else:
        #filter out noise from depth image
        depth_gauss = cv2.GaussianBlur(depth_gray,(9,9),0)

        thresh_depth,depth_thrshd = cv2.threshold(depth_gauss,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)
        depth_canny = cv2.Canny(depth_gauss,0.5*thresh_depth,thresh_depth) 

        depth_cont_img, cont_depth, hierachy_depth = cv2.findContours(depth_canny,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

        #close gaps and make edges fatter to use depth image as mask on RGB image
        depth_dltd = cv2.dilate(depth_canny,None,iterations=3)
        depth_mask = np.where((depth_dltd==0),0,1).astype('uint8')
        combi_canny = color_canny*depth_mask[:,:]

        combi_cont_img, cont_combi, hierachy_combi = cv2.findContours(combi_canny,cv2.RETR_TREE,cv2.CHAIN_APPROX_TC89_KCOS)

        #close gaps in combination image until solid shape is formed
        i = 0
        while len(cont_combi) > 1:
            i += 1
            print(i)
            if i > 100:
                print(len(cont_combi))
                break

            combi_clsd = cv2.morphologyEx(combi_canny,cv2.MORPH_CLOSE,None,iterations=i)
            combi_cont_img, cont_combi, hiearachy_combi = cv2.findContours(combi_clsd,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

        #erode shape to make it smaller
        # combi_erdd = cv2.erode(combi_clsd,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(25,25)),iterations=1)
        combi_erdd = cv2.erode(combi_clsd,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15)),iterations=1)
        # #smooth jagged edges of solid shape using median filtering
        combi_medblrrd = cv2.medianBlur(combi_erdd,25)

        combi_cont_img, cont_combi, hiearachy_combi = cv2.findContours(combi_medblrrd,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

        cont = cont_combi[0]
        #connect the end of the contour to the beginning
        cont_cnnctd = np.vstack((cont.reshape(-1,2),cont.reshape(-1,2)[0,:]))
        #scale contour 
        #WARNING: scale first before doing any operation otherwise moment computation will not work
        cont_scld = scale_contour(1.0,cont_cnnctd)
        # cont_pt_by_pt(cont_scld,'test',color_crppd,(0,255,0),1)

        # cv2.imshow('Depth canny',depth_canny)
        # cv2.imshow('Depth dilated',depth_dltd)
        # cv2.imshow('Combination canny',combi_canny)
        # cv2.imshow('Combination closed',combi_clsd)
        # cv2.imshow('Combination eroded',combi_erdd)
        # cv2.imshow('Combination median blurred',combi_medblrrd)

    #smooth countour for shape matching, 
    cont_sgnl_fltd = smooth_contours_w_butterworth(cont_scld,5,0.2,'low')
    cont_fltd = np.array(zip(list(cont_sgnl_fltd.real),list(cont_sgnl_fltd.imag)))
    # cont_pt_by_pt(cont_fltd,'test',color_crppd,(0,255,0),1,debug_mode=True)
    # cv2.drawContours(color_crppd,[cont_coord_fltd.reshape(-1,cont.shape[1],cont.shape[2]).astype(cont.dtype)],-1,(255,0,0),1)

    # interpolate with equidistant points using bspline and make sure trajectory is closed with per argument in splprep
    traj_bspline = generate_bspline(1e2,cont_fltd,smthnss=20,per_flag=True)
    # cont_pt_by_pt(traj_bspline,'test',color_crppd,(255,0,0),1)

    t = np.zeros(traj_bspline.shape[0])  # cumsum of point to point distances in pixel
    for i in range(traj_bspline.shape[0]):
        if i == 0:
            crv_len = 0.0
        else:
            crv_len += np.linalg.norm(traj_bspline[i] - traj_bspline[i-1])  # JB: this could be done using integration of spline
            t[i] = t[i-1] + np.linalg.norm(traj_bspline[i] - traj_bspline[i-1])

    # # https://stackoverflow.com/questions/12028690/polyline-following-a-curved-path
    amp = 3.0
    no_of_prds = 20

    pts_per_prd = np.round(traj_bspline.shape[0]/no_of_prds)
    x = traj_bspline.reshape(-1,2)[:,0]
    y = traj_bspline.reshape(-1,2)[:,1]
    pttrn_traj = np.zeros((x.shape[0],2))
    for j in range(x.shape[0]):  # going through all the base contour points
        if j < x.shape[0]-1:  # treat the last point differently in the overlay
            unit_perp = np.array([-(y[j+1]-y[j]),x[j+1]-x[j]]/np.linalg.norm(traj_bspline[j+1] - traj_bspline[j]))  # get perpendicular vector (why is y component taken negative?)
        else:
            unit_perp = np.array([-(2*y[j]-y[j]),2*x[j]-x[j]]/np.linalg.norm(2*traj_bspline[j] - traj_bspline[j]))
        pttrn_traj[j] = traj_bspline.reshape(-1,2)[j] + amp*np.sin(j*(2*np.pi/pts_per_prd))*unit_perp

        # f = no_of_prds/crv_len
        # swth_sum_term = 0.0
        # for n in range(1,100):
            # swth_sum_term += (np.sin(2.0*np.pi*n*f*t[j])/n)*(-1)**n
        # pttrn_traj[j] = traj_bspline.reshape(-1,2)[j] + (((amp/2.0)-(amp/np.pi)*swth_sum_term)*unit_perp - (0.5*amp*unit_perp))

    #should not need to filter since sine wave should be smooth
    #interpolate with equidistant points using bspline and make sure trajectory is closed with per argument in splprep
    pttrn_bspline = generate_bspline(2*traj_bspline.shape[0],pttrn_traj,smthnss=0,per_flag=True)

    #smooth countour for velocity and acceleration
    #WARNING: too low cutoff frequency causes start and end of contour to disconnect
    # pttrn_sgnl_fltd = smooth_contours_w_butterworth(pttrn_traj,5,0.4,'low')
    # pttrn_fltd = np.array(zip(list(pttrn_sgnl_fltd.real),list(pttrn_sgnl_fltd.imag)))
    # # cont_pt_by_pt(pttrn_fltd,'test',color_crppd,(255,255,255),1)
    # pttrn_bspline = generate_bspline(2*traj_bspline.shape[0],pttrn_fltd,smthnss=0,per_flag=True)

    cont_pt_by_pt(pttrn_bspline,'test',color_crppd,(0,255,255),1)
    # cv2.imshow('Color canny',color_canny)
    # cv2.imshow('Color cropped',color_crppd)
    # cv2.imshow('Depth cropped',depth_crppd)
    cv2.waitKey(0)

    # #plot displacement as scatter to make it easy to determine if enough trajectory points visually
    # dt = 0.1
    # dx = traj_bspline.reshape(-1,2)[:,0]
    # dy = traj_bspline.reshape(-1,2)[:,1]
    # # dx = pttrn_bspline.reshape(-1,2)[:,0]
    # # dy = pttrn_bspline.reshape(-1,2)[:,1]
    # print np.max(np.diff(dx))
    # print np.max(np.diff(dy))
    # fig1 = plt.figure()
    # ax1 = set_up_2dplot(fig1,'Secs','Displacement',20,12)
    # ax1.scatter(np.arange(0,dt*len(dx),dt),dx)
    # ax1.scatter(np.arange(0,dt*len(dy),dt),dy)
    # ax1.grid()

    # vel_x = np.gradient(dx,dt)
    # vel_y = np.gradient(dy,dt)
    # fig2 = plt.figure()
    # ax2 = set_up_2dplot(fig2,'Secs','Vel',20,12)
    # ax2.plot(np.arange(0,dt*len(vel_x),dt),vel_x)
    # ax2.plot(np.arange(0,dt*len(vel_y),dt),vel_y)
    # ax2.grid()

    # acc_x = np.gradient(vel_x,dt)
    # acc_y = np.gradient(vel_y,dt)
    # fig3 = plt.figure()
    # ax3 = set_up_2dplot(fig3,'Secs','Acc',20,12)
    # ax3.plot(np.arange(0,dt*len(acc_x),dt),acc_x)
    # ax3.plot(np.arange(0,dt*len(acc_y),dt),acc_y)
    # ax3.grid()
    # plt.show()

    #robot simulation begins from here
    #configuration
    node_name    = 'sine_npt'
    group_name   = 'r1_arm'

    #initialize robot and frame_id
    rospy.init_node(node_name, anonymous=False)
    group = MoveGroupCommander(group_name)
    group.set_pose_reference_frame('r1_link_0')
    #set velocity and acceleration
    group.set_max_velocity_scaling_factor(1.0)
    group.set_max_acceleration_scaling_factor(0.5)

    #translate and rescale trajectory as well as swap coordinates for rviz
    rviz_scale = 1/400 # 400 or 600
    rviz_center = [0.55, 0.0]
    pttrn_center = np.array([np.mean(pttrn_bspline[:,0]),np.mean(pttrn_bspline[:,1])])
    rviz_bspl = (pttrn_bspline - pttrn_center)*rviz_scale
    rviz_bspl[:,[0,1]] = rviz_bspl[:,[1,0]]  # JB: exchange columns. I guess that rotates the object 90 degrees
    rviz_bspl[:,0] += rviz_center[0]
    rviz_bspl[:,1] += rviz_center[1]
    #do the same for object contour
    if len(cont_color) == 2:
        rviz_cont = scale_contour(1.3, cont_color[0])
    else:
        combi_cont_img, cont_combi, hiearachy_combi = cv2.findContours(combi_erdd,cv2.RETR_TREE,cv2.CHAIN_APPROX_TC89_KCOS)
        rviz_cont = scale_contour(1.3, cont_combi[0])
    cont_center = np.array([np.mean(rviz_cont[:,0]),np.mean(rviz_cont[:,1])])
    rviz_cont = (rviz_cont - cont_center)*rviz_scale
    rviz_cont[:,[0,1]] = rviz_cont[:,[1,0]]
    rviz_cont[:,0] += rviz_center[0]
    rviz_cont[:,1] += rviz_center[1]

    #initialize polygon for target object visualization
    # obj_poly = PolygonStamped()
    # obj_poly.header.frame_id = group.get_current_pose().header.frame_id
    # for i in range(rviz_cont.shape[0]):
        # poly_pt = Point()
        # poly_pt.x = rviz_cont[i,0]
        # poly_pt.y = rviz_cont[i,1]
        # poly_pt.z = 0.05
        # obj_poly.polygon.points.append(poly_pt)
    obj_mrkr = Marker()
    obj_mrkr.header.frame_id = group.get_current_pose().header.frame_id
    obj_mrkr.type = obj_mrkr.CYLINDER
    obj_mrkr.action = obj_mrkr.ADD
    obj_mrkr.scale.x = 1.2*(2*np.max(np.abs(rviz_bspl[:,0]-rviz_center[0])))
    obj_mrkr.scale.y = 1.2*(2*np.max(np.abs(rviz_bspl[:,1]-rviz_center[1])))
    obj_mrkr.scale.z = 0.09
    obj_mrkr.pose.position.x = rviz_center[0]
    obj_mrkr.pose.position.y = rviz_center[1]
    obj_mrkr.color.a = 1.0
    obj_mrkr.color.r = 139/255
    obj_mrkr.color.g = 69/255
    obj_mrkr.color.b = 19/255

    #initialize marker for trajectory visualization
    traj_mrkr = Marker()
    traj_mrkr.header.frame_id = group.get_current_pose().header.frame_id
    traj_mrkr.type = traj_mrkr.LINE_STRIP
    traj_mrkr.action = traj_mrkr.ADD
    traj_mrkr.scale.x = 0.01
    traj_mrkr.color.a = 1.0
    traj_mrkr.color.r = 1.0
    traj_mrkr.color.g = 1.0
    traj_mrkr.color.b = 0.0

    #init zero pose
    q_zero = [0, 0, 0, 0, 0, 0, 0]

    #init start pose
    strt_pose = deepcopy(group.get_current_pose().pose)
    strt_pose.position.x = rviz_center[0]
    strt_pose.position.y = rviz_center[1]
    strt_pose.position.z = 0.6
    rot = Rotation.from_euler('xyz',[0.0,180.0,0.0],degrees=True)
    strt_pose.orientation.x = rot.as_quat()[0]
    strt_pose.orientation.y = rot.as_quat()[1]
    strt_pose.orientation.z = rot.as_quat()[2]
    strt_pose.orientation.w = rot.as_quat()[3]

    #convert trajectory into poses as well as markers
    pttrn_seq = []
    for i in range(pttrn_bspline.shape[0]):
        curr_pose = deepcopy(strt_pose)
        curr_pose.position.x = rviz_bspl[i,0]
        curr_pose.position.y = rviz_bspl[i,1]
        curr_pose.position.z = 0.25
        pttrn_seq.append(curr_pose)

        mrkr_pt = Point()
        mrkr_pt.x = curr_pose.position.x
        mrkr_pt.y = curr_pose.position.y
        mrkr_pt.z = curr_pose.position.z - 0.2
        traj_mrkr.points.append(mrkr_pt)

    # ipdb.set_trace()

    rospy.loginfo('Visualizing target object...')
    obj_pub = rospy.Publisher('/obj_marker', Marker, queue_size=10)
    # obj_pub = rospy.Publisher('/obj_polygon', PolygonStamped, queue_size=10)
    rospy.sleep(1) #need to wait otherwise not enough time for publisher to initialize
    obj_pub.publish(obj_mrkr)
    # obj_pub.publish(obj_poly)
    rospy.loginfo('Target object visualized')

    rospy.loginfo('Resetting robot to zero pose...')
    group.set_joint_value_target(q_zero)
    traj = group.plan()
    group.execute(traj[1])
    rospy.loginfo('Zero pose achieved')

    rospy.loginfo('Moving to starting pose...')
    group.set_pose_target(strt_pose)
    traj = group.plan()
    group.execute(traj[1])
    rospy.loginfo('Starting pose achieved')

    rospy.loginfo('Moving to beginning of trajectory...')
    strt_seq = [strt_pose, pttrn_seq[0]]
    # traj_strt, fraction = group.compute_cartesian_path(strt_seq[1:], eef_step=1e-3, jump_threshold=2.0)
    group.set_pose_target(strt_seq[-1])
    traj_strt_2 = group.plan()
    # print('traj_strt length: ', len(traj_strt.joint_trajectory.points))
    # print('fraction: ', fraction)
    group.execute(traj_strt_2[1])
    # group.execute(traj_strt)
    rospy.loginfo('Beginning of trajectory reached...')

    rospy.loginfo('Visualizing trajectory...')
    traj_pub = rospy.Publisher('/traj_marker', Marker, queue_size=10)
    rospy.sleep(1) #need to wait otherwise not enough time for publisher to initialize
    traj_pub.publish(traj_mrkr)
    rospy.loginfo('Trajectory visualized')

    rospy.loginfo('Moving through entire trajectory...')
    traj_pttrn, fraction = group.compute_cartesian_path(pttrn_seq[1:-1], eef_step=5e-4, jump_threshold=2.0)
    print('traj_pttrn length: ', len(traj_pttrn.joint_trajectory.points))
    print('fraction: ', fraction)
    group.execute(traj_pttrn)
    rospy.sleep(1) #need to wait otherwise not enough time for motion to complete
    if fraction < 1.0:
        rospy.logwarn('Could not perform linear interpolation over entire trajectory, terminating partway')
    else:
        rospy.loginfo('Entire trajectory completed')

    rospy.loginfo('Deleting trajectory visualization...')
    traj_mrkr.action = traj_mrkr.DELETE
    traj_pub.publish(traj_mrkr)
    rospy.loginfo('Trajectory visualization deleted')

    rospy.loginfo('Moving back to starting pose...')
    end_seq = [pttrn_seq[-1], strt_pose]
    traj_end, fraction = group.compute_cartesian_path(end_seq, eef_step=1e-3, jump_threshold=2.0)
    print('traj_end length: ', len(traj_end.joint_trajectory.points))
    print('fraction: ', fraction)
    group.execute(traj_end)
    rospy.loginfo('Starting pose achieved')

if __name__ == "__main__":
    main()
