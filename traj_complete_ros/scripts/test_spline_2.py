import collections
import numpy as np
from scipy import signal, interpolate
import matplotlib.pyplot as plt

import cv2
from cv_bridge import CvBridge

import rosbag
import sensor_msgs.point_cloud2
from traj_complete_ros.apply_pattern_to_contour import apply_pattern_to_contour, ApproxMethod
from traj_complete_ros.utils import BsplineGen, smooth_contours_w_butterworth,cont_pt_by_pt,set_up_2dplot, plot_displacement_scatter
import pkg_resources
import os


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
    outbag = rosbag.Bag(filename+'_new','w')

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

        if topic == '/XTION3/camera/depth_registered/points' and pcl_flag == False:
            for pt in sensor_msgs.point_cloud2.read_points(msg, skip_nans=True):
                x_lst.append(pt[0])
                y_lst.append(pt[1])
                z_lst.append(pt[2])
                rgb_lst.append(pt[3])
            pcl_flag = True

        elif topic == '/XTION3/camera/depth/image_rect' or topic ==  '/XTION3/camera/depth/image_rect/' and depth_flag == False:
            depth_msg = msg
            outbag.write(topic,msg)
            depth_flag = True

        elif topic == '/XTION3/camera/rgb/image_rect_color/compressed' and color_img_flag == False:
            color_img_msg = msg
            outbag.write(topic,msg)
            color_img_flag = True

    bag.close()
    outbag.close()

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

# # https://gamedev.stackexchange.com/a/98111
def strghtnss_metric(line_coord):
    '''test'''
    x = np.transpose(line_coord)[0,:]
    y = np.transpose(line_coord)[1,:]

    return np.sum(np.gradient(np.gradient(x)))**2 + np.sum(np.gradient(np.gradient(y)))**2


def main():
    #load
    # exp_filename = 'cameraOnly_sqFrame_2019-02-21-18-07-22'
    # exp_filename = 'cameraOnly_circShape_2019-03-01-15-09-33'
    # exp_filename = 'cameraOnly_lowNonRegShape_plainBckgrnd_2019-03-01-15-45-42'
    exp_filename = 'eL_LFwave_ontop_2019-01-29-12-48-44'
    data_dict = {exp_filename: 0}

    namedTup = collections.namedtuple('namedTup','x y z rgb depth_msg color_img_msg')

    for file_str in data_dict.keys():

        # data = load_frm_bag('/home/algernon/ros_melodic_ws/base_ws/src/icra-2021-trajectory-autocompletion-tan-thesis/vision_datasets/'+file_str+'.bag')
        data = load_frm_bag('../../vision_datasets/'+file_str+'.bag')
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
    cont_color, hierachy_color = cv2.findContours(color_dltd,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    cont_color, hierachy_color = cv2.findContours(color_dltd,cv2.RETR_TREE,cv2.CHAIN_APPROX_TC89_KCOS)

    cv2.imshow('Color canny',color_canny)

    if len(cont_color) == 2:

        cont = cont_color[0]

    else:
        #filter/smooth depth image so that Canny edges turn out smoother
        depth_hires = cv2.pyrUp(depth_gray)
        depth_hires = cv2.bilateralFilter(depth_hires,9,75,75)
        depth_bifltd = cv2.pyrDown(depth_hires)

        thresh_depth,depth_thrshd = cv2.threshold(depth_bifltd,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)
        depth_canny = cv2.Canny(depth_bifltd,0.5*thresh_depth,thresh_depth)

        cont_depth, hierachy_depth = cv2.findContours(depth_canny,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

        #close gaps and make edges fatter to use depth image as mask on RGB image
        depth_dltd = cv2.dilate(depth_canny,None,iterations=4)
        depth_mask = np.where((depth_dltd==0),0,1).astype('uint8')
        combi_canny = color_canny*depth_mask[:,:]

        cont_combi, hierachy_combi = cv2.findContours(combi_canny,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

        #close gaps in combination image until solid shape is formed
        i = 0
        while len(cont_combi) > 1:
            i += 1
            print(i)
            if i > 100:
                print(len(cont_combi))
                break

            combi_clsd = cv2.morphologyEx(combi_canny,cv2.MORPH_CLOSE,None,iterations=i)
            cont_combi, hiearachy_combi = cv2.findContours(combi_clsd,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

        #smooth jagged edges of solid shape using median filtering
        combi_hires = cv2.pyrUp(combi_clsd)
        combi_hires = cv2.medianBlur(combi_hires,55)
        combi_medblrrd = cv2.pyrDown(combi_hires)
        cont_combi, hiearachy_combi = cv2.findContours(combi_medblrrd,cv2.RETR_TREE,cv2.CHAIN_APPROX_TC89_KCOS)

        cont = cont_combi[0]
        # cv2.imshow('Depth canny',depth_canny)
        # cv2.imshow('Depth dilated',depth_dltd)
        # cv2.imshow('Combination canny',combi_canny)
        # cv2.imshow('Combination closed',combi_clsd)
        # cv2.imshow('Combination median blurred',combi_medblrrd)
        # cv2.waitKey(0)

    #connect the end of the contour to the beginning
    cont_cnnctd = np.vstack((cont.reshape(-1,2),cont.reshape(-1,2)[0,:]))
    # cont_pt_by_pt(cont_cnnctd,'test',color_crppd,(0,0,255),1)
    # cv2.drawContours(color_crppd, [cont_cnnctd], -1, (0,0,255), 1)

    #scale contour
    #WARNING: scale first before doing any operation otherwise moment computation will not work
    cont_scld = scale_contour(0.85,cont_cnnctd)
    # cont_pt_by_pt(cont_scld,'test',color_crppd,(0,255,0),1)

    #smooth countour for shape matching,
    #WARNING: too low cutoff frequency causes start and end of contour to disconnect
    cont_sgnl_fltd = smooth_contours_w_butterworth(cont_scld,5,0.2,'low')
    cont_fltd = np.array(zip(list(cont_sgnl_fltd.real),list(cont_sgnl_fltd.imag)))
    # cont_pt_by_pt(cont_fltd,'test',color_crppd,(0,255,0),1)
    # cv2.drawContours(color_crppd,[cont_coord_fltd.reshape(-1,cont.shape[1],cont.shape[2]).astype(cont.dtype)],-1,(255,0,0),1)

    # interpolate with equidistant points using bspline and make sure trajectory is closed with per argument in splprep
    traj_bspline = BsplineGen()
    traj_bspline.generate_bspline_pars(cont_fltd,per_flag=True)
    traj_bspline.nmbPts = 1e3
    traj_bspline_sampled = traj_bspline.generate_bspline_sample()
    cont_pt_by_pt(traj_bspline_sampled,'test',color_crppd,(255,0,0),1)

    no_of_prds = int(20) #good value is 20
    #################################################
    # waypts = np.array([[2.56760282,7.94209293],[3.04756108,8.06718675],[3.61099035,7.99895376],[3.98660986,7.66916096],[4.32049387,6.89585371],[4.41092079,6.07705779],[4.24397878,5.0194464],[3.90313886,4.0641845],[3.33970959,3.4728319],[2.81801582,3.56380922],[2.86670724,4.26888348],[3.12407617,4.73514227],[3.63881402,5.25826188],[4.16050778,5.45158869],[4.7448048,5.47433302],[5.1969394,5.16728455],[5.63516216,4.59867628],[6.01078167,4.00732367],[6.0246935,3.06343394],[5.73950091,2.1536607]])
    waypts = np.array([[1.90679071,7.94209293],[2.21285106,7.68053313],[2.49804365,7.38485682],[2.74150074,7.16878568],[2.99886966,6.90722587],[3.23537084,6.57743307],[3.40231284,6.03156913],[3.36057734,5.12179589],[3.06842883,4.23476698],[2.65802974,3.87085769],[2.28936614,4.18927832],[2.44239631,4.86023609],[2.87366316,5.37198353],[3.37448917,5.62217117],[3.75010869,5.54256602],[4.0422572,5.39472786],[4.38309712,5.07630723],[4.62655421,4.81474742],[4.87696722,4.51907112],[5.21085123,4.22339482]])
    x = np.linspace(0, 2*np.pi, 1000)
    y = 3*np.sin(x)
    # waypts = np.array(zip(list(x),list(y)))
    y = 3*np.sin(x)+ 3*np.cos(np.pi/3+x*10)
    # waypts = np.array(zip(list(x), list(y)))
    waypts = np.stack(list(zip(list(x), list(y))))

    #determine bspline for graphical input
    data = BsplineGen.generateLibraryData(waypts, name="sine")
    input_bspline_p = BsplineGen.fromLibraryData(**data["sine"])

    library_path = os.path.join("../config", "default_configuration.yaml")
    input_bspline_p.appendToLibrary(library_path, "double_sine")
    # input_bspline_p = BsplineGen()
    # input_bspline_p.generate_bspline_pars(waypts,per_flag=False)
    input_bspline_p.nmbPts = waypts.shape[0]
    input_bspline_sampled = input_bspline_p.generate_bspline_sample(2500)

    pttrn_bspline = apply_pattern_to_contour(input_bspline_p, traj_bspline_sampled, 31, approx_method=ApproxMethod.REG)
    pttrn_bspline_s = pttrn_bspline.generate_bspline_sample()

    # cont_pt_by_pt(trnsfrmd_apprx_bspline,'test',color_crppd,(0,255,255),1)
    cont_pt_by_pt(pttrn_bspline_s,'test',color_crppd,(255,255,0),1)

    # # cv2.imshow('Color gray',color_gray)
    # # cv2.imshow('Depth gray',depth_gray)
    # cv2.imshow('Color canny',color_canny)
    # cv2.imshow('Color cropped',color_crppd)
    # cv2.imshow('Depth cropped',depth_crppd)
    # cv2.waitKey(0)
    #
    # #plot displacement as scatter to make it easy to determine if enough trajectory points visually
    # dt = 0.1
    # # dx = traj_bspline.reshape(-1,2)[:,0]
    # # dy = traj_bspline.reshape(-1,2)[:,1]
    # dx = pttrn_bspline.reshape(-1,2)[:,0]
    # dy = pttrn_bspline.reshape(-1,2)[:,1]
    # print(np.max(np.diff(dx)))
    # print(np.max(np.diff(dy)))
    # fig1 = plt.figure()
    # ax1 = set_up_2dplot(fig1,'Secs','Displacement',20,12)
    # ax1.scatter(np.arange(0,dt*len(dx),dt),dx)
    # ax1.scatter(np.arange(0,dt*len(dy),dt),dy)
    # ax1.grid()
    #
    # vel_x = np.gradient(dx,dt)
    # vel_y = np.gradient(dy,dt)
    # fig2 = plt.figure()
    # ax2 = set_up_2dplot(fig2,'Secs','Vel',20,12)
    # ax2.plot(np.arange(0,dt*len(vel_x),dt),vel_x)
    # ax2.plot(np.arange(0,dt*len(vel_y),dt),vel_y)
    # ax2.grid()
    #
    # acc_x = np.gradient(vel_x,dt)
    # acc_y = np.gradient(vel_y,dt)
    # fig3 = plt.figure()
    # ax3 = set_up_2dplot(fig3,'Secs','Acc',20,12)
    # ax3.plot(np.arange(0,dt*len(acc_x),dt),acc_x)
    # ax3.plot(np.arange(0,dt*len(acc_y),dt),acc_y)
    # ax3.grid()
    # plt.show()

    # plot displacement as scatter to make it easy to determine if enough trajectory points visually
    plot_displacement_scatter(waypts, 'waypts')
    plot_displacement_scatter(input_bspline_sampled, 'input_bspline_sampled')
    plot_displacement_scatter(pttrn_bspline_s, 'pttrn_bspline_s')


if __name__ == "__main__":
    main()
