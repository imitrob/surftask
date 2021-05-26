import collections
import numpy as np
from scipy import signal, sparse, interpolate
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D

import cv2
from cv_bridge import CvBridge

import rosbag
import rospy
import sensor_msgs.point_cloud2

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

def generate_bspline(no_of_pts,contour,per_flag):
    '''test'''
    x = contour.reshape(-1,2)[:,0]
    y = contour.reshape(-1,2)[:,1]
    if per_flag is True:
        tck_params,xy_as_u = interpolate.splprep([x,y],s=0,per=1)
    elif per_flag is False:
        tck_params,xy_as_u = interpolate.splprep([x,y],s=0)
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
    # exp_filename = 'cameraOnly_sqFrame_2019-02-21-18-07-22'
    # exp_filename = 'cameraOnly_circShape_2019-03-01-15-09-33'
    # exp_filename = 'cameraOnly_lowNonRegShape_plainBckgrnd_2019-03-01-15-45-42'
    exp_filename = 'eL_LFwave_ontop_2019-01-29-12-48-44'
    data_dict = {exp_filename: 0}

    namedTup = collections.namedtuple('namedTup','x y z rgb depth_msg color_img_msg')

    for file_str in data_dict.keys():

        data = load_frm_bag('vision_datasets/'+file_str+'.bag')
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
    # color_cont_img, cont_color, hierachy_color = cv2.findContours(color_dltd,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    color_cont_img, cont_color, hierachy_color = cv2.findContours(color_dltd,cv2.RETR_TREE,cv2.CHAIN_APPROX_TC89_KCOS)

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

        depth_cont_img, cont_depth, hierachy_depth = cv2.findContours(depth_canny,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

        #close gaps and make edges fatter to use depth image as mask on RGB image
        depth_dltd = cv2.dilate(depth_canny,None,iterations=4)
        depth_mask = np.where((depth_dltd==0),0,1).astype('uint8')
        combi_canny = color_canny*depth_mask[:,:]

        combi_cont_img, cont_combi, hierachy_combi = cv2.findContours(combi_canny,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

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

        #smooth jagged edges of solid shape using median filtering
        combi_hires = cv2.pyrUp(combi_clsd)
        combi_hires = cv2.medianBlur(combi_hires,55)
        combi_medblrrd = cv2.pyrDown(combi_hires)
        combi_cont_img, cont_combi, hiearachy_combi = cv2.findContours(combi_medblrrd,cv2.RETR_TREE,cv2.CHAIN_APPROX_TC89_KCOS)

        cont = cont_combi[0]
        cv2.imshow('Depth canny',depth_canny)
        cv2.imshow('Depth dilated',depth_dltd)
        cv2.imshow('Combination canny',combi_canny)
        cv2.imshow('Combination closed',combi_clsd)
        cv2.imshow('Combination median blurred',combi_medblrrd) 
        cv2.waitKey(0)

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
    traj_bspline = generate_bspline(1e3,cont_fltd,per_flag=True)
    cont_pt_by_pt(traj_bspline,'test',color_crppd,(255,0,0),1)

    no_of_prds = int(20) #good value is 20
    #shift bspline indices so that patterns dont end on a corner if possible
    traj_bspline = np.vstack((traj_bspline[traj_bspline.shape[0]/no_of_prds:-1,:],traj_bspline[0:traj_bspline.shape[0]/no_of_prds,:]))
    #################################################
    # waypts = np.array([[2.56760282,7.94209293],[3.04756108,8.06718675],[3.61099035,7.99895376],[3.98660986,7.66916096],[4.32049387,6.89585371],[4.41092079,6.07705779],[4.24397878,5.0194464],[3.90313886,4.0641845],[3.33970959,3.4728319],[2.81801582,3.56380922],[2.86670724,4.26888348],[3.12407617,4.73514227],[3.63881402,5.25826188],[4.16050778,5.45158869],[4.7448048,5.47433302],[5.1969394,5.16728455],[5.63516216,4.59867628],[6.01078167,4.00732367],[6.0246935,3.06343394],[5.73950091,2.1536607]])
    waypts = np.array([[1.90679071,7.94209293],[2.21285106,7.68053313],[2.49804365,7.38485682],[2.74150074,7.16878568],[2.99886966,6.90722587],[3.23537084,6.57743307],[3.40231284,6.03156913],[3.36057734,5.12179589],[3.06842883,4.23476698],[2.65802974,3.87085769],[2.28936614,4.18927832],[2.44239631,4.86023609],[2.87366316,5.37198353],[3.37448917,5.62217117],[3.75010869,5.54256602],[4.0422572,5.39472786],[4.38309712,5.07630723],[4.62655421,4.81474742],[4.87696722,4.51907112],[5.21085123,4.22339482]])

    #determine bspline for graphical input
    input_bspline = generate_bspline(traj_bspline.shape[0],waypts,per_flag=False)
   
    #approximate bspline using endpoints of graphical input bspline
    apprx_xy = np.array([input_bspline[0],input_bspline[1],input_bspline[-2],input_bspline[-1]])
    apprx_bspline = generate_bspline(traj_bspline.shape[0]/no_of_prds,apprx_xy,per_flag=False)

    input_bspline_center, dists_frm_apprx = contour_center_n_dists_frm_apprx(input_bspline,apprx_bspline)

    #obtain "central" bspline by translating approximate bspline in the direction of the center of graphical input bspline
    shftd_apprx_bspline = apprx_bspline + (input_bspline_center - apprx_bspline[np.argmin(dists_frm_apprx)])

    #for manual implementation of constrained transformation using pseudoinverse
    M = np.zeros((shftd_apprx_bspline.shape[0]*2,4))
    for n in range(0,M.shape[0],2):
        M[n,:] = np.hstack((shftd_apprx_bspline[n/2,:],np.ones(1),np.zeros(1)))
        M[n+1,:] = np.hstack((shftd_apprx_bspline[n/2,-1],-1.0*shftd_apprx_bspline[n/2,0],np.zeros(1),np.ones(1)))

    trnsfrmd_apprx_bspline = np.zeros(traj_bspline.shape)
    trnsfrmd_waypts = np.zeros((waypts.shape[0]*(traj_bspline.shape[0]/shftd_apprx_bspline.shape[0]),waypts.shape[1]))
    j_range = range(0,trnsfrmd_apprx_bspline.shape[0],shftd_apprx_bspline.shape[0])
    k_range = range(0,trnsfrmd_waypts.shape[0],waypts.shape[0])
    for j,k in zip(j_range,k_range):

        A_r1 = np.hstack((np.cos(0),-np.sin(0),np.zeros(1)))
        A_r2 = np.hstack((np.sin(0),np.cos(0),np.zeros(1)))
        A_r1r2 = np.vstack((A_r1,A_r2))
        A_r3 = np.hstack((np.zeros(2),np.ones(1)))

        A_r1r2 = cv2.estimateRigidTransform(shftd_apprx_bspline.reshape(1,-1,2), traj_bspline[j:j+shftd_apprx_bspline.shape[0],:].reshape(1,-1,2),fullAffine=False)
        if A_r1r2 is None:
            print('Failed to estimate rigid transformation, trying manual implementation')
            print(j)
            print(k)
            A_elems = np.dot(np.linalg.pinv(M),traj_bspline[j:j+shftd_apprx_bspline.shape[0],:].reshape(-1,1))

            #shear, coordinate flipping and non-uniform scaling constrained
            A_r1 = A_elems[0:3].reshape(-1)
            A_r2 = np.hstack((-1.0*A_elems[1],A_elems[0],A_elems[3]))
            A_r1r2 = np.vstack((A_r1,A_r2))

        A_mat = np.vstack((A_r1r2,A_r3))
        trnsfrmd_apprx_bspline[j:j+shftd_apprx_bspline.shape[0],:] = cv2.transform(shftd_apprx_bspline.reshape(1,-1,2),A_mat[0:2])[0]
        trnsfrmd_waypts[k:k+waypts.shape[0],:] = cv2.transform(waypts.reshape(1,-1,2),A_mat[0:2])[0]
        ##############################################

    #smooth countour for shape matching, 
    #WARNING: too low cutoff frequency causes start and end of contour to disconnect
    trnsfrmd_waypts_sgnl_fltd = smooth_contours_w_butterworth(trnsfrmd_waypts,5,0.4,'low')
    trnsfrmd_waypts_fltd = np.array(zip(list(trnsfrmd_waypts_sgnl_fltd.real),list(trnsfrmd_waypts_sgnl_fltd.imag)))

    pttrn_bspline = generate_bspline(2*traj_bspline.shape[0],trnsfrmd_waypts_fltd,per_flag=True)

    # cont_pt_by_pt(trnsfrmd_apprx_bspline,'test',color_crppd,(0,255,255),1)
    cont_pt_by_pt(pttrn_bspline,'test',color_crppd,(255,255,0),1)

    # cv2.imshow('Color gray',color_gray)
    # cv2.imshow('Depth gray',depth_gray)
    cv2.imshow('Color canny',color_canny)
    cv2.imshow('Color cropped',color_crppd)
    cv2.imshow('Depth cropped',depth_crppd)
    cv2.waitKey(0)

    #plot displacement as scatter to make it easy to determine if enough trajectory points visually
    dt = 0.1
    # dx = traj_bspline.reshape(-1,2)[:,0]
    # dy = traj_bspline.reshape(-1,2)[:,1]
    dx = pttrn_bspline.reshape(-1,2)[:,0]
    dy = pttrn_bspline.reshape(-1,2)[:,1]
    print(np.max(np.diff(dx)))
    print(np.max(np.diff(dy)))
    fig1 = plt.figure()
    ax1 = set_up_2dplot(fig1,'Secs','Displacement',20,12)
    ax1.scatter(np.arange(0,dt*len(dx),dt),dx)
    ax1.scatter(np.arange(0,dt*len(dy),dt),dy)
    ax1.grid()

    vel_x = np.gradient(dx,dt)
    vel_y = np.gradient(dy,dt)
    fig2 = plt.figure()
    ax2 = set_up_2dplot(fig2,'Secs','Vel',20,12)
    ax2.plot(np.arange(0,dt*len(vel_x),dt),vel_x)
    ax2.plot(np.arange(0,dt*len(vel_y),dt),vel_y)
    ax2.grid()

    acc_x = np.gradient(vel_x,dt)
    acc_y = np.gradient(vel_y,dt)
    fig3 = plt.figure()
    ax3 = set_up_2dplot(fig3,'Secs','Acc',20,12)
    ax3.plot(np.arange(0,dt*len(acc_x),dt),acc_x)
    ax3.plot(np.arange(0,dt*len(acc_y),dt),acc_y)
    ax3.grid()
    plt.show()

if __name__ == "__main__":
    main()
