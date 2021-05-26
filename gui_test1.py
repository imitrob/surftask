import collections
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import signal, sparse, interpolate

import cv2
from cv_bridge import CvBridge

import rosbag
import rospy
import sensor_msgs.point_cloud2

from PyQt5 import QtCore, QtWidgets, QtGui
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT

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
    # outbag = rosbag.Bag(filename+'_new','w')

    # topic_name == '/XTION3/camera/depth_registered/points':
    # no_of_msgs = bag.get_message_count(topic_name)

    time = None
    x_arr = None
    y_arr = None
    z_arr = None
    #no need to record time for images, assume static image so use flag to bypass all other images
    depth_flag = False
    color_img_flag = False
    i = 0
    for topic, msg, t in bag.read_messages():

        if topic == '/end_point':
            if i == 0:
                time = np.zeros([bag.get_message_count(topic)])
                x_arr = time.copy()
                y_arr = time.copy()
                z_arr = time.copy()
                t_offset = t.to_nsec()

            time[i] = (t.to_nsec() - t_offset)*1e-9
            x_arr[i] = msg.pose.position.x
            y_arr[i] = msg.pose.position.y
            z_arr[i] = msg.pose.position.z
            i += 1

        elif topic == '/XTION3/camera/depth/image_rect' or topic ==  '/XTION3/camera/depth/image_rect/' and depth_flag == False:
            depth_msg = msg
            # outbag.write(topic,msg)
            depth_flag = True

        elif topic == '/XTION3/camera/rgb/image_rect_color/compressed' and color_img_flag == False:
            color_img_msg = msg
            # outbag.write(topic,msg)
            color_img_flag = True

    bag.close()
    # outbag.close()

    return (time,x_arr,y_arr,z_arr,depth_msg,color_img_msg)

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
    u_interp = np.linspace(xy_as_u.min(),xy_as_u.max(), int(no_of_pts))
    
    return np.transpose(np.array(interpolate.splev(u_interp, tck_params)))

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
    """Sets the font sizes and labels of a 2D plot if z_label is None or a 3D plot is z_label is specified

    Parameters
    ----------
    fig		        : matplotlib.figure.Figure, required
                            Figure instance for plotting
    label_font_size	: int, required
                            Font size of x and y labels
    tick_font_size	: int, required
                            Font size of values on x and y axes
    x_label	        : string, required
                            Label of x-axis
    y_label	        : string, required
                            Label of y-axis
    z_label	        : string, optional
                            Label of z-axis

    Outputs
    ----------
    ax                  : matplotlib.axes.SubplotBase
                            Axes instance for plotting

    """
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

class Vision():

    def cnvrtToCV2(self,data_named_tup):

        # # https://stackoverflow.com/questions/47751323/get-depth-image-in-grayscale-in-ros-with-imgmsg-to-cv2-python
        bridge = CvBridge()

        #load as opencv images from ROS msg
        depth_img = bridge.imgmsg_to_cv2(data_named_tup.depth_msg, "passthrough").copy()
        color_img = bridge.compressed_imgmsg_to_cv2(data_named_tup.color_img_msg, "passthrough").copy()

        self._depth_crppd,self._color_crppd = crop_and_fill_nans(depth_img,color_img,border_offset=30)

    # # https://www.programiz.com/python-programming/property
    @property
    def depth_crppd(self):
        return(self._depth_crppd)

    @property
    def color_crppd(self):
        return(self._color_crppd)

    def cnvrtToGray(self,color_crppd,depth_crppd):

        # # https://stackoverflow.com/questions/46689428/convert-np-array-of-type-float64-to-type-uint8-scaling-values
        #convert to grayscale image since Canny works only on uint8
        self._color_gray = cv2.cvtColor(color_crppd,cv2.COLOR_BGR2GRAY)
        #can't convert float32 to uint8 using cvtColor so normalize then scale to (0-255)
        depth_gray = cv2.normalize(depth_crppd,None,0.0,1.0,cv2.NORM_MINMAX)
        self._depth_gray = cv2.convertScaleAbs(depth_gray,None,np.iinfo(self.color_gray.dtype).max)

    @property
    def color_gray(self):
        return(self._color_gray)

    @property
    def depth_gray(self):
        return(self._depth_gray)

    def cnvrtToQpixmap(self,color_crppd,depth_crppd):

        # # https://stackoverflow.com/questions/34232632/convert-python-opencv-image-numpy-array-to-pyqt-qpixmap-image
        height, width, channel = color_crppd.shape
        bytes_per_line = 3 * width
        # https://stackoverflow.com/questions/48639185/pyqt5-qimage-from-numpy-array
        q_img = QtGui.QImage(color_crppd.copy().data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888).rgbSwapped()
        self._q_pixmap_color = QtGui.QPixmap.fromImage(q_img)

        self.cnvrtToGray(color_crppd,depth_crppd)
        #can't convert float32 to uint8 using cvtColor so normalize then scale to (0-255)
        height, width = self._depth_gray.shape
        bytes_per_line = 1 * width
        q_img = QtGui.QImage(self._depth_gray.copy().data, width, height, bytes_per_line, QtGui.QImage.Format_Grayscale8)
        self._q_pixmap_depth = QtGui.QPixmap.fromImage(q_img)

    @property
    def q_pixmap_color(self):
        return(self._q_pixmap_color)

    @property
    def q_pixmap_depth(self):
        return(self._q_pixmap_depth)

    def dtrmn2d(self,color_gray,depth_gray):

        # # https://stackoverflow.com/questions/4292249/automatic-calculation-of-low-and-high-thresholds-for-the-canny-operation-in-open
        #determine thresholds for canny edge detection automatically
        thresh_color,color_thrshd = cv2.threshold(color_gray,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)
        color_canny = cv2.Canny(self._color_gray,0.5*thresh_color,thresh_color)

        color_dltd = cv2.dilate(color_canny,None,iterations=1)
        color_cont_img, cont_color, hierachy_color = cv2.findContours(color_dltd,cv2.RETR_TREE,cv2.CHAIN_APPROX_TC89_KCOS)

        #since a frame should have a set of 2 contours, inner and outer
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
                # print(i)
                if i > 100:
                    print len(cont_combi)
                    break

                combi_clsd = cv2.morphologyEx(combi_canny,cv2.MORPH_CLOSE,None,iterations=i)
                combi_cont_img, cont_combi, hiearachy_combi = cv2.findContours(combi_clsd,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

            #smooth jagged edges of solid shape using median filtering
            combi_hires = cv2.pyrUp(combi_clsd)
            combi_hires = cv2.medianBlur(combi_hires,55)
            combi_medblrrd = cv2.pyrDown(combi_hires)
            combi_cont_img, cont_combi, hiearachy_combi = cv2.findContours(combi_medblrrd,cv2.RETR_TREE,cv2.CHAIN_APPROX_TC89_KCOS)

            cont = cont_combi[0]

        #connect the end of the contour to the beginning
        self._cont = np.vstack((cont.reshape(-1,2),cont.reshape(-1,2)[0,:]))

    @property
    def cont(self):
        return(self._cont)

    def scaleCntr(self,cont,scaling_factor):

        m = cv2.moments(cont)
        cont_center = np.array([int(m['m10']/m['m00']),int(m['m01']/m['m00'])])
        self._cont_scaled = cont.reshape(-1,2).astype(np.float64)
        for coord in self._cont_scaled:
            coord -= cont_center
            coord *= scaling_factor
            coord += cont_center

    @property
    def cont_scaled(self):
        return(self._cont_scaled)

    def smthCntrWBttrwrth(self,cont_scaled,fltr_ordr,norm_cutoff_freq,fltr_type_str):

        # # https://stackoverflow.com/questions/51259361/smooth-a-bumpy-circle/51267877
        #filtering contour with butterworth filter so its not so jagged
        cont_coord = np.array(cont_scaled).reshape((-1,2))
        x_coord = np.array([coord[0] for coord in cont_coord])
        y_coord = np.array([coord[1] for coord in cont_coord])
        cmplx_sgnl = x_coord + 1.0j*y_coord

        # # https://dsp.stackexchange.com/questions/49460/apply-low-pass-butterworth-filter-in-python
        b,a = signal.butter(fltr_ordr,norm_cutoff_freq,fltr_type_str)
        cont_sgnl_fltd = signal.filtfilt(b,a,cmplx_sgnl)
        self._cont_fltd = np.array(zip(list(cont_sgnl_fltd.real),list(cont_sgnl_fltd.imag)))

    @property
    def cont_fltd(self):
        return(self._cont_fltd)

class Trajectory():

    def genBspline(self,cont_fltd,no_of_pts,per_flag):

        x = cont_fltd.reshape(-1,2)[:,0]
        y = cont_fltd.reshape(-1,2)[:,1]
        if per_flag is True:
            tck_params,xy_as_u = interpolate.splprep([x,y],s=0,per=1,quiet=2)
        elif per_flag is False:
            tck_params,xy_as_u = interpolate.splprep([x,y],s=0,quiet=2)
        u_interp = np.linspace(xy_as_u.min(),xy_as_u.max(), int(no_of_pts))
        
        self._trj_bspl = np.transpose(np.array(interpolate.splev(u_interp, tck_params)))

    @property
    def trj_bspl(self):
        return(self._trj_bspl)

    def genSine(self,trj_bspl,amp,no_of_prds):

        pts_per_prd = np.round(trj_bspl.shape[0]/no_of_prds)
        x = trj_bspl.reshape(-1,2)[:,0]
        y = trj_bspl.reshape(-1,2)[:,1]
        self._sin_trj = np.zeros((x.shape[0],2))
        for j in range(x.shape[0]):
            if j < x.shape[0]-1:
                unit_perp = np.array([-(y[j+1]-y[j]),x[j+1]-x[j]]/np.linalg.norm(trj_bspl[j+1] - trj_bspl[j]))
            else:
                unit_perp = np.array([-(2*y[j]-y[j]),2*x[j]-x[j]]/np.linalg.norm(2*trj_bspl[j] - trj_bspl[j]))
            self._sin_trj[j] = trj_bspl.reshape(-1,2)[j] + amp*np.sin(j*(2*np.pi/pts_per_prd))*unit_perp

    @property
    def sin_trj(self):
        return(self._sin_trj)


class QWin(QtWidgets.QMainWindow):

    def __init__(self,*args,**kwargs):
        super(QWin,self).__init__(*args,**kwargs)

        vis_object = Vision()
        trj_object = Trajectory()

        # # https://stackoverflow.com/questions/12459811/how-to-embed-matplotlib-in-pyqt-for-dummies
        self._endpt_fig = plt.figure(figsize=(4,4))
        self._cnvs_endpt = FigureCanvasQTAgg(self._endpt_fig)
        # self.toolbar = NavigationToolbar2QT(self.canvas,self)

        bttn_endpt = QtWidgets.QPushButton('1. Plot endpoints from bagfile',self)
        bttn_endpt.clicked.connect(self.pltEndpts)

        # # https://micropyramid.com/blog/understand-self-and-__init__-method-in-python-class/
        self._label_clr = QtWidgets.QLabel()
        self._label_clr.setAlignment(QtCore.Qt.AlignCenter)

        # # https://stackoverflow.com/questions/38090345/pass-extra-arguments-to-pyqt-slot-without-losing-default-signal-arguments
        bttn_clr = QtWidgets.QPushButton('2. Load RGB from bagfile',self)
        bttn_clr.clicked.connect(lambda: self.dsplyColor(vis_object,trj_object))

        self._3d_fig = plt.figure(figsize=(4,4))
        self._cnvs_3d = FigureCanvasQTAgg(self._3d_fig)

        bttn_3d = QtWidgets.QPushButton('4. Determine 3d trajectory',self)
        # bttn_3d.clicked.connect(lambda: self.dsply2d(vis_object))
        bttn_3d.setDisabled(True)

        self._label_2d = QtWidgets.QLabel()
        self._label_2d.setAlignment(QtCore.Qt.AlignCenter)

        self._bttn_2d = QtWidgets.QPushButton('3. Determine 2d trajectory',self)
        self._bttn_2d.clicked.connect(lambda: self.dsply2d(vis_object,trj_object))
        self._bttn_2d.setDisabled(True)

        #############################################################

        layout_man = QtWidgets.QHBoxLayout()
        layout_mode = QtWidgets.QVBoxLayout()

        self._radio_ml = QtWidgets.QRadioButton('Supervised learning mode')
        self._radio_ml.setDisabled(True)
        self._radio_man = QtWidgets.QRadioButton('Manual mode')
        self._radio_man.setChecked(True)
        self._radio_man.setDisabled(True)
        self._combo_func = QtWidgets.QComboBox()
        self._combo_func.addItems(['Line','Sine','Dot','Dash'])
        self._combo_func.setFixedWidth(120)
        self._combo_func.setDisabled(True)
        self._combo_func.activated.connect(lambda: self.bsplineCallback(vis_object,trj_object))

        layout_man.addWidget(self._radio_man)
        layout_man.addWidget(self._combo_func)

        layout_mode.addWidget(self._radio_ml)
        layout_mode.addLayout(layout_man)

        self._line_edit_amp = QtWidgets.QLineEdit()
        self._line_edit_amp.setValidator(QtGui.QRegExpValidator(QtCore.QRegExp('^([1-4][.][0-9]|[5][.][0])$')))
        self._sldr_amp = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        callback_lst = [self.ampEditCallback,self.ampSldrCallback]
        arg_lst = [vis_object,trj_object]
        self.sliderTmplt(layout_mode,'Amplitude',self._line_edit_amp,self._sldr_amp,10,50,30,callback_lst,arg_lst)

        self._line_edit_no_of_prds = QtWidgets.QLineEdit()
        self._line_edit_no_of_prds.setValidator(QtGui.QRegExpValidator(QtCore.QRegExp('^([1-2][0-9]|[3][0])$')))
        self._sldr_no_of_prds = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        callback_lst = [self.periodEditCallback,self.periodSldrCallback]
        arg_lst = [vis_object,trj_object]
        self.sliderTmplt(layout_mode,'Number of periods',self._line_edit_no_of_prds,self._sldr_no_of_prds,5,15,10,callback_lst,arg_lst)


        #############################################################

        layout_sldr = QtWidgets.QVBoxLayout()

        self._line_edit_scale = QtWidgets.QLineEdit()
        self._line_edit_scale.setValidator(QtGui.QRegExpValidator(QtCore.QRegExp('^([0][.][5-9][0-9]|[1][.][0-5][0-9])$')))
        self._sldr_scale = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        callback_lst = [self.scalerEditCallback,self.scalerSldrCallback]
        arg_lst = [vis_object,trj_object]
        self.sliderTmplt(layout_sldr,'Scaler',self._line_edit_scale,self._sldr_scale,50,150,75,callback_lst,arg_lst)

        self._line_edit_filter = QtWidgets.QLineEdit()
        self._line_edit_filter.setValidator(QtGui.QRegExpValidator(QtCore.QRegExp('^([0][.][0-8][5-9]|[0][.][9][0-5])$')))
        self._sldr_filter = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        callback_lst = [self.filterEditCallback,self.filterSldrCallback]
        arg_lst = [vis_object,trj_object]
        self.sliderTmplt(layout_sldr,'Filter normalized cutoff frequency',self._line_edit_filter,self._sldr_filter,5,95,20,callback_lst,arg_lst)

        self._line_edit_no_of_pts = QtWidgets.QLineEdit()
        self._line_edit_no_of_pts.setValidator(QtGui.QRegExpValidator(QtCore.QRegExp('^([1-9][0][0]|[1][0][0][0])$')))
        self._sldr_no_of_pts = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        callback_lst = [self.ptsEditCallback,self.ptsSldrCallback]
        arg_lst = [vis_object,trj_object]
        self.sliderTmplt(layout_sldr,'Number of points in bspline',self._line_edit_no_of_pts,self._sldr_no_of_pts,1,10,1,callback_lst,arg_lst)

        #############################################################

        layout_main = QtWidgets.QGridLayout()
        layout_main.addWidget(self._cnvs_endpt,0,0)
        layout_main.addWidget(bttn_endpt,1,0)
        layout_main.addWidget(self._label_clr,2,0)
        layout_main.addWidget(bttn_clr,3,0)
        layout_main.addWidget(self._cnvs_3d,0,1)
        layout_main.addWidget(bttn_3d,1,1)
        layout_main.addWidget(self._label_2d,2,1)
        layout_main.addWidget(self._bttn_2d,3,1)
        layout_main.addLayout(layout_mode,0,2)
        layout_main.addLayout(layout_sldr,2,2) 
        # q_layout.addWidget(self.toolbar,1,0)

        q_widget = QtWidgets.QWidget()
        q_widget.setLayout(layout_main)

        # q_toolbar = QtWidgets.QToolBar('test')
        # q_toolbar.addAction(button_img)

        # self.addToolBar(q_toolbar)
        self.setCentralWidget(q_widget)

    def setTitle(self,title_str):
        self.setWindowTitle(title_str)

    def setWindowSize(self,width,height):
        self.resize(width,height)

    def sliderTmplt(self,outer_layout,label_str,q_line_edit,q_sldr,min_val,max_val,init_val,callback_mthd_lst,callback_arg_lst):

        q_label = QtWidgets.QLabel(label_str) 
        q_label.setAlignment(QtCore.Qt.AlignCenter)

        q_line_edit.setAlignment(QtCore.Qt.AlignCenter)
        q_line_edit.setFixedWidth(120)
        q_line_edit.setText(str(init_val))
        q_line_edit.editingFinished.connect(lambda: callback_mthd_lst[0](callback_arg_lst[0],callback_arg_lst[1]))
        q_line_edit.setDisabled(True)

        q_sldr.setMinimum(min_val)
        q_sldr.setMaximum(max_val)
        q_sldr.setValue(init_val)
        q_sldr.valueChanged.connect(lambda: callback_mthd_lst[1](callback_arg_lst[0],callback_arg_lst[1]))
        q_sldr.setDisabled(True)

        layout_top = QtWidgets.QHBoxLayout()
        layout_top.addWidget(q_label)
        layout_top.addWidget(q_line_edit)

        layout_bttm = QtWidgets.QVBoxLayout()
        layout_bttm.addLayout(layout_top)
        layout_bttm.addWidget(q_sldr)

        outer_layout.addLayout(layout_bttm)

    def loadData(self):

        q_str_fname = QtWidgets.QFileDialog.getOpenFileName(self,'Open bag file','','Rosbag files (*.bag)')
        namedTup = collections.namedtuple('namedTup','time x y z depth_msg color_img_msg')
        data = load_frm_bag(str(q_str_fname[0]))
        self._data_named_tup = namedTup(data[0],data[1],data[2],data[3],data[4],data[5])

    @property
    def data_named_tup(self):
        return(self._data_named_tup)

    def pltEndpts(self):

        self.loadData()
        self._endpt_fig.clear()
        ax = set_up_plt(self._endpt_fig,20,12,'x','z','y')
        ax.scatter(self._data_named_tup.x,self._data_named_tup.z,self._data_named_tup.y)
        set_axes_equal(ax)
        ax.set_aspect('equal',adjustable='box')
        ax.grid()
        ax.view_init(55,-45)
        self._endpt_fig.subplots_adjust(left=-0.1,right=1,bottom=0.05,top=0.95)
        self._cnvs_endpt.draw()

    def dsplyColor(self,vis_obj,trj_obj):
       
        self.loadData()
        vis_obj.cnvrtToCV2(self._data_named_tup)
        vis_obj.cnvrtToQpixmap(vis_obj.color_crppd,vis_obj.depth_crppd)
        self._label_clr.setPixmap(vis_obj.q_pixmap_color)
        self._bttn_2d.setEnabled(True)

    #should only be used after dsplyColor() has been executed
    def dsply2d(self,vis_obj,trj_obj):

        vis_obj.cnvrtToGray(vis_obj.color_crppd,vis_obj.depth_crppd)
        vis_obj.dtrmn2d(vis_obj.color_gray,vis_obj.depth_gray)

        self._sldr_scale.setEnabled(True)
        self._line_edit_scale.setEnabled(True)
        self._sldr_filter.setEnabled(True)
        self._line_edit_filter.setEnabled(True)
        self._sldr_no_of_pts.setEnabled(True)
        self._line_edit_no_of_pts.setEnabled(True)

        self._radio_ml.setEnabled(True)
        self._radio_man.setEnabled(True)
        if self._radio_man.isChecked():
            self._combo_func.setEnabled(True)

        self.scalerSldrCallback(vis_obj,trj_obj)

    #should only be used after dsply2d() has been executed
    def scalerEditCallback(self,vis_obj,trj_obj):

        self._sldr_scale.setValue(float(self._line_edit_scale.text())*100)
        self.scalerSldrCallback(vis_obj,trj_obj)

    def scalerSldrCallback(self,vis_obj,trj_obj):

        #need to divide since slider values only work with integers
        vis_obj.scaleCntr(vis_obj.cont,self._sldr_scale.value()/100.0) 
        self._line_edit_scale.setText(str(self._sldr_scale.value()/100.0))
        self.filterSldrCallback(vis_obj,trj_obj)

    def filterEditCallback(self,vis_obj,trj_obj):

        self._sldr_filter.setValue(float(self._line_edit_filter.text())*100)
        self.filterSldrCallback(vis_obj,trj_obj)

    def filterSldrCallback(self,vis_obj,trj_obj):
        
        vis_obj.smthCntrWBttrwrth(vis_obj.cont_scaled,5,self._sldr_filter.value()/100.0,'low')
        self._line_edit_filter.setText(str(self._sldr_filter.value()/100.0))
        self.ptsSldrCallback(vis_obj,trj_obj)

    def ptsEditCallback(self,vis_obj,trj_obj):

        self._sldr_no_of_pts.setValue(int(self._line_edit_no_of_pts.text())/100)
        self.ptsSldrCallback(vis_obj,trj_obj)

    def ptsSldrCallback(self,vis_obj,trj_obj):

        self._line_edit_no_of_pts.setText(str(self._sldr_no_of_pts.value()*100))
        self.bsplineCallback(vis_obj,trj_obj)

    def bsplineCallback(self,vis_obj,trj_obj):

        if str(self._combo_func.currentText()) != 'Line':
            self._sldr_amp.setEnabled(True)
            self._line_edit_amp.setEnabled(True)
            self._sldr_no_of_prds.setEnabled(True)
            self._line_edit_no_of_prds.setEnabled(True)
        else:
            self._sldr_amp.setDisabled(True)
            self._line_edit_amp.setDisabled(True)
            self._sldr_no_of_prds.setDisabled(True)
            self._line_edit_no_of_prds.setDisabled(True)

        trj_obj.genBspline(vis_obj.cont_fltd,self._sldr_no_of_pts.value()*100,per_flag=True)

        if str(self._combo_func.currentText()) == 'Sine':
            trj_obj.genSine(trj_obj.trj_bspl,self._sldr_amp.value()/10.0,self._sldr_no_of_prds.value()*2)
            trj_obj.genBspline(trj_obj.sin_trj,self._sldr_no_of_pts.value()*100,per_flag=True)

        color_traj = vis_obj.color_crppd.copy()
        for coord in trj_obj.trj_bspl.reshape(-1,2):
            color_traj[coord[1].astype(np.int),coord[0].astype(np.int)] = (0,255,0)

        vis_obj.cnvrtToQpixmap(color_traj,vis_obj.depth_crppd)
        self._label_2d.setPixmap(vis_obj.q_pixmap_color)

    def ampEditCallback(self,vis_obj,trj_obj):

        self._sldr_amp.setValue(float(self._line_edit_amp.text())*10.0)
        self.ampSldrCallback(vis_obj,trj_obj)

    def ampSldrCallback(self,vis_obj,trj_obj):

        self._line_edit_amp.setText(str(self._sldr_amp.value()/10.0))
        self.bsplineCallback(vis_obj,trj_obj)

    def periodEditCallback(self,vis_obj,trj_obj):

        self._sldr_no_of_prds.setValue(int(self._line_edit_no_of_prds.text())/2)
        self.periodSldrCallback(vis_obj,trj_obj)

    def periodSldrCallback(self,vis_obj,trj_obj):

        self._line_edit_no_of_prds.setText(str(self._sldr_no_of_prds.value()*2))
        self.bsplineCallback(vis_obj,trj_obj)

def main():

    q_app = QtWidgets.QApplication([])
    q_win = QWin()
    q_win.setTitle('Trajectory generator')
    q_win.setWindowSize(1000,800)

    q_win.show()
    q_app.exec_()

if __name__ == "__main__":
	main()
