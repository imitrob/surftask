# import ipdb
import os

import numpy as np
from scipy import interpolate
import collections
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from copy import deepcopy

def load_sgmnts_frm_dat(filename,sgmnt_len):
    time_arr,x_arr,y_arr,z_arr = np.loadtxt(filename)

    #center trajectories around origin to remove bias from dataset
    x_arr -= np.mean(x_arr)
    y_arr -= np.mean(y_arr)
    z_arr -= np.mean(z_arr)

    sgmnt_lst = []
    strt_indx = 0
    for i in range(time_arr.shape[0]):
        if i == 0:
            traj_len = 0.0
        else:
            traj_len += np.linalg.norm(np.array([x_arr[i],y_arr[i],z_arr[i]]) - np.array([x_arr[i-1],y_arr[i-1],z_arr[i-1]]))

        if traj_len > sgmnt_len:
            sgmnt_lst.append([traj_len,time_arr[strt_indx:i],x_arr[strt_indx:i],y_arr[strt_indx:i],z_arr[strt_indx:i]])
            i += 1
            strt_indx = i
            traj_len = 0.0

    return sgmnt_lst

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

def main():

    path = 'ML_datfiles/'
    dir_lst = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path,f))]
    file_nstd_lst = []
    for dir_str in dir_lst:
        file_nstd_lst.append([f for f in os.listdir(path+dir_str) if os.path.isfile(os.path.join(path+dir_str,f))])

    fig3 = plt.figure(3)
    ax3 = set_up_plt(fig3,36,20,'x','z','y')

    namedTup = collections.namedtuple('namedTup','traj_len time x y z label')
    data_dict = {}
    for i in range(len(dir_lst)):
        newpath = path + dir_lst[i] + '/'
        data_dict[dir_lst[i]] = {}

        for file_str in file_nstd_lst[i]:
            # if '21' in file_str:
                # print(file_str)
                # break
            if '1000' in file_str or '21' in file_str:
                print(file_str)
                data = np.loadtxt(newpath+file_str)
                # time = data[0]
                x = data[1] - np.mean(data[1])
                y = data[2] - np.mean(data[2])
                z = data[3] - np.mean(data[3])
                mask = np.where(np.abs(np.diff(x)) + np.abs(np.diff(y)) + np.abs(np.diff(z)) > 0)
                x = np.r_[x[mask], x[-1]]
                y = np.r_[y[mask], y[-1]]
                z = np.r_[z[mask], z[-1]]
                for j in range(x.shape[0]):
                    if j == 0:
                        traj_len = 0.0
                    else:
                        traj_len += np.linalg.norm(np.array([x[j],y[j],z[j]]) - np.array([x[j-1],y[j-1],z[j-1]]))
                print(traj_len)

                tck_params,xyz_as_u = interpolate.splprep([x,y,z],s=0)
                u_interp = np.linspace(xyz_as_u.min(),xyz_as_u.max(), 2000)
                input_bspl_3d = np.transpose(np.array(interpolate.splev(u_interp, tck_params)))
                if '1000' in file_str:
                    ax3.scatter(input_bspl_3d[:,0],input_bspl_3d[:,2],input_bspl_3d[:,1],s=1,color='b')
                if '21' in file_str:
                    ax3.scatter(input_bspl_3d[:,0],input_bspl_3d[:,2],input_bspl_3d[:,1],s=1,color='r')

        # if '21' in file_str:
            # break

    set_axes_equal(ax3)
    ax3.set_aspect('equal', adjustable='box')
    ax3.grid()
    fig3.set_tight_layout({'pad':1.0})
    fig3.set_size_inches(10,12)
    plt.show()

    data = np.loadtxt(newpath+file_str)
    x = data[1]
    y = data[2]
    z = data[3]
    # # https://stackoverflow.com/questions/47948453/scipy-interpolate-splprep-error-invalid-inputs
    #elements of xyz must not be equal to each other
    mask = np.where(np.abs(np.diff(x)) + np.abs(np.diff(y)) + np.abs(np.diff(z)) > 0)
    x = np.r_[x[mask], x[-1]]
    y = np.r_[y[mask], y[-1]]
    z = np.r_[z[mask], z[-1]]
    # JB: can be more efficient with numpy
    for j in range(x.shape[0]):
        if j == 0:
            traj_len = 0.0
        else:
            traj_len += np.linalg.norm(np.array([x[j],y[j],z[j]]) - np.array([x[j-1],y[j-1],z[j-1]]))

    fig1 = plt.figure()
    ax1 = set_up_plt(fig1,36,20,'x','z')
    ax1.scatter(x,z,s=1,color='k')
    ax1.set_xlim(np.min(x)-0.05,np.max(x)+0.05)
    ax1.set_ylim(np.min(z)-0.05,np.max(z)+0.05)
    ax1.set_aspect('equal')
    ax1.grid()
    fig1.set_tight_layout({'pad':1.0})

    no_of_pts = x.shape[0]
    print(no_of_pts)

    no_of_waypts_2d = input('Please input number of waypoints\n')
    waypts_2d = []
    while len(waypts_2d) < no_of_waypts_2d:
        print('Input waypoints with cursor')
        waypts_2d = np.asarray(plt.ginput(no_of_waypts_2d,timeout=-1)).reshape(-1,2)
    mask = np.where(np.abs(np.diff(waypts_2d[:,0])) + np.abs(np.diff(waypts_2d[:,1])) > 0)
    waypts_2d = np.r_[waypts_2d[mask],waypts_2d[-1].reshape(1,waypts_2d.shape[1])]
    input_bspl_2d = generate_bspline(no_of_pts,waypts_2d,per_flag=False)

    y_mu = np.mean(y)
    y_sigma = np.abs(np.abs(y_mu) - np.max(np.abs(y)))/3  # JB: is this way of calculating the std deviation correct? why is the z component generated?
    waypts_3d = np.hstack((waypts_2d,np.random.normal(y_mu,y_sigma,(waypts_2d.shape[0],1))))

    tck_params,xyz_as_u = interpolate.splprep([waypts_3d[:,0],waypts_3d[:,1],waypts_3d[:,2]],s=0)
    u_interp = np.linspace(xyz_as_u.min(),xyz_as_u.max(), no_of_pts)
    input_bspl_3d = np.transpose(np.array(interpolate.splev(u_interp, tck_params)))

    ax1.scatter(input_bspl_2d[:,0],input_bspl_2d[:,1],s=1,color='b')
    ax1.scatter(waypts_2d[:,0],waypts_2d[:,1],marker='+',color='r')

    fig2 = plt.figure()
    ax2 = set_up_plt(fig2,36,20,'x','z','y')
    ax2.scatter(x,z,y,s=1,color='k')
    ax2.scatter(input_bspl_3d[:,0],input_bspl_3d[:,1],input_bspl_3d[:,2],s=1,color='b')
    ax2.scatter(waypts_3d[:,0],waypts_3d[:,1],waypts_3d[:,2],marker='+',color='r')
    set_axes_equal(ax2)
    ax2.set_aspect('equal', adjustable='box')
    ax2.grid()
    fig2.set_tight_layout({'pad':1.0})
    fig2.set_size_inches(10,12)

    plt.show()

    file_name = raw_input('Saving as .dat file, please input file name\n')
    np.savetxt('ML_datfiles/synthesized/'+file_name+'.dat',np.array((np.zeros((waypts_3d.shape[0],)),waypts_3d[:,0],waypts_3d[:,2],waypts_3d[:,1])))

if __name__ == "__main__":
	main()
