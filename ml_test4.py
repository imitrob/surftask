import os

import numpy as np
from scipy import interpolate
import collections
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from copy import deepcopy

from sklearn import metrics, preprocessing
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier

import ipdb

def search_params(clf_dict,param_dict,x_trn,y_trn):
    """Determines best parameters for a dict of classifiers via GridSearchCV and returns a tuned dict of classifiers"""
    clf_dict_tuned = dict()
    acc_dict = dict()
    print('Starting grid search...')
    for name, clf_handle in clf_dict.iteritems():
        print(clf_handle)
        clssfr = GridSearchCV(clf_handle, param_dict[name], scoring='accuracy', cv=5, iid=True)
        # clssfr = GridSearchCV(clf_handle, param_dict[name], scoring='accuracy', cv=[(slice(None),slice(None))], iid=True)
        clssfr.fit(x_trn, y_trn)

        clf_dict_tuned[name] = clssfr
        acc_dict[name] = clssfr.best_score_

    clf_dict_tuned = collections.OrderedDict(sorted(clf_dict_tuned.items()))
    return clf_dict_tuned, acc_dict

def load_sgmnts_frm_dat(filename,sgmnt_len):
    time_arr,x_arr,y_arr,z_arr = np.loadtxt(filename)

    #center trajectories at origin to remove bias from dataset
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

        if (traj_len > sgmnt_len) or (i == time_arr.shape[0]-1):
            sgmnt_lst.append([traj_len,time_arr[strt_indx:i],x_arr[strt_indx:i],y_arr[strt_indx:i],z_arr[strt_indx:i]])
            i += 1
            strt_indx = i
            traj_len = 0.0

    return sgmnt_lst

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
    dir_lst = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path,f)) and 'synthesized' not in f]
    file_nstd_lst = []
    for dir_str in dir_lst:
        file_nstd_lst.append([f for f in os.listdir(path+dir_str) if os.path.isfile(os.path.join(path+dir_str,f))])

    fig_a = plt.figure()
    ax_a = set_up_plt(fig_a,36,20,'Index','log_rho')
    fig_b = plt.figure()
    ax_b = set_up_plt(fig_b,36,20,'Index','log_mu')
    fig_c = plt.figure()
    ax_c = set_up_plt(fig_c,36,20,'Index','LCG')
    fig_d = plt.figure()
    ax_d = set_up_plt(fig_d,36,20,'Index','LTG')
    # fig_e = plt.figure()
    # ax_e = set_up_plt(fig_e,36,20,'x','z','y')
    # plt.ion()

    namedTup = collections.namedtuple('namedTup','traj_len time x y z l_dist g_dist crvtr trsn LCG LTG label')
    data_dict = {}
    for i in range(len(dir_lst)):
        newpath = path + dir_lst[i] + '/'
        data_dict[dir_lst[i]] = {}
        for file_str in file_nstd_lst[i]:

            lst_of_sgmnt_data = load_sgmnts_frm_dat(newpath+file_str,0.25)
            file_str = file_str[:-4]
            data_dict[dir_lst[i]][file_str] = {}
            j = 0

            for data in lst_of_sgmnt_data:

                x = data[2]
                y = data[3]
                z = data[4]
                # # https://stackoverflow.com/questions/47948453/scipy-interpolate-splprep-error-invalid-inputs
                #elements of xyz must not be equal to each other
                mask = np.where(np.abs(np.diff(x)) + np.abs(np.diff(y)) + np.abs(np.diff(z)) > 0)
                x = np.r_[x[mask], x[-1]]
                y = np.r_[y[mask], y[-1]]
                z = np.r_[z[mask], z[-1]]

                no_of_pts = int(1e2)
                tck_params, xyz_as_u = interpolate.splprep([x,y,z],s=1e-4)
                u_interp = np.linspace(xyz_as_u.min(),xyz_as_u.max(),no_of_pts)
                input_bspl = interpolate.splev(u_interp,tck_params)
                input_coord = np.array(list(zip(input_bspl[0],input_bspl[1],input_bspl[2])))

                lcl_cntr = np.mean(input_coord,axis=0)
                lcl_dist_ftrs = np.zeros(u_interp.shape)
                glb_dist_ftrs = np.zeros(u_interp.shape)
                coord_vel = np.zeros(input_coord.shape)
                coord_acc = np.zeros(input_coord.shape)
                for k in range(input_coord.shape[0]):
                    lcl_dist_ftrs[k] = np.linalg.norm(input_coord[k]-lcl_cntr)
                    #since center of trajectory should be at origin
                    glb_dist_ftrs[k] = np.linalg.norm(input_coord[k])
                    # # http://thomas.lewiner.org/pdfs/curvature_cg.pdf, based on method T4
                    if k > 0 and k < input_coord.shape[0]-1:
                        p0 = input_coord[k]
                        p_q = input_coord[k-1]
                        pq = input_coord[k+1]
                        pq_p0 = pq - p0
                        p0_p_q = p0 - p_q
                        pq_p_q = pq - p_q
                        p_q_p0 = p_q - p0
                        p0pq_norm = np.linalg.norm(p0-pq)
                        p_qp0_norm = np.linalg.norm(p_q-p0)
                        coord_vel[k] = (pq_p0/p0pq_norm) + (p0_p_q/p_qp0_norm) - (pq_p_q/(p_qp0_norm+p0pq_norm))
                        coord_acc[k] = 2*((p_q_p0*p0pq_norm + pq_p0*p_qp0_norm)/(p_qp0_norm*p0pq_norm*(p_qp0_norm+p0pq_norm)))
                coord_jrk = np.gradient(coord_acc,axis=0)
                coord_jrk[0] = np.zeros((1,input_coord.shape[1]))
                coord_jrk[-1] = np.zeros((1,input_coord.shape[1]))

                crvtr_ftrs = np.zeros(u_interp.shape)
                trsn_ftrs = np.zeros(u_interp.shape)
                for l in range(1,input_coord.shape[0]-1):
                    # # http://mathworld.wolfram.com/Curvature.html
                    crvtr_ftrs[l] = np.abs(np.linalg.norm(np.cross(coord_vel[l],coord_acc[l]))/np.linalg.norm(coord_vel[l])**3)
                    # # https://en.wikipedia.org/wiki/Torsion_of_a_curve
                    trsn_ftrs[l] = np.abs(np.dot(np.cross(coord_vel[l],coord_acc[l]),coord_jrk[l])/np.linalg.norm(np.cross(coord_vel[l],coord_acc[l]))**2)
                #scaled because some curvature and torsion values are relatively very large which increases training time
                crvtr_scld = crvtr_ftrs/np.max(crvtr_ftrs)
                trsn_scld = trsn_ftrs/np.max(trsn_ftrs)

                # # https://www.researchgate.net/publication/263581564_A_Note_on_3D_Curves 
                delta_crvtr = np.gradient(crvtr_ftrs)
                delta_trsn = np.gradient(trsn_ftrs)
                # #intializing with zeros sometimes cause problems with feature selection so resorting to intializing with very small values
                # log_rho = np.ones(u_interp.shape)*1e-16
                # log_mu = np.ones(u_interp.shape)*1e-16
                # LCG_ftrs = np.ones(u_interp.shape)*1e-16
                # LTG_ftrs = np.ones(u_interp.shape)*1e-16
                log_rho = np.zeros(u_interp.shape)
                log_mu = np.zeros(u_interp.shape)
                LCG_ftrs = np.zeros(u_interp.shape)
                LTG_ftrs = np.zeros(u_interp.shape)
                for m in range(1,input_coord.shape[0]-1):
                    if crvtr_ftrs[m] > 0:
                        # log_rho[m] = np.log(1/crvtr_ftrs[m])
                        CG = (1/crvtr_ftrs[m])*(np.linalg.norm(coord_vel[m])/delta_crvtr[m])
                        if CG > 0:
                            log_rho[m] = np.log(1/crvtr_ftrs[m])
                            LCG_ftrs[m] = np.log(CG)
                    if trsn_ftrs[m] > 0:
                        # log_mu[m] = np.log(1/trsn_ftrs[m])
                        TG = (1/trsn_ftrs[m])*(np.linalg.norm(coord_vel[m])/delta_trsn[m])
                        if TG > 0:
                            log_mu[m] = np.log(1/trsn_ftrs[m])
                            LTG_ftrs[m] = np.log(TG)
                # ipdb.set_trace()
                log_rho = log_rho[np.nonzero(log_rho)]
                log_mu = log_mu[np.nonzero(log_mu)]
                LCG_ftrs = LCG_ftrs[np.nonzero(LCG_ftrs)]
                LTG_ftrs = LTG_ftrs[np.nonzero(LTG_ftrs)]

                data_dict[dir_lst[i]][file_str]['sgmnt'+str(j)] = namedTup(data[0],data[1],input_bspl[0],input_bspl[1],input_bspl[2],lcl_dist_ftrs,glb_dist_ftrs,crvtr_scld,trsn_scld,log_rho,log_mu,LCG_ftrs)
                j += 1

                ax_a.plot(range(log_rho.shape[0]),log_rho)
                ax_b.plot(range(log_mu.shape[0]),log_mu)
                ax_c.plot(range(LCG_ftrs.shape[0]),LCG_ftrs)
                ax_d.plot(range(LTG_ftrs.shape[0]),LTG_ftrs)
                # ax_c.scatter(log_rho[np.nonzero(log_rho)],LCG_ftrs[np.nonzero(LCG_ftrs)])
                # ax_d.scatter(log_mu[np.nonzero(log_mu)],LTG_ftrs[np.nonzero(LTG_ftrs)])
                # ax_e.plot(range(trsn_trnsfrmd.shape[0]),trsn_trnsfrmd)
                # plt.show()
                # plt.waitforbuttonpress(timeout=-1)
                # raw_input()

        plt.show(block=True)
        # raw_input()
        # ax_a.cla()
        # ax_b.cla()
        # ax_c.cla()
    
    #remember to check whether segments are almost the same length
    data_lst = []
    label_lst = []
    for dir_str in data_dict.keys():
        for file_str in data_dict[dir_str].keys():
            for sgmnt in data_dict[dir_str][file_str].keys():
                dctnry = data_dict[dir_str][file_str][sgmnt]
                data_lst.append(list(zip(dctnry.x,dctnry.y,dctnry.z,dctnry.l_dist,dctnry.g_dist,dctnry.crvtr,dctnry.trsn,dctnry.LCG,dctnry.LTG)))
                label_lst.append(dctnry.label)
    
    # List of tuples of classifiers
    clf_dec_tree = DecisionTreeClassifier(max_features='sqrt')
    clf_knn = KNeighborsClassifier(algorithm='ball_tree')
    clf_lin_svc = SVC(kernel='linear')
    clf_rbf_svc = SVC(kernel='rbf')

    # List of tuples of name and classifier handles
    clf_lst = [ ('Decision Tree', clf_dec_tree),\
                ('K Nearest Neighbors', clf_knn),\
                ('SVC (Linear)', clf_lin_svc),\
                ('SVC (RBF)', clf_rbf_svc)]

    # Dictionaries of each classifier's parameters for grid search
    params_dec_tree = dict( min_samples_split = list(np.linspace(0.1,1.0,10)),\
                            min_samples_leaf  = list(np.linspace(0.01,0.5,10)) )
    params_knn = dict(  n_neighbors = list(np.linspace(1,20,10).astype(int)),\
                        leaf_size   = list(np.linspace(1,30,10).astype(int)) )
    params_lin_svc = dict(  C = list(np.linspace(1,10,10).astype(int)),\
                            cache_size = [1000] ) # # https://datascience.stackexchange.com/questions/989/svm-using-scikit-learn-runs-endlessly-and-never-completes-execution
    params_rbf_svc = dict(  C     = list(np.linspace(1,10,10).astype(int)),\
                            gamma = list(np.linspace(0,3,11)) )

    params_lst = [  ('Decision Tree', params_dec_tree),\
                    ('K Nearest Neighbors', params_knn),\
                    ('SVC (Linear)', params_lin_svc),\
                    ('SVC (RBF)', params_rbf_svc)]

    acc_lst_dct = dict()
    best_params_lst_dct = dict()
    for clf in dict(clf_lst).keys():
        acc_lst_dct[clf] = []
        best_params_lst_dct[clf] = []
    acc_lst_dct = collections.OrderedDict(sorted(acc_lst_dct.items()))
    best_params_lst_dct = collections.OrderedDict(sorted(best_params_lst_dct.items()))

    data_arr = np.array(data_lst)
    data_arr = data_arr.reshape(data_arr.shape[0],-1)
    label_arr = np.array(label_lst)
    k_range = np.linspace(data_arr.shape[1]/10,data_arr.shape[1],10).astype(np.int)
    for k_val in k_range:
        print(k_range)
    # trn_sizes = np.linspace(0.1,0.9,9)
    # for t_val in trn_sizes:
        # print trn_sizes
        # # https://scikit-learn.org/stable/modules/feature_selection.html
        # Feature selection
        # k_val = int(0.4*data_arr.shape[1])
        data_slctd = SelectKBest(f_classif,k=k_val).fit_transform(data_arr,label_arr)
        print(data_slctd.shape)

        trn_data, tst_data, trn_label, tst_label = train_test_split(data_slctd,label_arr,test_size=0.5,random_state=1,stratify=label_arr)
        # trn_data, tst_data, trn_label, tst_label = train_test_split(data_slctd,label_arr,train_size=t_val,random_state=1,stratify=label_arr)
        print(trn_data.shape)
        print(tst_data.shape)
        print(tst_label)
        trn_data = trn_data.reshape(trn_data.shape[0],-1)
        tst_data = tst_data.reshape(tst_data.shape[0],-1)

        # Determine the best parameters and set them
        clf_dct = collections.OrderedDict(sorted(dict(clf_lst).items()))
        tuned_clf_dct, acc_dct = search_params(clf_dct,dict(params_lst),trn_data,trn_label)

        for tuned_name, tuned_clf in tuned_clf_dct.items():
            print(tuned_name)
            print(tuned_clf.best_params_)
            print(acc_dct[tuned_name])
            # clf_dct[tuned_name].set_params(**tuned_clf.best_params_)
            acc_lst_dct[tuned_name].append(acc_dct[tuned_name])
            best_params_lst_dct[tuned_name].append(tuned_clf.best_params_)

    for name in best_params_lst_dct.keys():
        print(name)
        print(best_params_lst_dct[name])

    fig9 = plt.figure(9)
    ax9 = set_up_plt(fig9,36,20,'Fraction of total features','Highest accuracy score')
    # ax9 = set_up_2dplot(fig9,36,20,'Fraction of total dataset used for training','Highest accuracy score')
    plt_k_range = k_range/np.max(k_range).astype(np.float)
    barwidth = (plt_k_range[1]-plt_k_range[0])/10
    # barwidth = (trn_sizes[1]-trn_sizes[0])/10
    l = -1.5
    for clf in acc_lst_dct.keys():
        ax9.bar(plt_k_range+(l*barwidth),acc_lst_dct[clf],barwidth,label=str(clf))
        # ax9.bar(trn_sizes+(l*barwidth),acc_lst_dct[clf],barwidth,label=str(clf))
        l += 1
    ax9.set_xticks(plt_k_range)
    # ax9.set_xticks(trn_sizes)
    ax9.set_yticks(np.linspace(0,1,11))
    ax9.set_xlim([0,1.2])
    # ax9.set_xlim([0,1.0])
    ax9.set_ylim([0,1.3])
    ax9.legend(prop=dict(size=20))
    ax9.grid()
    fig9.set_tight_layout({'pad':1.0})
    fig9.set_size_inches(10,12)
    plt.show(block=True)

if __name__ == "__main__":
	main()

