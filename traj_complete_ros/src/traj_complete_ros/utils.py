import numpy as np
#from traject_msgs.msg import ContourArrayStamped, Contour, Point2D
from sensor_msgs.msg import PointCloud2, PointField
# import tf
from tf import transformations as tr
from scipy import signal, interpolate
import cv2
import matplotlib.pyplot as plt
from uuid import uuid4
import yaml
import copy

from traject_msgs.msg import ContourPrimitive
import tf.transformations as tftrans


def contours2msg(contours, stamp, frame):
    msg = ContourArrayStamped()
    msg.contours = [Contour(points=[Point2D(**{k: v for k, v in zip(["x", "y"], p.ravel())}) for p in c]) for c in contours]
    msg.header.stamp, msg.header.frame_id = stamp, frame
    return msg


def contour_msg2list(contour_msg):
    return [np.array([np.array([p.x, p.y], dtype=np.int) for p in c.points], dtype=np.int) for c in contour_msg.contours]


def ftl_pcl2numpy(pcl_msg, uses_stupid_bgr_format=True):
    # warning: this assumes that the machine and PCL have the same endianness!
    step = int(pcl_msg.point_step / 4)  # point offset in dwords
    rgb_offset = int(next((f.offset for f in pcl_msg.fields if f.name == "rgb")) / 4)  # rgb offset in dwords
    dwordData = np.frombuffer(pcl_msg.data, dtype=np.byte).view(np.float32).reshape(-1, step)  # organize the bytes into double words (4 bytes)
    xyz = dwordData[:, :3]  # assuming x, y, z are the first three dwords
    rgb = dwordData[:, rgb_offset].view(np.uint32)  # extract rgb data
    rgb3d = rgb.copy().view(np.uint8).reshape(-1, 4)[:, slice(1, None) if pcl_msg.is_bigendian else slice(None, 3)]
    if uses_stupid_bgr_format:
        rgb3d = np.fliplr(rgb3d)
    return xyz, rgb3d


def ftl_numpy2pcl(xyz, orig_header, rgb=None, rgb3d=None, uses_stupid_bgr_format=True):
    """
    inverse method to ftl_pcl2numpy.
    Converts np array to ROS2 PointCloud2.

    @arg rgb: additional np array with color. Then use xyz+rgb pointcloud. Otherwise only xyz is used (default).
    """
    itemsize = np.dtype(np.float32).itemsize
    assert xyz.shape[0] == 3, "xyz must be only 'xyz' data in shape (3,N)"
    num_points = xyz.shape[1]

    if rgb3d is not None:
        if uses_stupid_bgr_format:
            rgb3d = np.fliplr(rgb3d)
        rgb = np.pad(rgb3d, ((0, 0), (0, 1)), mode="constant", constant_values=255).view(np.float32).ravel()
    if rgb is not None:
        assert num_points == len(rgb), "color must have same number of points"
        fields = [PointField(name=n, offset=i*itemsize, datatype=PointField.FLOAT32, count=1) for i, n in enumerate(list('xyz') + ['rgb'])]
        fields[-1].offset = 16
        k = 5
        dataa = np.concatenate((xyz.T, np.zeros((num_points, 1), dtype=np.float32), rgb[:, np.newaxis]), axis=1).astype(np.float32)
    else:
        fields = [PointField(name=n, offset=i*itemsize, datatype=PointField.FLOAT32, count=1) for i, n in enumerate(list('xyz'))]
        # fields[-1].offset = 12  # FIXME probably not 16 is in xyz+rgb version
        k = 3
        dataa = xyz.astype(np.float32).T

    pcl = PointCloud2(
        header=orig_header,
        height=1,
        width=num_points,
        fields=fields,
        point_step=(itemsize * k),  # =xyz + padding + rgb
        row_step=(itemsize * k * num_points),
        data=dataa.tobytes()
        )
    return pcl


def getTransformFromTF(tf_msg):
    trans = np.r_["0,2,0", [getattr(tf_msg.transform.translation, a) for a in "xyz"]]
    rot = tr.quaternion_matrix(np.r_[[getattr(tf_msg.transform.rotation, a) for a in "xyzw"]])[:3, :3]
    return trans, rot


class BsplineGen():

    @classmethod
    def fromLibraryData(cls, params, xout, nmbPts, **kwargs):
        """Generates BsplineGen object from a dictionary. Recommended use is:
        bsp_gen = BsplineGen.fromLibraryData(**data)
        assuming "data" contains the BsplineGen parameters.

        Args:
            params (list of lists): B-spline parameters
            xout (list): B-spline points?
            nmbPts (int): Number of points

        Returns:
            BsplineGen: BsplineGen object initialized with the data from the library dictionary.
        """
        bsplineGen = cls()
        bsplineGen.pars = params
        bsplineGen.pars[0] = np.array(bsplineGen.pars[0])
        bsplineGen.pars[1][0] = np.array(bsplineGen.pars[1][0])
        bsplineGen.pars[1][1] = np.array(bsplineGen.pars[1][1])
        bsplineGen.xout = np.array(xout)
        bsplineGen.nmbPts = nmbPts
        return bsplineGen

    @classmethod
    def fromWaypoints(cls, points, per_flag=False):
        bsplineGen = cls()
        bsplineGen.generate_bspline_pars(points, per_flag)
        bsplineGen.nmbPts = points.shape[0]
        return bsplineGen

    @classmethod
    def generateLibraryData(cls, points, per_flag=False, name=""):
        """Generates a dictionary object with the BsplineGen parameters
        from waypoints (e.g., a demonstrated contour)

        Args:
            points (Nx2 ndarray): The contour from which the pattern should be created.
            per_flag (bool, optional): A parameter for the "genereate_bspline_pars" function. Defaults to False.
            name (str, optional): Name of the pattern. If empty, uuid4 is generated as the name. Defaults to "".

        Returns:
            dict: The dictionary element to be stored in a pattern library.
        """
        bsplineGen = cls()
        params, xout = bsplineGen.generate_bspline_pars(points, per_flag)
        nmbPts = points.shape[0]
        return cls.splineToDict(params, xout, nmbPts, name)

    @staticmethod
    def splineToDict(params, xout, nmbPts, name=""):
        """Converts BsplineGen data into dictionary.
        """
        params = copy.deepcopy(params)
        params[0] = params[0].tolist()
        params[1][0] = params[1][0].tolist()
        params[1][1] = params[1][1].tolist()

        name = name or str(uuid4())

        return {
            name: {
                "name": name,
                "xout": xout.tolist(),
                "params": params,
                "nmbPts": nmbPts
            }
        }

    @staticmethod
    def appendDataToLibrary(data, libraryPath):
        """Appends a pattern data to a library

        Args:
            data (dict): A dictionary with BsplineGen data generated by "splineToDict", "generateLibraryData", or "toDict" functions.
            libraryPath (str): Path to the pattern library.
        """
        with open(libraryPath, "r") as file:
            library = yaml.safe_load(file)

        library["curve_library"].update(data)

        with open(libraryPath, "w") as file:
            yaml.dump(library, file)

    @classmethod
    def storePatternToLibrary(cls, libraryPath, points, per_flag=False, name=""):
        """Generates BsplineGen params from a demonstrated pattern (points) and stores them to a library.
        """
        data = cls.generateLibraryData(points, per_flag, name)
        cls.appendDataToLibrary(data, libraryPath)

    @classmethod
    def loadPatternFromLibrary(cls, libraryPath, name):
        with open(libraryPath, "r") as file:
            library = yaml.safe_load(file)["curve_library"]

        if name in library:
            return cls.fromLibraryData(**library[name])
        else:
            return None  # TODO: raise not found error?

    def __init__(self):
        self.pars = []
        self.xout = []
        self.nmbPts = []

    def toDict(self, name=""):
        if not (np.any(self.xout) and self.nmbPts):  # cannot convert "empty" spline to dict
            return None  # TODO: raise error?
        return self.splineToDict(self.pars, self.xout, self.nmbPts, name)

    def appendToLibrary(self, libraryPath, name=""):
        self.appendDataToLibrary(self.toDict(name), libraryPath)

    def generate_bspline_pars(self, points, per_flag=False):
        '''generates bspline parameters from points'''
        x = points.reshape(-1, 2)[:, 0]
        y = points.reshape(-1, 2)[:, 1]
        try:
            if per_flag is True:
                self.pars, self.xout = interpolate.splprep([x, y], s=0, per=1)
            elif per_flag is False:
                self.pars, self.xout = interpolate.splprep([x, y], s=0)
        except Exception as e:
            print(e)
            raise(ValueError('Pattern and contour dimensions probably mismatch.'))

        return self.pars, self.xout

    def generate_bspline_sample(self, nmbPts=None):
        '''returns sampled bspline by nmbPts'''
        nmbPts = nmbPts or self.nmbPts
        u_interp = np.linspace(self.xout.min(), self.xout.max(), nmbPts)

        return np.transpose(np.array(interpolate.splev(u_interp, self.pars)))


def cont_pt_by_pt(cont_coord, img_name, img, color_arr, pause_time, debug_mode=False):
    '''test'''
    for coord in cont_coord.reshape(-1, 2):
        if debug_mode is True:
            print(coord)
        img[coord[1].astype(np.int), coord[0].astype(np.int)] = color_arr
        cv2.imshow(img_name, img)
        cv2.waitKey(pause_time)

    if debug_mode is True:
        print('-----')
        print(cont_coord.reshape(-1, 2)[0])
        print(cont_coord.reshape(-1, 2)[-1])
        print(cont_coord.shape)
        print('-----')
        print('waitKey(0)')
        cv2.imshow(img_name, img)
        cv2.waitKey(0)


def set_up_2dplot(fig, x_label, y_label, label_font_size, tick_font_size):
    '''test'''
    ax = fig.add_subplot(111)
    ax.set_xlabel(x_label, fontsize=label_font_size)
    ax.set_ylabel(y_label, fontsize=label_font_size)
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(tick_font_size)

    return ax


def smooth_contours_w_butterworth(contour, fltr_ordr, norm_cutoff_freq, fltr_type_str):
    '''test'''
    # # https://stackoverflow.com/questions/51259361/smooth-a-bumpy-circle/51267877
    # filtering contour with butterworth filter so its not so jagged
    cont_coord = np.array(contour).reshape((-1, 2))
    x_coord = np.array([coord[0] for coord in cont_coord])
    y_coord = np.array([coord[1] for coord in cont_coord])
    cmplx_sgnl = x_coord + 1.0j*y_coord

    # # https://dsp.stackexchange.com/questions/49460/apply-low-pass-butterworth-filter-in-python
    b, a = signal.butter(fltr_ordr, norm_cutoff_freq, fltr_type_str)
    return signal.filtfilt(b, a, cmplx_sgnl)


def plot_displacement_scatter(data, title):
    # plot displacement as scatter to make it easy to determine if enough trajectory points visually
    dt = 0.1
    # dx = traj_bspline.reshape(-1,2)[:,0]
    # dy = traj_bspline.reshape(-1,2)[:,1]
    dx = data.reshape(-1, 2)[:, 0]
    dy = data.reshape(-1, 2)[:, 1]
    print(np.max(np.diff(dx)))
    print(np.max(np.diff(dy)))
    fig1 = plt.figure()
    ax1 = set_up_2dplot(fig1, 'x','' , 20, 12)
    # ax1.scatter(np.arange(0,dt*len(dx),dt),dx)
    # ax1.scatter(np.arange(0,dt*len(dy),dt),dy)
    ax1.scatter(dx, dy, linewidths=1, s=20)
    #ax1.grid()
    plt.show()


def get_contour_primitive(curve_primitive):
    nr_points = 250
    assert isinstance(curve_primitive, ContourPrimitive)
    if curve_primitive.type == ContourPrimitive.SPIRAL:
        if curve_primitive.ccw:
            counters = np.linspace(0, curve_primitive.rounds * 2 * np.pi, nr_points)
        else:
            counters = np.linspace(0, -curve_primitive.rounds * 2 * np.pi, nr_points)
        factors = np.linspace(curve_primitive.r_start, curve_primitive.r_end, nr_points)
        heights = np.linspace(curve_primitive.h_start, curve_primitive.h_end, nr_points)

        curve_points = np.array(
            [[R * np.sin(counter), R * np.cos(counter), h] for counter, R, h in zip(counters, factors, heights)])
        curve_normals = np.array([curve_primitive.normal for _ in range(nr_points)])
    if curve_primitive.type == ContourPrimitive.RECTANGLE:
        nr_points_a = int(curve_primitive.a * curve_primitive.res)
        nr_points_b = int(curve_primitive.b * curve_primitive.res)
        a = np.linspace(-0.5 * curve_primitive.a, 0.5 * curve_primitive.a, nr_points_a)
        b = np.linspace(-0.5 * curve_primitive.b, 0.5 * curve_primitive.b, nr_points_b)
        h = np.linspace(curve_primitive.h_start, curve_primitive.h_end, 2 * (nr_points_a + nr_points_b))

        # rectangle starts in top left corner, center is in the middle
        curve_points = np.zeros(shape=(2 * (nr_points_a + nr_points_b), 3))
        curve_points[0:nr_points_a] = np.array(
            [a, nr_points_a * [curve_primitive.b * 0.5], h[0:nr_points_a]]).transpose()
        curve_points[nr_points_a:nr_points_a + nr_points_b] = np.array(
            [nr_points_b * [curve_primitive.a * 0.5], -b, h[nr_points_a:nr_points_a + nr_points_b]]).transpose()
        curve_points[nr_points_a + nr_points_b:2 * nr_points_a + nr_points_b] = np.array(
            [-1.0 * a, nr_points_a * [curve_primitive.b * (-0.5)],
             h[nr_points_a + nr_points_b:2 * nr_points_a + nr_points_b]]).transpose()
        curve_points[2 * nr_points_a + nr_points_b:] = np.array(
            [[-0.5 * curve_primitive.a] * nr_points_b, b, h[2 * nr_points_a + nr_points_b:]]).transpose()

        if curve_primitive.ccw:
            curve_points = np.flip(curve_points, axis=0)

        curve_normals = np.array([curve_primitive.normal for _ in range(2 * (nr_points_a + nr_points_b))])

    # transform points and normals
    rot = tftrans.quaternion_matrix([curve_primitive.pose.orientation.x,
                                     curve_primitive.pose.orientation.y,
                                     curve_primitive.pose.orientation.z,
                                     curve_primitive.pose.orientation.w])
    trans = tftrans.translation_matrix([curve_primitive.pose.position.x,
                                        curve_primitive.pose.position.y,
                                        curve_primitive.pose.position.z])

    curve_points = np.dot(np.hstack([curve_points, np.ones((curve_points.shape[0], 1))]), rot)
    curve_points = np.dot(curve_points, trans)
    curve_normals = np.dot(curve_normals, rot[:3, :3])

    # make sure, the velocity is set to some reasonable range
    # exec_options.cart_vel_limit = max(min(exec_options.cart_vel_limit, 0.2), 0.005)

    local_trans = tftrans.translation_matrix([curve_primitive.pose.position.x,
                                              curve_primitive.pose.position.y,
                                              curve_primitive.pose.position.z])
    local_rot = tftrans.quaternion_matrix([curve_primitive.pose.orientation.x,
                                           curve_primitive.pose.orientation.y,
                                           curve_primitive.pose.orientation.z,
                                           curve_primitive.pose.orientation.w])
    curve_shifted = np.dot(local_rot, curve_points.transpose()).transpose()
    curve_shifted = np.dot(local_trans, curve_shifted.transpose()).transpose()
    curve_points = curve_shifted

    # remove duplicate points from data
    to_delete = np.where(np.linalg.norm(curve_points[1:] - curve_points[:-1], axis=1) <= 0.0001)
    curve_points = np.delete(curve_points, to_delete, axis=0)
    curve_normals = np.delete(curve_normals, to_delete, axis=0)

    return curve_points[:, :2]