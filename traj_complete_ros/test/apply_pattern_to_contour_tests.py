import os

import yaml
from rospkg import rospack
from rospkg import RosPack
from traject_msgs.msg import ContourPrimitive
import numpy as np

from traj_complete_ros.utils import BsplineGen
from traj_complete_ros.utils import get_contour_primitive
from traj_complete_ros.apply_pattern_to_contour import apply_pattern_to_contour
from traj_complete_ros.utils import plot_displacement_scatter

def get_cfg():
    default_config = "config/default_configuration.yaml"
    rospack = RosPack()
    with open(os.path.join(rospack.get_path('traj_complete_ros'), default_config), "r") as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
        print('config: {}'.format(cfg))
    return cfg


def make_contour_primitive():
    cp = ContourPrimitive()
    cp.a = 0.2
    cp.b = 0.1
    cp.ccw = False
    cp.pose.position.x = 0.2
    cp.pose.orientation.w = 1
    cp.type = ContourPrimitive.SPIRAL
    cp.r_start = 0.05
    cp.r_end = 0.05
    cp.rounds = 1.0
    cp.normal = [0, 0, 1]
    cp.h_start = 0.02
    cp.h_end = 0.02
    cp.res = 100
    contour = get_contour_primitive(curve_primitive=cp)
    return contour

def test_apply_pattern_to_contour():
    cfg = get_cfg()
    bsp = BsplineGen.fromLibraryData(**cfg["curve_library"]['knot'])
    contour = make_contour_primitive()
    num_pattern_reps = 10
    n_points = max([bsp.nmbPts * num_pattern_reps, 2 * contour.shape[0], 200])
    n_points = 401

    approx_method = 2
    pattern_rotation = 20
    pattern_trim_start = 0
    pattern_trim_trail = 0
    pattern_gap = 4
    contour_splined = np.array(
        list(reversed(BsplineGen.fromWaypoints(contour).generate_bspline_sample(n_points))))

    applied_bsp = apply_pattern_to_contour(bsp, contour_splined, num_pattern_reps,
                                           approx_method=approx_method, pattern_rotation=pattern_rotation,
                                           pattern_trim_start=pattern_trim_start,
                                           pattern_trim_trail=pattern_trim_trail,
                                           use_gap=pattern_gap)

    plot_displacement_scatter(applied_bsp.generate_bspline_sample(n_points), 'pttrn_bspline_s')

    assert True
    print('happy we are')

    # n_points = max([bsp.nmbPts * self.num_pattern_reps, 2*current_contour.shape[0], 200])
    # contour_splined = np.array(list(reversed(BsplineGen.fromWaypoints(current_contour).generate_bspline_sample(n_points))))
    #
    # assert False
