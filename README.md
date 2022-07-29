## Purpose of this repository

This repository holds the code used to generate the results presented in the paper
*R. Skoviera, J. K. Behrens and K. Stepanova, "SurfMan: Generating Smooth End-Effector Trajectories on 3D Object Surfaces for Human-Demonstrated Pattern Sequence," in IEEE Robotics and Automation Letters, vol. 7, no. 4, pp. 9183-9190, Oct. 2022, doi: [10.1109/LRA.2022.3189178](https://ieeexplore.ieee.org/document/9817640?source=authoralert).*
by Radoslav Skoviera, Jan Kristof Behrens, and Karla Stepanova. 
For more information, please consult our webpage [http://imitrob.ciirc.cvut.cz/surftask.html](http://imitrob.ciirc.cvut.cz/surftask.html).

### Disclaimer

The content of this repository is a prototypical proof of concept implementation.
We were using it on a real robot, but we don't take any responsibility for damages
incurred by using our code.

## Installation  

### Requirements  

This is a ROS package and as such, it requires ROS to be installed. It was developed under ROS Melodic:  
http://wiki.ros.org/melodic/Installation/Ubuntu  

The package is written for Python 2.7.  

Additionally, it requires these Python packages, besides _standard_, _ROS_, and _"scientific"_ (e.g. numpy, scipy, sklearn, pandas, matplotlib) Python packages:  
* open3d==0.12.0  
* cv2==4.2.0 (openCV for Python)  
* open3d
* toppra>=0.4.1 (time parameterization tool)
* numba
* fastdtw

Other ROS dependencies
- [Descartes motion planner](https://github.com/ros-industrial-consortium/descartes)
- [PyKDL](https://github.com/gt-ros-pkg/hrl-kdl)
- [RelaxedIK]() (optional)
- Capek testbed ROS packages (the robotic workplace used in the paper)

### Installation  

1) Clone the package from [https://github.com/imitrob/surftask](https://github.com/imitrob/surftask) into a ROS workspace ([ROS workspace](http://wiki.ros.org/catkin/Tutorials/create_a_workspace)).
2) Install missing dependencies using rosdep: `rosdep install --from-src src -y` TODO: check command and update package xml
3) Build using the `catkin build` command.  


## Input data  

You can use the system in two regimes:  
* on real data (online)  
* on pre-recorded data (offline)  

### Running on real data  

You will need:  
* a suitable RGBD camera connected to the computer  
* a ROS "driver" package for that camera that provides data on the necessary ROS topics (listed below)  

Expected topics:  
`/camera/color/image_raw [sensor_msgs/Image]`  
>  the RGB image from the camera  

`/camera/aligned_depth_to_color/image_raw [sensor_msgs/Image]`  
>  the depth image from the camera aligned to the color image  

`/camera/color/camera_info [sensor_msgs/CameraInfo]`  
`/camera/aligned_depth_to_color/camera_info [sensor_msgs/CameraInfo]`  
>  camera information (e.g. intrinsic matrix) for the color and depth cameras  

`/camera/depth/color/points [sensor_msgs/PointCloud2]`  
>  the pointcloud from the depth camera  

We recommend using Intel RealSense D435 or D455 cameras with the RealSense ROS package: https://github.com/IntelRealSense/realsense-ros  

### Running on pre-recorded data  

For convenience, we provide a ROS bagfile with all the necessary data - image data from the camera and contour detections.  

**Download the bagfile [here](https://drive.google.com/file/d/1p2BZwrEM5qTO04XNjmWiur6AsTcuHaYT/view?usp=sharing)**

To play the bagfile, start ROS master (`roscore`) and issue the following command in the terminal (assuming you are in a folder with the bagfile):  
`rosbag play -l 2020-10-27-19-30-25.bag`  
The `-l` flag ensures the bagfile will be "looping", otherwise the playback would stop at the end of the bagfile (12.7 seconds).  

## Usage  

The usage depends on the specific parts you want to run.

### Running in "demo" mode

This mode serves only to demonstrate the pattern application pipeline.

0) Start ROS master with `roscore`.  
1) Assuming that you are using pre-recorded data (see above), play the bagfile.  
2) Start "dummy" robot node (necessary when not using a real robot):  
  `rosrun traj_complete_ros dummy_control.py`
3) Start the GUI node:  
  `rosrun traj_complete_ros interface_node.py`
4) A GUI should show up. Controls are described below.

### Running in "full" mode

This mode assumes you have an RGBD camera and a robot to execute the trajectories.

0) Start ROS master with `roscore`.  
1) Launch the detection pipeline (change the launch file to launch the camera drivers, if necessary):  
  `roslaunch traj_complete_ros detection_pipe.launch`
2) Start the GUI node:  
  `rosrun traj_complete_ros interface_node.py`
3) A GUI should show up. Controls are described below.
#### Robot Control
4) Start your robot and its movegroup interface_node.
   `roslaunch capek_launch virtual_robot.launch`
5) Start the [Descartes planner server}(traj_complete_ros/cpp/descartes_planner_server.cpp)
  `rosrun traj_complete_ros dps`
6) Start the robot control node:
  `rosrun traj_complete_ros robot_control_node.py`
7) OPTIONAL: for advanced logging run
  `rosrun traj_complete_ros logger_node.py`



## GUI  

The interface node starts a GUI for the pattern application. Here is how to use it:  
* press __spacebar__ to pause the image update (otherwise the contour detections could be unstable)  
* use __shift__+__click__ to click inside a contour to select it  
* alternatively, press __s__ to cycle through contours present in the image  
* use __ctrl__+__click__ to draw a custom contour  
* press __c__ to delete the custom contour  
* press __a__ to apply pattern to the selected (or custom contour)  
* hit __enter__ to send the contour to the robot for execution  

With a contour selected (or custom contour drawn), use the sliders below the window to change the pattern or contour parameters.  
