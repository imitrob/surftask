## Installation  

### Requirements  

This is a ROS package and as such, it requires ROS to be installed. It was developed under ROS Melodic:  
http://wiki.ros.org/melodic/Installation/Ubuntu  

The package is written for Python 2.7.  

Additionally, it requires these Python packages, besides _standard_, _ROS_, and _"scientific"_ (e.g. numpy, scipy) Python packages:  
* open3d==0.12.0  
* cv2==4.2.0 (openCV for Python)  
# TODO: add Python packages from the robotic side  

### Installation  

1) Download the package from [https://github.com/imitrob/surftask](https://github.com/imitrob/surftask).
2) Create [ROS workspace](http://wiki.ros.org/catkin/Tutorials/create_a_workspace) and copy the source codes to the _src_ folder of the workspace.  
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
# TODO add robotic parts



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