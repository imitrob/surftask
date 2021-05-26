
## Start simulated robot and real camera for testing

run simulated robots

```roslaunch capek_launch virtual_robot.launch```

apply camera calibration and start realsense

```roslaunch traj_complete_ros vision.launch```

You should see two robots, pancakes, and some blue frosting by running the following code. This node is meant to become 
the vison node and will later detect the objects and publish their poses and contours.

```rosrun traj_complete_ros object_tracker_node.py```


## Relaxed IK notes

start rust relaxedIK node:
```shell script
roslaunch relaxed_ik load_info_file.launch
```
```shell script
roslaunch relaxed_ik relaxed_ik_rust.launch
```


## Work with panda

```shell script
roslaunch panda_gazebo panda_world.launch start_moveit:=false 
```

This will start gazebo with the panda robot. MoveIt is not enabled. Panda_robot interface is usable.

```shell script
rosrun traj_complete_ros panda_controller.py
```

will launch a task space controller, which listens to a PointVel.msg 
```yaml
std_msgs/Header hearder
geometry_msgs/Pose pose
float64[] lin_vel
float64[] ang_vel_xyz
```
on the topic `/motion_goal`.

