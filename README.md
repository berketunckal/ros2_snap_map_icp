# ICP Localization Node

## Project Description

This project contains a ROS 2 node for determining a robot's position using the **ICP (Iterative Closest Point)** algorithm. The node estimates the robot's location by utilizing laser scans and a pre-existing map, providing continuous position updates.

### Features
- **ICP Algorithm**: Estimates the robot's position using laser scans and a known map.
- **TF Transformations**: Performs transformations between different coordinate frames using `tf2_ros`.
- **ROS Service**: Provides a `SetBool` ROS service to check if the initial position is set.
- **QoS Profiles**: Uses Quality of Service (QoS) profiles for laser scan and map messages.
- **ROS 2 Compatibility**: Designed to work seamlessly on ROS 2.

## Requirements

This project has the following software requirements:

- ROS 2 (Foxy, Galactic, or Humble distributions)
- PCL (Point Cloud Library)
- `laser_geometry` ROS package
- `tf2_ros` and `tf2_geometry_msgs` ROS packages

## Installation

1. **ROS 2 Installation**: Make sure ROS 2 is installed. You can follow the [ROS 2 installation guide](https://docs.ros.org/en/foxy/Installation.html).
   
2. **Install Required Dependencies**:
    ```bash
    sudo apt-get update
    sudo apt-get install ros-<ros2_distro>-pcl-conversions ros-<ros2_distro>-tf2-ros ros-<ros2_distro>-laser-geometry
    ```

3. **Clone the Project**:
    ```bash
    mkdir -p ~/ros2_ws/src
    cd ~/ros2_ws/src
    git clone https://github.com/berketunckal/ros2_snap_map_icp.git
    ```

4. **Build the Project**:
    ```bash
    cd ~/ros2_ws
    colcon build
    source install/setup.bash
    ```

## Usage
To check if the initial position is set, you can call the service with:

    ```
    ros2 service call /is_initial_pose_true std_srvs/srv/SetBool "{data: true}"
    ```

## Parameters and Constants

This node uses several parameters and constants:

### ICP Parameters
- `ICP_FITNESS_THRESHOLD`: Controls the fitness threshold for the ICP algorithm.
- `DIST_THRESHOLD`: Controls the distance threshold for the ICP algorithm.
- `ICP_NUM_ITER`: Controls the number of iterations for the ICP algorithm.

### QoS Profiles
- Laser scan messages: Uses a specific QoS profile.
- Map messages: Uses a different QoS profile.

### Various Constants
- Map frame: Specifies the frame name for the map.
- Odom frame: Specifies the frame name for odometry.
- Other constants: Includes additional constants for various frame names.

### Code Structure

The code structure of the `ICPLocalizationNode` class is as follows:

1. `ICPLocalizationNode` Class: The main class of the node that incorporates ROS 2 node functionalities, including message subscriptions, services, and timers.

The node also includes the following callback functions:

2. `mapCallback`: The callback function triggered when a map message is received.
3. `scanCallback`: The callback function triggered when a laser scan message is received.

Other important functions in the node include:

4. `processScan`: Processes the laser scan and runs the ICP algorithm.
5. `isInitialPoseTrue`: Service function that checks if the initial position is set.
6. `getTransform` and `waitForTransform`: Helper functions to perform TF transformations.


### Running the Node

You can run the node with the following command:

```
ros2 run snap_map_icp snap_map
```


## This is a fork of the discontinued code at https://github.com/code-iai/snap_map_icp .
