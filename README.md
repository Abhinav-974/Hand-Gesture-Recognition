# Hand_Gesture_Recognition
Using Hand Gesture Predictions to simulate robot motion in Gazebo

Prerequisites - This project uses Ubuntu 22.04 OS, ROS 2 Humble, Gazebo Fortress (Ignition Gazebo) as the main softwares.Note that these 3 are the recommended combination of softwares for your system here - https://gazebosim.org/docs/fortress/ros_installation

Instructions to install ROS 2 - https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debians.html
Instructions to install Ignition Gazebo - https://gazebosim.org/docs/fortress/install_ubuntu

Directions to implement this project (project setup) - 
Step 1. Create a ROS2 workspace. I'm calling it 'ros2_ws'. Once you build the workspace, and run 'ls' in the terminal of root directory (~/ros2_ws), you'll see 4  folders) as - build  install  log  src
(Tutorial - https://docs.ros.org/en/humble/Tutorials/Beginner-Client-Libraries/Creating-A-Workspace/Creating-A-Workspace.html)

Step 2. Create a package. I'm calling it 'hand_gest'. Command - ros2 pkg create --build-type ament_python hand_gest. Then, build the package by running - colcon build --packages-select hand_gest. The package contents will look like this - hand_gest  package.xml  resource  setup.cfg  setup.py  test

Step 3. Inside the hand_gest folder inside the hand_gest package, clone the github repo. You will need to edit the generated setup.py and package.xml files as shown - 
setup.py file
```
from setuptools import setup
import os
from glob import glob

package_name = 'hand_gest'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    # package_dir = {'':'src'},
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # need to put in the launch files and robot sdf files paths
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='(your name)',
    maintainer_email='(your email)',
    description='Hand Gesture Recognition',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        # name of the main python file and what it does to include
        'hand_gesture_detection = hand_gest.hand_gesture_detection:main',
        ],
    },
)
```
package.xml file
```
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>hand_gest</name>
  <version>0.0.0</version>
  <description>Hand Gesture Recognition</description>
  <maintainer email="(your email)">(your name)</maintainer>
  <license>Apache License 2.0</license>

  <test_depend>ament_copyright</test_depend>
  <test_depend>ament_flake8</test_depend>
  <test_depend>ament_pep257</test_depend>
  <test_depend>python3-pytest</test_depend>
<exec_depend>rclpy</exec_depend>
<exec_depend>std_msgs</exec_depend>
  <export>
    <build_type>ament_python</build_type>
  </export>
</package>
```
Step 4. In the main hand gesture code (hand_gesture_detection.py), you will need to edit the file paths in lines 42-45 as per your choice, as shown - 
```
# Load the gesture recognizer model
model = load_model('/home/abhinav/ros2_ws/src/hand_gest/hand_gest/mp_hand_gesture')
# Load class names
f = open('/home/abhinav/ros2_ws/src/hand_gest/hand_gest/gesture.names', 'r')
```
Directions to implement this project (project execution) - 

Step 5. Open a terminal inside the hand_gest folder inside the hand_gest package (~/ros2_ws/src/hand_gest/hand_gest). Run the robot.sdf file as - 
```ign gazebo robot.sdf```

Step 6. Open two more terminals, in the root workspace directory (~/ros2_ws). In the first one, first source the setup file as - ```source install/setup.bash```. Then run the main code as - ```ros2 run hand_gest hand_gesture_detection```.

Step 7. In the third terminal, establish a bridge to allow communication betwen ROS and Gazebo as - ```ros2 run ros_ign_bridge parameter_bridge /cmd_vel@geometry_msgs/msg/Twist@ignition.msgs.Twist```. More information on - https://gazebosim.org/docs/fortress/ros2_integration.

Start the gazebo simulation by pressing the play button. You will see the robot moving according to your hand gesture predictions! Enjoy!

We have provided the given list of gestures - ['okay', 'peace', 'thumbs up', 'thumbs down', 'call me', 'stop', 'rock', 'live long', 'fist', 'smile']. and a mapping between these gestures and the robot velocity commands as - 
{'fist':'stop robot','thumbs up':'forward motion','thumbs down':'backward motion','rock':'acceleration','okay':'turn left', 'stop':'turn right','live long':'CW forward','call me':'ACW forward','peace':'CW backward','smile':'ACW backward'}
