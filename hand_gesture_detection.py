# TechVidvan hand Gesture Recognizer
# import necessary packages

# ros2 run ros_ign_bridge parameter_bridge /cmd_vel@geometry_msgs/msg/Twist@ignition.msgs.Twist
# THIS IS THE COMMAND TO ESTABLISH THE CONNECTION BETWEEN GAZEBO AND ROS SUCCESSFULLY (in main ros2_ws directory)!!!
# THE PROGRAM WORKS!!!
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
from multiprocessing import Process,Pipe
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist

class MinimalPublisher(Node):

    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.hand_gestures)
        self.i = 0

    def hand_gestures(self):
        # initialize mediapipe

# Mp.solution.hands module performs the hand recognition algorithm. So we create the object and store it in mpHands.
# Using mpHands.Hands method we configured the model. The first argument is max_num_hands, that means the maximum number of hand will be detected
#  by the model in a single frame. MediaPipe can detect multiple hands in a single frame, but we’ll detect only one hand at a time in this project.
# Mp.solutions.drawing_utils will draw the detected key points for us so that we don’t have to draw them manually.
        mpHands = mp.solutions.hands
        hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
        mpDraw = mp.solutions.drawing_utils

# Using the load_model function we load the TensorFlow pre-trained model.
# Gesture.names file contains the name of the gesture classes. So first we open the file using python’s inbuilt open function and then read the file.
# After that, we read the file using the read() function.

        # Load the gesture recognizer model
        model = load_model('/home/abhinav/ros2_ws/src/hand_gest/hand_gest/mp_hand_gesture')
        # Load class names
        f = open('/home/abhinav/ros2_ws/src/hand_gest/hand_gest/gesture.names', 'r')
        classNames = f.read().split('\n')
        f.close()
        print(classNames)
        command_dict = {'fist':'stop robot','thumbs up':'forward motion','thumbs down':'backward motion','rock':'acceleration','okay':'turn left',
                        'stop':'turn right','live long':'CW forward','call me':'ACW forward','peace':'CW backward','smile':'ACW backward'}
        print('Mapping for robot velocity commands is - ',command_dict)

# We create a VideoCapture object and pass an argument ‘0’. It is the camera ID of the system. In this case, we have 1 webcam connected with the system. If you have multiple webcams then change the argument according to your camera ID. Otherwise, leave it default.
# The cap.read() function reads each frame from the webcam.
# cv2.flip() function flips the frame.
# cv2.imshow() shows frame on a new openCV window.
# The cv2.waitKey() function keeps the window open until the key ‘q’ is pressed.
        # Initialize the webcam
        cap = cv2.VideoCapture(0)
        while True:
            # Read each frame from the webcam
            _, frame = cap.read()
            x, y, c = frame.shape
            # Flip the frame vertically
            frame = cv2.flip(frame, 1)
            framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Get hand landmark prediction
            result = hands.process(framergb)
            # print(result)
            className = ''
            # post process the result
            if result.multi_hand_landmarks:
                landmarks = []
                for handslms in result.multi_hand_landmarks:
                    for lm in handslms.landmark:
                        # print(id, lm)
                        lmx = int(lm.x * x)
                        lmy = int(lm.y * y)
                        landmarks.append([lmx, lmy])

                    # Drawing landmarks on frames
                    mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

# MediaPipe works with RGB images but OpenCV reads images in BGR format. So, using cv2.cvtCOLOR() function we convert the frame to RGB format.
# The process function takes an RGB frame and returns a result class.
# Then we check if any hand is detected or not, using result.multi_hand_landmarks method.
# After that, we loop through each detection and store the coordinate on a list called landmarks.
# Here image height (y) and image width(x) are multiplied with the result because the model returns a normalized result. 
# This means each value in the result is between 0 and 1.
# And finally using mpDraw.draw_landmarks() function we draw all the landmarks in the frame.

                    # Predict gesture
                    prediction = model.predict([landmarks])
                    # print(prediction)
                    classID = np.argmax(prediction)
                    className = classNames[classID]
                    # msg = String()
                    # msg.data = '%s' % className 

                    self.velo_msg = Twist()
                    # rate = rclpy.Rate(5)
                    # if else conditions for velocity commands -

                    if className == "fist": # stop robot
                        self.velo_msg.linear.x = 0.0
                        self.velo_msg.angular.z = 0.0
                    elif className == "thumbs up": # forward motion
                        self.velo_msg.linear.x = 1.0
                        self.velo_msg.angular.z = 0.0
                    elif className == "thumbs down": # backward motion
                        self.velo_msg.linear.x = -1.0
                        self.velo_msg.angular.z = 0.0
                    elif className == "rock": # acceleration
                        self.velo_msg.linear.x = 5.0
                        self.velo_msg.angular.z = 0.0
                    elif className == "okay": # turn left
                        self.velo_msg.linear.x = 0.0
                        self.velo_msg.angular.z = 0.5
                    elif className == "stop": # turn right
                        self.velo_msg.linear.x = 0.0
                        self.velo_msg.angular.z = -0.5
                    elif className == "live long": # clockwise motion forward
                        self.velo_msg.linear.x = 0.5
                        self.velo_msg.angular.z = -0.5
                    elif className == "call me": # anti-clockwise motion forward
                        self.velo_msg.linear.x = 0.5
                        self.velo_msg.angular.z = 0.5
                    elif className == "peace": # clockwise motion backward 
                        self.velo_msg.linear.x = -0.5
                        self.velo_msg.angular.z = -0.5
                    elif className == "smile": # anti-clockwise motion backward
                        self.velo_msg.linear.x = -0.5
                        self.velo_msg.angular.z = 0.5
                    else:
                        self.velo_msg.linear.x = 0.0
                        self.velo_msg.angular.z = 0.0
                    # here I want to get the prediction classname
                    self.publisher_.publish(self.velo_msg)
                    self.get_logger().info('Publishing: "%s"' % className)
                    self.i += 1
                    print(className)
                
            # show the prediction on the frame
            cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (0,0,255), 2, cv2.LINE_AA)
            # Show the final output
            cv2.imshow("Output", frame) 
            if cv2.waitKey(1) == ord('q'):
                break

# The model.predict() function takes a list of landmarks and returns an array contains 10 prediction classes for each landmark.
# The output looks like this-
# [[2.0691623e-18 1.9585415e-27 9.9990010e-01 9.7559416e-05
# 1.6617223e-06 1.0814080e-18 1.1070732e-27 4.4744065e-16 6.6466129e-07 4.9615162e-21]]
# Np.argmax() returns the index of the maximum value in the list.
# After getting the index we can simply take the class name from the classNames list.
# Then using the cv2.putText function we show the detected gesture into the frame.

        # release the webcam and destroy all active windows
        cap.release()
        cv2.destroyAllWindows()
    
def main(args=None):
    rclpy.init(args=args)
    minimal_publisher = MinimalPublisher()
    rclpy.spin(minimal_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

