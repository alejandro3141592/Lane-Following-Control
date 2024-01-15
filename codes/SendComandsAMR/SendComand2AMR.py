#!/usr/bin/env python

#Send commands to the Steering of the AMR via Serial, it receives the sterring angle, via a ROS topic

import rospy
from std_msgs.msg import Int8
import serial
import time



def steering_callback(data):
    #rospy.loginfo("Received data from ROS topic: %d", data.data)
    send_to_serial(83, 1, data.data)

def send_to_serial(id, len, data):
    global last_send_time
    try:
        
        string_to_send = f'{id} {len} {data}'

        if time.time() - last_send_time >= 1:
            print(string_to_send)
            serial_port.write(str.encode(string_to_send))
            last_send_time = time.time()
            rospy.loginfo("Sent data to serial port: %d", data)
        #time.sleep(1)

        
    except serial.SerialException as e:
        rospy.logerr("Serial port error: %s", str(e))

def listener():
    rospy.init_node('ros_to_serial', anonymous=True)
    rospy.Subscriber("AMR_Steering", Int8, steering_callback)  # Replace with your ROS topic name
    rospy.spin()

if __name__ == '__main__':
    last_send_time = time.time()
    while True:
        try:
            serial_port = serial.Serial('/dev/ttyUSB0', baudrate=115200, timeout=1)  # Replace with your serial port and baud rate
            print(f"Serial port /dev/ttyUSB1 opened successfully.")
            break
        except serial.SerialException as e:
            print(f"Error: {e}. Retrying in {3} seconds...")
            time.sleep(3)
                

    listener()
