#!/usr/bin/env python

#Receives an Image via a ROS topic, it applies a Lane Detection algorith to it, and returns the sterring value via other ROS topic

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from std_msgs.msg import Int8

Kp = 1 #constante proporcional
left_line = np.array([0, 0, 0, 0])
right_line = np.array([0, 0, 0, 0])

last_lines = None
last_circle_coords = [None, None]

def resize(image):
    new_width = 600
    new_height = 520
    new_image = cv2.resize(image, (new_width, new_height))
    return new_image

def grey(image):
    blue_channel, green_channel, red_channel = cv2.split(image)
    imageG = red_channel
    imageG.dtype = np.uint8
    inverse_image = cv2.bitwise_not(imageG)
    ret, threshG = cv2.threshold(inverse_image, 170, 255, cv2.THRESH_TRUNC)
    ret2, threshBlack = cv2.threshold(threshG, 150, 255, cv2.THRESH_TOZERO)
    return imageG

def gauss(image):
    return cv2.GaussianBlur(image, (5, 5), 0)

def binarizar_imagen(img, umbral=225):
    _, binarizada = cv2.threshold(img, umbral, 255, cv2.THRESH_BINARY)
    return binarizada

def adelgazar_lineas_verticales(img_binaria):
    img_binaria = 0 + img_binaria
    kernel = np.ones((1, 3), np.uint8)
    img_adelgazada = cv2.erode(img_binaria, kernel, iterations=3)
    return img_adelgazada

def canny(image, low_threshold, high_threshold):
    edges = cv2.Canny(image, low_threshold, high_threshold, apertureSize=7)
    return edges

def region(image):
    height, width = image.shape
    triangle = np.array([
        [(0, 520), (0, 245), (600, 245), (600, 520)]
    ], np.int32)

    mask = np.zeros_like(image)
    mask = cv2.fillPoly(mask, [triangle], 150)
    masked = cv2.bitwise_and(image, mask)
    return masked

def encontrar_lineas_verticales(img):
    lines = cv2.HoughLinesP(img, 1, np.pi/180, threshold=50, minLineLength=0.7, maxLineGap=250)
    vertical_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            vertical_lines.append(line)
            # if abs(x2 - x1) > 0:
            #     slope = (y2 - y1) / (x2 - x1)
            #     if abs(slope) > 0.3:
            #         vertical_lines.append(line)
    return vertical_lines

def proyectar_lineas(img, lines, num_lines=2):
    lines_image = np.zeros_like(img)
    img_copy = img.copy()
    means = []
    if lines is not None:
        lines = sorted(lines, key=lambda x: np.arctan2(x[0, 3] - x[0, 1], x[0, 2] - x[0, 0]))

        right_line = None
        left_line = None

        for line in lines:
            #print(line)
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / (x2 - x1)
            
            if abs(slope) > 0:
                if slope > 0 and right_line is None:
                    right_line = line[0]
                elif slope < 0 and left_line is None:
                    left_line = line[0]

        if right_line is not None:
            x1, y1, x2, y2 = right_line
            slope = (y2 - y1) / (x2 - x1)
            cv2.line(img_copy, (x1, y1), (x2, y2), (255, 0, 0), 5)
            print("right line  ", end="")
            print(right_line, end="")
            print("slope ", end="")
            print(slope)
            means.append((x1 + x2)//2)


        if left_line is not None:
            x1, y1, x2, y2 = left_line
            slope = (y2 - y1) / (x2 - x1)
            cv2.line(img_copy, (x1, y1), (x2, y2), (255, 0, 0), 5)
            print("left line  ", end="")
            print(left_line, end="")
            print("slope ", end="")
            print(slope)
            means.append((x1 + x2)//2)
        print("")


        cv2.imshow("lines", lines_image)
        selected_lines = [left_line, right_line]


        
        x_avg = int(np.mean(means))

        # Obtener la altura para proyectar el círculo
        height, _ = img.shape[:2]
        y_position = int(height * (1 - 1/4))

        # Actualizar la información del último frame exitoso
        last_lines = selected_lines
        last_circle_coords = (x_avg, y_position)

        # Dibujar un círculo púrpura en la copia de la imagen original
        
        cv2.circle(img_copy, last_circle_coords, 10, (255, 0, 255), -1)
    
    return img_copy, last_circle_coords[0]


def proyectar_circulo_y_lineas(img, lines):
    global last_lines, last_circle_coords

    if lines:
        lines = sorted(lines, key=lambda x: np.arctan2(x[0, 3] - x[0, 1], x[0, 2] - x[0, 0]))

        right_line = None
        left_line = None

        for line in lines:
            #print(line)
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / (x2 - x1)
            
            if abs(slope) > 0:
                if slope > 0 and right_line is None:
                    right_line = line[0]
                elif slope < 0 and left_line is None:
                    left_line = line[0]

            if right_line is not None:
                x1, y1, x2, y2 = right_line
                slope = (y2 - y1) / (x2 - x1)

                cv2.line(right_line, (x1, y1), (x2, y2), (255, 0, 0), 5)
                print("right line  ", end="")
                print(right_line, end="")
                print("slope ", end="")
                print(slope)


            if left_line is not None:
                x1, y1, x2, y2 = left_line
                slope = (y2 - y1) / (x2 - x1)
                cv2.line(left_line, (x1, y1), (x2, y2), (255, 0, 0), 5)
                print("left line  ", end="")
                print(left_line, end="")
                print("slope ", end="")
                print(slope)
        
        selected_lines = [left_line, right_line]
        x_avg = int(np.mean([(line[0] + line[0]) // 2 for line in selected_lines]))

        # Obtener la altura para proyectar el círculo
        height, _ = img.shape[:2]
        y_position = int(height * (1 - 1/4))

        # Actualizar la información del último frame exitoso
        last_lines = selected_lines
        last_circle_coords = (x_avg, y_position)

        # Dibujar un círculo púrpura en la copia de la imagen original
        img_copy = img.copy()
        cv2.circle(img_copy, last_circle_coords, 10, (255, 0, 255), -1)

        # Dibujar las líneas azules en la copia de la imagen original
        for line in last_lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img_copy, (x1, y1), (x2, y2), (255, 0, 0), 5)


    # Si no hay suficientes líneas en este frame, usar la información del último frame exitoso
    if last_lines is not None and last_circle_coords is not None:
        img_copy = img.copy()
        cv2.circle(img_copy, last_circle_coords, 10, (255, 0, 255), -1)

        for line in last_lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img_copy, (x1, y1), (x2, y2), (255, 0, 0), 5)

        print("Usando información del último frame exitoso.")
        return img_copy, last_circle_coords[0]

    print("No hay suficientes líneas para proyectar el círculo y las líneas azules.")
    return img, last_circle_coords[0]

max_window_width = 600  # Adjust this to your screen width
max_window_height = 520  # Adjust this to your screen height

class ZED2ImageSubscriber:
    def __init__(self):
        rospy.init_node('zed2_image_subscriber', anonymous=True)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('/Pix2Pix/Image', Image, self.image_callback)
        self.steering_publisher = rospy.Publisher('AMR_Steering', Int8, queue_size=10)
        self.frame_width = 640
        self.frame_height =  360



    def proccess_Image(self, or_frame):
        copy = np.copy(or_frame)
        New_copy = resize(copy)
        grey_img = grey(New_copy)
        gaussian = gauss(grey_img)
        # Binarize the resulting image from the dynamic Otsu filter
        img_binarizada = binarizar_imagen(gaussian)
        # Thin the vertical lines in the final image
        img_adelgazada = adelgazar_lineas_verticales(img_binarizada)
        edges = canny(img_adelgazada, 20, 100)
        isolated_region = region(edges)
        cv2.imshow('Lane isolated_region',isolated_region)
        # Find vertical lines in the thinned image
        lines_verticales = encontrar_lineas_verticales(isolated_region)

        # Project the four closest lines onto the original image
        #img_con_lineas = proyectar_lineas(New_copy, lines_verticales, num_lines=0.5)

        # Extend the lines and mark the intersection points
        img_with_intersection = New_copy

        # Project a purple circle and blue lines at the calculated coordinates
        Result, center_x = proyectar_lineas(New_copy, lines_verticales)
        try:
            error = center_x - (640/2)
            print("center_x")
            print(center_x)

            print("window_width")
            print(640/2)

            # Aplicar el control proporcional
            control_signal = Kp * error
            print("error: ", error)
            print("control signal: ", control_signal)
        except:
            control_signal = 0

        # Display the processed frame
        cv2.namedWindow('Lane Detection', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Lane Detection',max_window_width,max_window_height)
        cv2.imshow('Lane Detection',Result)  # Display the edges here
        
        scaled_control = max(0, min(100, 50 + control_signal))

    
        return scaled_control




    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            print(e)
            return

        # Display the image
        cv2.imshow("image_received", cv_image)
        key = cv2.waitKey(1)
        if key == 27:  # 27 corresponds to the ASCII code for the ESC key
            rospy.signal_shutdown("ESC key pressed")


        scaled_control = self.proccess_Image(cv_image)

        


        print(scaled_control)

        steering_msg = Int8()
        steering_msg.data = int(scaled_control)
        
        self.steering_publisher.publish(steering_msg)


def main():

    try:
        zed2_subscriber = ZED2ImageSubscriber()
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()