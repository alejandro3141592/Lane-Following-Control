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



def resize(image):
# Cambiar la resolución a 640x480
    new_width = 600
    new_height = 520
    new_image = cv2.resize(image, (new_width, new_height))
    return new_image

def grey(image):
    blue_channel, green_channel, red_channel = cv2.split(image)
    imageG = red_channel
    imageG.dtype = np.uint8
    inverse_image = cv2.bitwise_not(imageG)
    ret, threshG = cv2.threshold(inverse_image, 170, 255, cv2.THRESH_TRUNC)# prueba 11 am 150 
    ret2, threshBlack = cv2.threshold(threshG, 150, 255, cv2.THRESH_TOZERO)# prueba 11 am 140
    return imageG#threshBlack

def binarizar_imagen(img, umbral=145):
    _, binarizada = cv2.threshold(img, umbral, 255, cv2.THRESH_BINARY)
    return binarizada

def adelgazar_lineas_verticales(img_binaria):
    img_binaria = 0 + img_binaria
    kernel = np.ones((1, 3), np.uint8)
    img_adelgazada = cv2.erode(img_binaria, kernel, iterations=2)
    return img_adelgazada



def binaryOtsu(image):
   ret1, th1 = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,10)
   #ret2, th2 = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,3,10)
   print('Umbral de th1:', ret1)
 
    #ret,thresh = cv2.threshold(image,165,255,cv2.THRESH_BINARY)

   return th1

def gauss(image):
    return cv2.GaussianBlur(image, (3,3), 0)

def canny(image, low_threshold, high_threshold):
    edges = cv2.Canny(image, low_threshold, high_threshold)
    cv2.imshow('Lane edges',edges)
    return edges

def region(image):
    height, width = image.shape
    triangle = np.array([
       # [(500, 1000), (1000, 680), (1400, 1000)]#Triangulo PruebaTarde
       # [(500, 1000), (1100, 580), (1700, 1000)]#Triangulo Prueba
       #  [(1350, 1980), (2200, 1370), (3560, 2080)]#Triangulo Demo
        # [(150,520),(360,350),(600,520)]
        [(0,520),(250,200),(350,200) ,(600,520)]
    ], np.int32)

    mask = np.zeros_like(image)
    mask = cv2.fillPoly(mask, [triangle], 150)
    masked = cv2.bitwise_and(image, mask)
    return masked

def encontrar_lineas_verticales(edges):
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=20, maxLineGap=100)
    vertical_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(x2 - x1) > 0:
                slope = (y2 - y1) / (x2 - x1)
                if abs(slope) > 0.5:
                    vertical_lines.append(line)
    return vertical_lines

def proyectar_lineas(img, lines, num_lines=2):
    lines_image = np.zeros_like(img)
    if lines is not None:
        lines = sorted(lines, key=lambda x: np.arctan2(x[0, 3] - x[0, 1], x[0, 2] - x[0, 0]))

        right_line = None
        left_line = None

        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / (x2 - x1)

            if abs(slope) > 0.5:
                if slope > 0 and right_line is None:
                    right_line = line
                elif slope < 0 and left_line is None:
                    left_line = line

        if right_line is not None:
            x1, y1, x2, y2 = right_line[0]
            cv2.line(lines_image, (x1, y1), (x2, y2), (255, 0, 0), 5)

        if left_line is not None:
            x1, y1, x2, y2 = left_line[0]
            cv2.line(lines_image, (x1, y1), (x2, y2), (255, 0, 0), 5)

    return lines_image

def imprimir_coordenadas_x(lines):
    for i, line in enumerate(lines):
        x1, _, x2, _ = line[0]
        x_position = (x1 + x2) // 2
        print(f'Línea {i + 1} - Coordenada X: {x_position}')

def proyectar_circulo_y_lineas(img, lines):
    if lines:
        # Filtrar líneas verticales
        vertical_lines = [line for line in lines if abs((line[0][0] + line[0][2]) // 2 - img.shape[1] // 2) < img.shape[1] // 4]

        if len(vertical_lines) >= 4:  # Cambiado a 4 líneas
            # Ordenar las líneas por su distancia al centro
            vertical_lines.sort(key=lambda line: abs((line[0][0] + line[0][2]) // 2 - img.shape[1] // 2))

            # Tomar las cuatro líneas más cercanas al centro
            selected_lines = vertical_lines[:4]  # Cambiado a 4 líneas

            # Calcular el promedio de las coordenadas en X
            x_avg = int(np.mean([(line[0][0] + line[0][2]) // 2 for line in selected_lines]))

            # Obtener la altura para proyectar el círculo
            height, _ = img.shape[:2]
            y_position = int(height * (1 - 1/9))

            # Dibujar un círculo púrpura en la copia de la imagen original
            img_copy = img.copy()
            cv2.circle(img_copy, (x_avg, y_position), 10, (255, 0, 255), -1)

            # Dibujar las líneas azules en la copia de la imagen original
            for line in selected_lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(img_copy, (x1, y1), (x2, y2), (255, 0, 0), 5)

            return img_copy, x_avg

    print("No hay suficientes líneas para proyectar el círculo y las líneas azules.")
    return img
        

max_window_width = 1920  # Adjust this to your screen width
max_window_height = 1080  # Adjust this to your screen height

class ZED2ImageSubscriber:
    def __init__(self):
        rospy.init_node('zed2_image_subscriber', anonymous=True)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('/zed2/zed_node/left/image_rect_color', Image, self.image_callback)
        self.steering_publisher = rospy.Publisher('AMR_Steering', Int8, queue_size=10)
        self.frame_width = 640
        self.frame_height =  360



    def proccess_Image(self, or_frame):
        copy = np.copy(or_frame)
        New_copy=resize(copy)
        grey_img = grey(New_copy)
        cv2.imshow("GREY", grey_img)
    #  Binary = binaryOtsu(grey_img)
        
    # gaussian = gauss(grey_img)
        #edges = canny(gaussian, 90, 100)
    # isolated_region = region(edges)# nuevo orden
    #  Binary = binaryOtsu(isolated_region)
        gaussian = gauss(grey_img)
        cv2.imshow("gaussian", gaussian)
        img_binarizada = binarizar_imagen(gaussian)
        cv2.imshow("img_binarizada", img_binarizada)
        img_adelgazada = adelgazar_lineas_verticales(img_binarizada)
        edges = canny(img_adelgazada, 20, 100)  # Ajustar estos valores según sea necesario
        isolated_region = region(edges)
        cv2.imshow("isolated_region", isolated_region)
    # Hough = cv2.HoughLinesP(isolated_region, 1, np.pi/180, 20, np.array([]), minLineLength=10, maxLineGap=50)
    # result= draw_lane_lines(copy, lane_lines(copy, Hough))
    # Find vertical lines in the thinned image
        lines_verticales = encontrar_lineas_verticales(isolated_region)

    # Proyectar un círculo y líneas azules en las coordenadas calculadas
        
        if lines_verticales:

            Result, center_x=proyectar_circulo_y_lineas(New_copy, lines_verticales)
            error = center_x - (640/2)
            print("center_x")
            print(center_x)

            print("window_width")
            print(640/2)

            # Aplicar el control proporcional
            control_signal = Kp * error
            print("error: ", error)
            print("control signal: ", control_signal)


            # Display the processed frame
            cv2.namedWindow('Lane Detection', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Lane Detection',max_window_width,max_window_height)
            cv2.imshow('Lane Detection',Result)  # Display the edges here
            
            scaled_control = max(0, min(100, 50 + control_signal))
        else:
            scaled_control = 50
    
        return scaled_control




    def image_callback(self, msg):
        print("a")
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            print(e)
            return

        # Display the image
        cv2.imshow("ZED2 Left Camera", cv_image)
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
