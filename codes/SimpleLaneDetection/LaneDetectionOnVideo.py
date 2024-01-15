#Reads a video, on each frame performs a lane detection, and send a steering messafe to AMR via Serial 

#Importing Libraries
import numpy as np
import cv2
import serial

#Color Conversion function
def grey(image):
   R, G, B = image[:, :, 0], image[:, :, 1], image[:, :, 2]
   imageG = B
   imageG.dtype = np.uint8
   return imageG

#Image Smoothing function
def gauss(image):
    return cv2.GaussianBlur(image, (3, 3), 0)

#Edge Detection
def canny(image, low_threshold, high_threshold):
    edges = cv2.Canny(image, low_threshold, high_threshold)
    return edges

#Region Segmentation
def region(image):
    height, width = image.shape
    triangle = np.array([
         [(0, 380), (320, 0), (640, 380)]
    ], np.int32)

    mask = np.zeros_like(image)
    mask = cv2.fillPoly(mask, [triangle], 150)
    masked = cv2.bitwise_and(image, mask)
    return masked


def average(image, lines):
    left = []
    right = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        y_int = parameters[1]
        if slope < 0:
            left.append((slope, y_int))
        else:
            right.append((slope, y_int))
    right_avg = np.average(right, axis=0)
    left_avg = np.average(left, axis=0)
    left_line = make_points(image, left_avg)
    right_line = make_points(image, right_avg)
    return np.array([left_line, right_line])

def make_points(image, average): 
    slope, y_int = average 
    y1 = image.shape[0]
    y2 = int(y1 * (3/5))
    x1 = int((y1 - y_int) / slope)
    x2 = int((y2 - y_int) / slope)
    return np.array([x1, y1, x2, y2])

def display_lines(image, lines):
    lines_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line
            cv2.line(lines_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return lines_image


    

def send_to_serial(id, len, data):
    try:
        
        string_to_send = f'{id} {len} {data}'
        print(string_to_send)
        serial_port.write(string_to_send)

        print("Sent data to serial port: %d", data)
    except serial.SerialException as e:
        print("Serial port error: %s", str(e))


#Configuración del puerto serial

video_path = "/home/alejandro/catkin_ws/src/lane_following/src/TestVideos/AMR1Nov.avi"
start_frame = 9000
video = cv2.VideoCapture(video_path)

video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
serial_port = '/dev/ttyUSB0'  # Cambia esto al nombre de tu puerto serial
baud_rate = 9600  # Ajusta la velocidad del puerto serial según tus necesidades
ser = serial.Serial(serial_port, baud_rate)

#video = cv2.VideoCapture(3)

# Get the frame width and height
frame_width = int(video.get(3))
frame_height = int(video.get(4))

# Calculate the aspect ratio of the frames
frame_aspect_ratio = frame_width / frame_height

# Set the maximum window width and height based on your screen resolution
max_window_width = 1920  # Adjust this to your screen width
max_window_height = 1080  # Adjust this to your screen height

# Calculate the window width and height while maintaining the aspect ratio
window_width = min(frame_width, max_window_width)
window_height = int(window_width / frame_aspect_ratio)

Kp = 0.07 #constante proporcional
left_line = np.array([0, 0, 0, 0])
right_line = np.array([0, 0, 0, 0])

while True:
    ret, or_frame = video.read()
    if not ret:
        print("Error al capturar el marco.")
        continue

    



    cv2.imshow('FRame', or_frame)
    copy = np.copy(or_frame)
    grey_img = grey(copy)
    gaussian = gauss(grey_img)
    edges = canny(gaussian, 10, 10)
    isolated_region = region(edges)
    lines = cv2.HoughLinesP(isolated_region, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    #averaged_lines = average(copy, lines)
    #left_line = averaged_lines[0]
    #right_line = averaged_lines[1]
    if lines is not None and len(lines) > 0:
        averaged_lines = average(copy, lines)
        left_line = averaged_lines[0]
        right_line = averaged_lines[1]
   
    black_lines = display_lines(copy, averaged_lines)
    lanes = cv2.addWeighted(copy, 0.8, black_lines, 1, 1)
        
        # Calcular el centro de los carriles (center_x) y el error
    center_x = (left_line[0] + right_line[0]) / 2
    error = center_x - (window_width / 2)

        # Aplicar el control proporcional
    control_signal = Kp * error
    print("error: ", error)
    print("control signal: ", control_signal)
        

        # Escalar la señal de control al rango de 0 a 100
    scaled_control = max(0, min(100, 50 + control_signal))
    send_to_serial(83, 1, scaled_control)
    print(scaled_control)

        # Display the processed frame
    cv2.namedWindow('Lane Detection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Lane Detection', window_width, window_height)
    cv2.imshow('Lane Detection',lanes)  # Display the edges here

    key = cv2.waitKey(25)
    if key == 27:
        break

# Cerrar la conexión del puerto serial al salir del bucle
ser.close()
cv2.destroyAllWindows()
video.release()