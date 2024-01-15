#Reads a video, displays it frame by frame, and the user draws 2 white lines into the image,
#then the program save the original image and the edited one as one image, stacked horizontally



import cv2
import numpy as np

# Global variables to store the coordinates of the two lines
line1_start, line1_end = None, None
line2_start, line2_end = None, None

# Flag to indicate whether Enter key is pressed
enter_pressed = False
a_key_pressed = False

# Flag to indicate which line is currently being drawn (1 or 2)
current_line = 1

# Counter for naming the saved images
image_counter = 9443

# Mouse callback function for drawing lines
def draw_lines(event, x, y, flags, param):
    global line1_start, line1_end, line2_start, line2_end, a_key_pressed, current_line

    if a_key_pressed:
        # Reset the flag and line coordinates
        a_key_pressed = False
        # line1_start, line1_end, line2_start, line2_end = None, None, None, None

    if event == cv2.EVENT_LBUTTONDOWN:
        # Left mouse button clicked - start drawing line
        if current_line == 1:
            line1_start = (x, y)
        elif current_line == 2:
            line2_start = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        # Left mouse button released - end drawing line
        if current_line == 1:
            line1_end = (x, y)
        elif current_line == 2:
            line2_end = (x, y)

def display_video(video_path, start_frame=900):
    global current_line, enter_pressed, a_key_pressed, image_counter

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Couldn't open the video file.")
        return

    # Set the starting frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Create a window and set the mouse callback function
    cv2.namedWindow('Resized Frame')
    cv2.setMouseCallback('Resized Frame', draw_lines)

    ret, frame = cap.read()
    while True:
        # Read a frame from the video
        if enter_pressed:
            ret, frame = cap.read()
            image_counter +=1
            

        if not ret:
            print("End of video.")
            break

        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Resize the frame to 320 x 180 pixels
        resized_frame = cv2.resize(gray_frame, (256*2, 256*2))
        black_frame = np.zeros_like(resized_frame)
        copy_frame = resized_frame.copy()
        lines_frame_color = resized_frame.copy()


     
        # Draw the first line if the start and end points are defined
        if line1_start is not None and line1_end is not None:
            cv2.line(lines_frame_color, line1_start, line1_end, (255, 255, 255), 8)

        # Draw the second line if the start and end points are defined
        if line2_start is not None and line2_end is not None:
            cv2.line(lines_frame_color, line2_start, line2_end, (255, 255, 255), 8)
   

        # Concatenate the resized frame with lines drawn onto it
        concatenated_frame_color = np.concatenate((copy_frame, lines_frame_color), axis=1)
        

        # Display the concatenated frame in a new window
        cv2.imshow('Concatenated Frame', concatenated_frame_color)
        

        # Display the resized frame
        cv2.imshow('Resized Frame', resized_frame)

        # Save the concatenated frame to a .jpg file with a changing name
        if enter_pressed:
            filename = f"/home/alejandro/catkin_ws/src/lane_following/src/Data/AvenidaAumented/concatenated_frame_{image_counter}.jpg"
            cv2.imwrite(filename, concatenated_frame_color)
            print(f"Saved {filename}")
            enter_pressed = False



        # Wait for user input
        key = cv2.waitKey(1) 

        # Check if the user pressed 'Space bar' (32) or Enter (13) or Esc (27)
        if key == 32:  # 'Space bar' key
            # Set the flag to indicate 'Space bar' key is pressed
            a_key_pressed = True

            # Toggle between drawing the first and second lines
            current_line = 3 - current_line  # Toggle between 1 and 2

        elif key == 13:  # Enter key
            # Set the flag to indicate Enter key is pressed
            enter_pressed = True

            # Toggle between drawing the first and second lines

        elif key == 27:  # Esc key
            break

    # Release the video capture object and close the windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = '/home/alejandro/Downloads/AMR25Nov.mp4'  # Replace with the path to your AVI file
    start_frame = 1510  # Replace with the desired starting frame
    display_video(video_path, start_frame)
