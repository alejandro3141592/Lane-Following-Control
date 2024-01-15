import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class VideoSubscriber:
    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/Pix2Pix/Image", Image, self.image_callback)
        self.video_writer = None

    def image_callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except Exception as e:
            print(e)
            return

        if self.video_writer is None:
            # Define the codec and create a VideoWriter object
            height, width, channels = cv_image.shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'XVID' for older versions of OpenCV
            self.video_writer = cv2.VideoWriter("EStacionamiento.mp4", fourcc, 20.0, (width, height))

        # Write the frame to the video file
        self.video_writer.write(cv_image)

        # Display the image (optional)
        cv2.imshow("Video", cv_image)
        cv2.waitKey(1)

def main():
    rospy.init_node('video_subscriber', anonymous=True)
    video_subscriber = VideoSubscriber()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
        if video_subscriber.video_writer is not None:
            video_subscriber.video_writer.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
