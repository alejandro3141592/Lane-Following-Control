#!/usr/bin/env python

# Test a trained model of pix2pix with the images adquired with a ZED CAM, using ROS.


import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import tensorflow as tf
import os
import pathlib
import time
import datetime

from matplotlib import pyplot as plt
from IPython import display

import numpy as np
import cv2

def load(image_file):
  # Read and decode an image file to a uint8 tensor
  image = tf.io.read_file(image_file)
  image = tf.io.decode_jpeg(image)
  
  # Split each image tensor into two tensors:
  # - one with a real building facade image
  # - one with an architecture label image 
  w = tf.shape(image)[1]
  w = w // 2
  real_image = image[:, w:, :]
  input_image = image[:, :w, :]
#   print("Shape")
#   print(tf.shape(input_image))

# Threshold the image
  #real_image = tf.where(real_image < 230, 0, 255)

  # Convert both images to float32 tensors
  input_image = tf.cast(input_image, tf.float32)

  real_image = real_image
  real_image = tf.cast(real_image, tf.float32)

  return input_image, real_image




PATH = '/home/alejandro/catkin_ws/src/lane_following/src/SeparatedData/Jardineras'



inp, re = load('/home/alejandro/catkin_ws/src/lane_following/src/SeparatedData/Jardineras/test/concatenated_frame_2.jpg')




# The facade training set consist of 400 images
BUFFER_SIZE = 4806
# The batch size of 1 produced better results for the U-Net in the original pix2pix experiment
BATCH_SIZE = 1
# Each image is 256x256 in size
IMG_WIDTH = 256
IMG_HEIGHT = 256

def resize(input_image, real_image, height, width):
  input_image = tf.image.resize(input_image, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  real_image = tf.image.resize(real_image, [height, width],
                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  return input_image, real_image

def random_crop(input_image, real_image):
  stacked_image = tf.stack([input_image, real_image], axis=0)
  cropped_image = tf.image.random_crop(
      stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 1])

  return cropped_image[0], cropped_image[1]


# Normalizing the images to [-1, 1]
def normalize(input_image, real_image):
  input_image = (input_image / 127.5) - 1
  real_image = (real_image / 127.5) - 1

  return input_image, real_image

# Normalizing the images to [-1, 1]
def normalize(input_image, real_image):
  input_image = (input_image / 127.5) - 1
  real_image = (real_image / 127.5) - 1

  return input_image, real_image



def load_image_train(image_file):
  input_image, real_image = load(image_file)
  input_image, real_image = random_jitter(input_image, real_image)
  input_image, real_image = normalize(input_image, real_image)

  return input_image, real_image



def load_image_test(image_file):
  input_image, real_image = load(image_file)
  input_image, real_image = resize(input_image, real_image,
                                   IMG_HEIGHT, IMG_WIDTH)
  input_image, real_image = normalize(input_image, real_image)

  return input_image, real_image







@tf.function()
def random_jitter(input_image, real_image):
  # Resizing to 286x286
  input_image, real_image = resize(input_image, real_image, 286, 286)

  # Random cropping back to 256x256
  input_image, real_image = random_crop(input_image, real_image)

  if tf.random.uniform(()) > 0.5:
    # Random mirroring
    input_image = tf.image.flip_left_right(input_image)
    real_image = tf.image.flip_left_right(real_image)

  return input_image, real_image









train_dataset = tf.data.Dataset.list_files('/home/alejandro/catkin_ws/src/lane_following/src/SeparatedData/Jardineras/training/*')
train_dataset = train_dataset.map(load_image_train,
                                  num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE)



test_dataset = tf.data.Dataset.list_files('/home/alejandro/catkin_ws/src/lane_following/src/SeparatedData/Jardineras/test/*')
test_dataset = test_dataset.map(load_image_test)
test_dataset = test_dataset.batch(BATCH_SIZE)



OUTPUT_CHANNELS = 1


def downsample(filters, size, apply_batchnorm=True):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())

  result.add(tf.keras.layers.LeakyReLU())

  return result


down_model = downsample(3, 4)
down_result = down_model(tf.expand_dims(inp, 0))
#print (down_result.shape)



def upsample(filters, size, apply_dropout=False):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
    tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))

  result.add(tf.keras.layers.BatchNormalization())

  if apply_dropout:
      result.add(tf.keras.layers.Dropout(0.5))

  result.add(tf.keras.layers.ReLU())

  return result



up_model = upsample(3, 4)
up_result = up_model(down_result)
#print (up_result.shape)


def Generator():
  inputs = tf.keras.layers.Input(shape=[256, 256, 1])

  down_stack = [
    downsample(64, 4, apply_batchnorm=False),  # (batch_size, 128, 128, 64)
    downsample(128, 4),  # (batch_size, 64, 64, 128)
    downsample(256, 4),  # (batch_size, 32, 32, 256)
    downsample(512, 4),  # (batch_size, 16, 16, 512)
    downsample(512, 4),  # (batch_size, 8, 8, 512)
    downsample(512, 4),  # (batch_size, 4, 4, 512)
    downsample(512, 4),  # (batch_size, 2, 2, 512)
    downsample(512, 4),  # (batch_size, 1, 1, 512)
  ]

  up_stack = [
    upsample(512, 4, apply_dropout=True),  # (batch_size, 2, 2, 1024)
    upsample(512, 4, apply_dropout=True),  # (batch_size, 4, 4, 1024)
    upsample(512, 4, apply_dropout=True),  # (batch_size, 8, 8, 1024)
    upsample(512, 4),  # (batch_size, 16, 16, 1024)
    upsample(256, 4),  # (batch_size, 32, 32, 512)
    upsample(128, 4),  # (batch_size, 64, 64, 256)
    upsample(64, 4),  # (batch_size, 128, 128, 128)
  ]

  initializer = tf.random_normal_initializer(0., 0.02)
  last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation='tanh')  # (batch_size, 256, 256, 3)

  x = inputs

  # Downsampling through the model
  skips = []
  for down in down_stack:
    x = down(x)
    skips.append(x)

  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    x = tf.keras.layers.Concatenate()([x, skip])

  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)

generator = Generator()
tf.keras.utils.plot_model(generator, show_shapes=True, dpi=64)


gen_output = generator(inp[tf.newaxis, ...], training=False)


LAMBDA = 100



loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)



def generator_loss(disc_generated_output, gen_output, target):
  gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

  # Mean absolute error
  l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

  total_gen_loss = gan_loss + (LAMBDA * l1_loss)

  return total_gen_loss, gan_loss, l1_loss


def Discriminator():
  initializer = tf.random_normal_initializer(0., 0.02)

  inp = tf.keras.layers.Input(shape=[256, 256, 1], name='input_image')
  tar = tf.keras.layers.Input(shape=[256, 256, 1], name='target_image')

  x = tf.keras.layers.concatenate([inp, tar])  # (batch_size, 256, 256, channels*2)

  down1 = downsample(64, 4, False)(x)  # (batch_size, 128, 128, 64)
  down2 = downsample(128, 4)(down1)  # (batch_size, 64, 64, 128)
  down3 = downsample(256, 4)(down2)  # (batch_size, 32, 32, 256)

  zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (batch_size, 34, 34, 256)
  conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                kernel_initializer=initializer,
                                use_bias=False)(zero_pad1)  # (batch_size, 31, 31, 512)

  batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

  leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

  zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (batch_size, 33, 33, 512)

  last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                kernel_initializer=initializer)(zero_pad2)  # (batch_size, 30, 30, 1)

  return tf.keras.Model(inputs=[inp, tar], outputs=last)


discriminator = Discriminator()
tf.keras.utils.plot_model(discriminator, show_shapes=True, dpi=64)

disc_out = discriminator([inp[tf.newaxis, ...], gen_output], training=False)


def discriminator_loss(disc_real_output, disc_generated_output):
  real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

  generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

  total_disc_loss = real_loss + generated_loss

  return total_disc_loss

generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)


checkpoint_dir = '/home/alejandro/catkin_ws/src/lane_following/src/training_checkpoints/AvenidaTec2'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)



def generate_images(model, test_input, tar):
  prediction = model(test_input, training=True)
  plt.figure(figsize=(15, 15))

  display_list = [test_input[0], tar[0], prediction[0]]
  title = ['Input Image', 'Ground Truth', 'Predicted Image']

  for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.title(title[i])
    # Getting the pixel values in the [0, 1] range to plot.
    plt.imshow(display_list[i] * 0.5 + 0.5)
    plt.axis('off')
  plt.show()
  return prediction


def predict(model, test_input):
  prediction = model(test_input, training=True)
  return prediction



log_dir="logs/"

summary_writer = tf.summary.create_file_writer(
  log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))


print(tf.train.latest_checkpoint(checkpoint_dir))
checkpoint.restore("/home/alejandro/catkin_ws/src/lane_following/src/training_checkpoints/Estacionamiento/ckpt-4")

# Run the trained model on a few examples from the test set
# for inp, tar in test_dataset.take(5):
#   generate_images(generator, inp, tar)


# plt.show()

def apply_lane_detection(image):
    # Step 1: Convert the image to grayscale


    # Step 2: Apply Gaussian blur to the grayscale image
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    # Step 3: Apply Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Step 4: Define a region of interest (ROI)
    height, width = edges.shape
    roi_vertices = [(0, 0), (0, height), (width, 0), (width, height)]
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, [np.array(roi_vertices, np.int32)], 255)
    # masked_edges = cv2.bitwise_and(edges, mask)
    # cv2.imshow("masked_edges", masked_edges)
    # Step 5: Apply Hough Transform to detect lines
    lines = cv2.HoughLinesP(edges,1,np.pi/180,threshold=90, minLineLength=50, maxLineGap=10)
    
    # Step 6: Draw detected lines on the original image
    line_image = np.zeros_like(image).astype(np.uint8)
    if lines is not None:
        for points in lines:
            # Extracted points nested in the list
            x1,y1,x2,y2=points[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 255, 255), 5)
            # print(x1)

    #cv2.imshow("lines", line_image)
    # Step 7: Combine the original image with the detected lines
    result = cv2.addWeighted(image, 0.8, line_image, 1, 0)

    return result


def preprocess_frame(frame):
    # Resize the frame to the desired input size

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    frame = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
    # Normalize the pixel values to the range [-1, 1]
    frame = (frame / 127.5) - 1.0
    # Expand dimensions to match the model's expected input shape
    frame = tf.expand_dims(frame, axis=0)
    return frame


# Function to generate and display images from video frames
def generate_and_display_from_video(generator_model, frame):


    # Preprocess the frame
    input_frame = preprocess_frame(frame)
    cv2.imshow("input_frame", frame)

    # Generate an image using the generator
    generated_image = generator_model(input_frame, training=True).numpy()*255
    




    numpy_image = np.squeeze(generated_image)
    numpy_image = (numpy_image - np.min(numpy_image)) / (np.max(numpy_image) - np.min(numpy_image))


    numpy_image = (numpy_image*255).astype(np.uint8)
    

    # Display the frame in the "Video" window
    cv2.imshow("Video", numpy_image)

    return numpy_image

    #result = apply_lane_detection(numpy_image)

    #cv2.imshow("Result", result)




class ZED2ImageSubscriber:
    def __init__(self):
        rospy.init_node('zed2_image_subscriber', anonymous=True)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('/zed2/zed_node/left/image_rect_color', Image, self.image_callback)
        self.image_pub = rospy.Publisher("/Pix2Pix/Image", Image, queue_size=10)
        self.bridge = CvBridge()
        self.rate = rospy.Rate(100) 

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            print(e)
            return

        # Display the image
        cv2.imshow("ZED2 Left Camera", cv_image)

        # Check for the ESC key
        key = cv2.waitKey(1)

        numpy_image = generate_and_display_from_video(generator, cv_image)

        ros_image_msg = self.bridge.cv2_to_imgmsg(numpy_image, encoding="mono8")
        self.image_pub.publish(ros_image_msg)
        self.rate.sleep()

        if key == 27:  # 27 corresponds to the ASCII code for the ESC key
            rospy.signal_shutdown("ESC key pressed")


def main():
    try:
        
        zed2_subscriber = ZED2ImageSubscriber()
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
        cv2.destroyAllWindows()

main()