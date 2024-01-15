# Lane-Following-Control

As a part of the project to develop an Autonomous Mobile Robot, a lane follower was implemented. Nevertheless, because on the campus where the car is supposed to drive, there are a large number of different scenarios, and in a lot of them is complicated to implement a  line follower with simple line detection algorithms, I decided to implement a Pix2Pix Architecture, which consists in generating an image from a conditioned input, which is also an image, and in this way achieving to obtain an image with the lines drawn.

In the following diagram, you can see how the implementation was made.


![Implementation](https://github.com/alejandro3141592/Lane-Following-Control/assets/132953325/9d2a5cef-8133-4f11-92e5-b7250c7a3f92)


First, the Data Set Generation was made, for this, a video was recorded traveling around the campus, after this, on every frame we manually drew the lines of the lanes, and we saved both the original frame and the output frame. 

![Pix2PixImage2](https://github.com/alejandro3141592/Lane-Following-Control/assets/132953325/ea04761f-0dfa-46b3-9cec-019eeeec8e4a)

![Pix2PixImage](https://github.com/alejandro3141592/Lane-Following-Control/assets/132953325/d303b43f-2461-49be-b3e6-eeb275c3a0d5)

Then a process of random jitter and crop was made to perform a data augmentation on the data set. After this, the Neural Network was coded, which included a Generator, in charge of generating the images with the lines, and a Discriminator, in charge of identifying if the input image is real, from the original data set, or if it is produced by the Generator. With these we can compute some loss functions and train or model, to finally evaluate the results of the network.
![Captura de pantalla 2024-01-14 232141](https://github.com/alejandro3141592/Lane-Following-Control/assets/132953325/46410cee-f376-4531-96a7-e8924b21e8a5)


After the training process, the model was evaluated with several pre-recorded videos, here is an example:
![Lineas Dibujadas ‐ Hecho con Clipchamp](https://github.com/alejandro3141592/Lane-Following-Control/assets/132953325/67504e5e-9a81-43af-a700-f3a601e6975f)

Finally, the output of the Pix2Pix model was used in a line detection algorithm, here is a video of the whole environment running:
https://github.com/alejandro3141592/Lane-Following-Control/assets/132953325/889dcc8a-f6f7-4b98-88a3-ff7282423232

And here is a video of the results obtained from testing the algorithm in the vehicle:
![Enviroment ‐ Hecho con Clipchamp (3)](https://github.com/alejandro3141592/Lane-Following-Control/assets/132953325/ba55ca40-7ca5-4fb2-8aa1-2d96ecc387a2)


