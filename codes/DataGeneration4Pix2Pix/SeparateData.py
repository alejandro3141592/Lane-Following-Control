# Read a folder of the Data directory, then reads all the images in the folder and, separete the data in 
# 2/3 for training, 1/6 for test and 1/6 for validation. It writes the images into the SeparateData Directory

from PIL import Image
import os
import shutil
import random

def resize_and_split_dataset(input_folder, output_folder, output_size=(256, 256), train_ratio=2/3, test_ratio=1/6, validation_ratio=1/6):
    # Create output folders if they don't exist
    for folder in ['training', 'test', 'validation']:
        folder_path = os.path.join(output_folder, folder)
        os.makedirs(folder_path, exist_ok=True)

    # List all files in the input folder
    all_files = os.listdir(input_folder)
    total_files = len(all_files)

    # Calculate the number of files for each split
    train_size = int(total_files * train_ratio)
    test_size = int(total_files * test_ratio)
    validation_size = total_files - train_size - test_size

    # Shuffle the list of files
    random.shuffle(all_files)

    # Resize and move files to training folder
    for file_name in all_files[:train_size]:
        source_path = os.path.join(input_folder, file_name)
        destination_path = os.path.join(output_folder, 'training', file_name)
        resize_and_move(source_path, destination_path, output_size)

    # Resize and move files to test folder
    for file_name in all_files[train_size:train_size + test_size]:
        source_path = os.path.join(input_folder, file_name)
        destination_path = os.path.join(output_folder, 'test', file_name)
        resize_and_move(source_path, destination_path, output_size)

    # Resize and move files to validation folder
    for file_name in all_files[train_size + test_size:]:
        source_path = os.path.join(input_folder, file_name)
        destination_path = os.path.join(output_folder, 'validation', file_name)
        resize_and_move(source_path, destination_path, output_size)

def resize_and_move(source_path, destination_path, output_size):
    # Resize the image
    image = Image.open(source_path)
    resized_image = image.resize(output_size)

    # Save the resized image
    resized_image.save(destination_path)


if __name__ == "__main__":
    # Replace 'input_folder' with the path to your folder containing images
    input_folder = '/home/alejandro/catkin_ws/src/lane_following/src/Data/AvenidaAumented'
    
    # Replace 'output_folder' with the path where you want to create the training, test, and validation folders
    output_folder = '/home/alejandro/catkin_ws/src/lane_following/src/SeparatedData/AvenidaAumented'

    output_size = (256*2, 256)

    resize_and_split_dataset(input_folder, output_folder, output_size)