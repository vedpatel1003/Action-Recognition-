import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import torchvision.transforms as transforms

# Define the path to the image file
demo_num = '0/'
directory_path = 'datasets/clapping/' + demo_num
file_name = '0.jpg'
file_path = directory_path + file_name

# Create the folder if it doesn't exist
if not os.path.exists(directory_path):
    os.makedirs(directory_path)

# Define the video capture device
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

count = 0
while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Preprocess the frame to match the input format of the ResNet-50 model
    frame_new = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_new = cv2.resize(frame_new, (224, 224))
    cv2.imshow('frame', frame_new)
    if cv2.waitKey(1) == ord('q'):
        break

    cv2.imwrite(file_path, frame_new)
    count += 1
    file_path = directory_path + str(count) + '.jpg'

# Release the video capture device and close the OpenCV window
cap.release()
cv2.destroyAllWindows()
