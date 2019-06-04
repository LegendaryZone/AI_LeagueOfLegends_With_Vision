
from PIL import ImageGrab
import numpy as np
import cv2
import argparse
import pyautogui
import time

# Import packages
import os
import cv2
import numpy as np
import tensorflow as tf
import argparse
import sys

pyautogui.FAILSAFE = False

import keyboard
import string
# from xml_to_csv import main

from threading import *
from PIL import ImageGrab


keys = list(string.ascii_lowercase)

# Set up camera constants
IM_WIDTH = 1920
IM_HEIGHT = 1080

STD_DIMENSIONS =  {
    "480p": (640, 480),
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "4k": (3840, 2160),
}


# This is needed since the working directory is the object_detection folder.
sys.path.append('..')

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'ezreal_graph'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'training', 'object-detection.pbtxt')

# Number of classes the object detector can identify
NUM_CLASSES = 5

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)


# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')


# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()
font = cv2.FONT_HERSHEY_SIMPLEX


# Initialize USB webcam feed
camera = cv2.VideoCapture(0)
ret = camera.set(3,IM_WIDTH)
ret = camera.set(4,IM_HEIGHT)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))

total_image_data = []
image_data = []

ezreal_x_position = []
ezreal_y_position = []
image_count = 0
count = 0

key_data = []

def listen(key):
    while True:
        keyboard.wait(key)
        if key == "q":
            key_dir = "q"
            key_data.append(key_dir)

            print(key_data)
        print("[+] pressed", key)
        time.sleep(1)

threads = [Thread(target=listen, kwargs={"key":key}) for key in keys]
for thread in threads:
    thread.start()

try: 
    os.mkdir("new_images/")
    print("A folder is created")

except:
    pass

while True:

        image_count += 1
        # print("image_count", image_count)

        t1 = cv2.getTickCount()

        # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
        # i.e. a single-column array, where each item in the column has the pixel RGB value
        # ret, frame = camera.read()
        # frame_expanded = np.expand_dims(frame, axis=0)
        # print(frame_expanded.shape)
        printscreen = np.array(ImageGrab.grab(bbox = (0, 400, 1000, 1200)))
            

        printscreen = cv2.cvtColor(printscreen, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(printscreen, (331,331))
        # print(frame.shape)

        frame_expanded = np.array(frame).reshape(1, 331, 331, 3)


        # Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: frame_expanded})
        # print(scores)


        # Draw the results of the detection (aka 'visulaize the results')
        image, x_coord, y_coord, x2_coord, y2_coord = vis_util.visualize_boxes_and_labels_on_image_array(
            frame,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=3,
            min_score_thresh=0.9)

            

        if image_count <= 10:

            cv2.imwrite("./new_images/new_images" + "." + str(round(t1)) + ".jpg", frame[int(y_coord)-50:int(y2_coord)+50, int(x_coord)-50:int(x2_coord)+50])
            print("A picture {} is taken".format(image_count))        print("Ezreal", x_coord, y_coord)

        # creator csv file and tfrecord file.
        # if image_count == 10:
        #     main()
        #     print("created a new tf.record")

        # Flask sending this data to another desktop
        
        if len(key_data) == 1:

            key_data = []
            count += 1
            print("count", count)

            if x_coord == 0.0 and y_coord ==0.0:
                pass

            else:

                x_data = (x_coord + x2_coord) / 2
                y_data = (y_coord + y2_coord) / 2 

                # Ezreal Position
                ezreal_x_position.append(x_data)
                ezreal_y_position.append(y_data)
                print("x position", ezreal_x_position)
                print("y position", ezreal_y_position)

                print("the number of image_data" , len(image_data))

                if len(ezreal_x_position) > 1:

                    # move to the right side
                    if ezreal_x_position[count-1] - ezreal_x_position[(count-2)] > 0 or ezreal_y_position[count-1] - ezreal_y_position[(count-2)] < 0:

                        first_index = 1
                        image_data.append([frame, first_index])
                        # print("image_data1", image_data)
                        print("move to the right side or downside",
                        ezreal_x_position[count-1] - ezreal_x_position[(count-2)], ezreal_y_position[count-1] - ezreal_y_position[(count-2)])

                    # move to the left side
                    if ezreal_x_position[count-1] - ezreal_x_position[(count-2)] < 0 or ezreal_y_position[count-1] - ezreal_y_position[(count-2)] > 0:

                        second_index = 2
                        image_data.append([frame, second_index])
                        print("move to the left side or upside",
                        ezreal_x_position[count-1] - ezreal_x_position[(count-2)], ezreal_y_position[count-1] - ezreal_y_position[(count-2)])


                    # move forward
                    if ezreal_x_position[count-1] - ezreal_x_position[(count-2)] == 0 or ezreal_y_position[count-1] - ezreal_y_position[(count-2)] == 0:

                        third_index = 0
                        image_data.append([frame, third_index])
                        print("doesn't move",
                        ezreal_x_position[count-1] - ezreal_x_position[(count-2)], ezreal_y_position[count-1] - ezreal_y_position[(count-2)])


        cv2.putText(frame,"FPS: {0:.2f}".format(frame_rate_calc),(30,50),font,1,(255,255,0),2,cv2.LINE_AA)

        # All the results have been drawn on the frame, so it's time to display it.
        cv2.imshow('Object detector', frame)

        t2 = cv2.getTickCount()
        time1 = (t2-t1)/freq
        frame_rate_calc = 1/time1
        out.write(frame)

        if count >= 10:
            image_data = np.array(image_data)
            np.save("image_data.npy", image_data)
            print(image_data, image_data.shape)

            break
        # Press 'q' to quit
        if cv2.waitKey(10) == ord('w'):
            break

camera.release()
cv2.destroyAllWindows()

