import cv2
import torch
import os
import time
import uuid
import numpy as np
import tkinter as tk
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings("ignore")


def calculate_focal(distance_from_cam, head_width, img):
    """
    Uses OpenCV's face haar cascade which detects you head and returns corners indicating where your head is. With these coordinates, 
    this function creates a box around it and calculates the focal length of the camera which is required to calculate the distance later on.

    Parameters
    ----------

    distance_from_cam (float): Measured distance from camera given by user
    head_width (float): Measured width of users head 
    img (numpy.ndarray): Array indicating opencv image

    Returns
    -------
    focal (float): focal length of user's camera
    output.jpg: Test image with rectangle around face

    """

    # Extracts cascade and image dimensions
    face_classifier = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    image_height = img.shape[0]
    image_width = img.shape[1]

    # Uses haar cascade to find all faces.
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))

    # Draws a rectangle around your head and creates a new image
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.imwrite("output.jpg", img)

    # This formula calculates the focal length. It also normalizes it using the dimensions of the image which
    # is necessary since the video feed is not always the same size as the test image.
    focal = (w*(image_height/image_width)*distance_from_cam)/head_width
    return focal


def detect_bounding_box(vid, focal, head_width, head_boxes):
    """
    Uses video frame, focal length, user given head width, and a 2D list containing coordinates for each detected head
    to 
    1. calculate distance of head from camera
    2. draw rectangle around each head
    3. display this information to the user in the video feed.

    Parameters
    ----------

    vid (numpy.ndarray): Video frame
    focal (float): focal length of user's camera
    head_width (float): Measured width of users head 
    head_boxes (2D List): Contains coordinates for all instances of heads that are detected

    Returns
    -------
    vid (numpy.ndarray): Adjusted video frame with entire interface.
    """

    for i in range(len(head_boxes)):
        # Obtain the corners of head box. Confidence and class variables can be ignored.
        x1, y1, x2, y2, confidence, c = head_boxes[i]

        # Calculates width of head in pixels and uses this to calculate the distance from the camera. Makes distance 0 if head isn't detected
        w = x2-x1
        if(w == 0):
            dist = 0
        else:
            dist = (focal*head_width)/(w) # THE BIG FORMULA

        # Displays all this information on the original video frame
        text = "Object " + str(i) + " Distance: " + str(dist) + " Width: " +  str(w)
        cv2.rectangle(vid, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(vid, text, (10, i*25 + 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,0,0), 2)
    return vid

def image_capture(num_images, countdown_time):
    """
    Function for image collection. The user will follow this process:
    
    1. There is a tkinter gui that displays a 5 secondcountdown. Pose for the camera :). .
    2. Once the image is taken, move to a different spot and pose again.
    3. Do this for the amount of images you want to take

    Keep in mind, you want to give the model a wide variety of images.
    Get up close, go far away, look up, look down. You can even face away from the camera if you want the back of your head to be part of the dataset.
    Get some friends to help if you would like!
    The more, the better.

    Parameters
    ----------
    num_images (int): amount of images user wants to take for their dataset


    Returns
    -------
    Populates user indicated folder at path
    """

    # Creates the path if it does not already exist
    if not os.path.exists(os.path.join("data")):
        os.makedirs(os.path.join("data", "images"))
        os.makedirs(os.path.join("data", "labels"))

    path_to_images = os.path.join("data", "images")


    # Takes a live feed
    video_capture = cv2.VideoCapture(0)
    print("Collecting {} images for facial detection".format(num_images))

    for i in range(num_images):

        # Gui countdown
        visible_countdown(countdown_time, i+1)

        # "Image capture", we take the current frame after the wait from the countdown
        result, training_frame = video_capture.read()
        
        # Saves image in path. It creates a unique id for each image as well so there are no file mixups.
        image_name = os.path.join(path_to_images, 'image' + str(i+1) + '.' + str(uuid.uuid1())+'.jpg')
        cv2.imwrite(image_name, training_frame)

        print("Image: {}".format(i+1))
        
        # If you want to cancel early, press q
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

def visible_countdown(t, img_num):
    """
    Since it would be hard to tell when an image is "taken", we need a way to display that information to the user during the detection process.
    This uses a tkinter gui to display a countdown for the user for a time specified by them.

    Parameters
    ----------

    t (int): Amount of time between image takes.
    img_num (int): Indicates the image that we are on.
    """
    root = tk.Tk()
    root.attributes('-fullscreen', True)

    label = tk.Label(root, font=("Helvetica", 100))
    label.pack(expand=True)
    while t:
        text = "Image number: " + str(img_num) + "\n" + str(t)
        label.config(text=text)
        root.update()
        time.sleep(1)
        t -= 1

def facial_detection(dist, head_width, img, path_to_model):
    """
    MAIN FUNCTION:
    This loads the custom model you created, runs a live feed, passes each frame through the model which returns coordinates to the head box corners,
    extracts these coordinates, and uses these coordinates and the focal length to obtain a live video feed which displays the depth of all detected 
    heads from the camera

    Parameters
    ----------

    dist (float): Measured distance from camer given by user
    head_width (float): Measured width of users head 
    img (numpy.ndarray): Array indicating opencv test image needed for focal calibration
    path_to_model (str): Indicates path to model in yolov5 folder
    """

    focal = calculate_focal(float(dist), float(head_width), img)
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=path_to_model, force_reload=True)
    video_capture = cv2.VideoCapture(0)
    while True:
        head_box = [[0, 0, 0, 0, 0, 0]]

        result, video_frame = video_capture.read()
        if result is False:
            print("Video capture unsuccessful")
            break
            
        results = model(video_frame)
        bounding_box = results.xyxy[0].cpu().tolist()
        if bounding_box:
            head_box = []
            for boxes in range(len(bounding_box)):
                head_box.append([int(i) for i in bounding_box[boxes]])

        vid = detect_bounding_box(video_frame, focal, head_width, head_box)
        cv2.imshow("Distance Detection Test", vid)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # --- USER NEEDS TO CHANGE THESE VARIABLES ---

    ## FOCAL CALIBRATION
    # Values for focal calibration in inches
    distance_from_camera = 24.0
    head_width = 6.0

    # Test image needed for focal calibration
    img = cv2.imread('test.jpg')


    ## IMAGE COLLECTION
    # How many images you want to take
    img_count = 5

    # Amount of time between image captures
    countdown_time = 3


    ## MAIN FUNCTION
    # Indicates where your model is. If training is done properly, most likely only will need to change experiment folder
    path_to_model = os.path.join("yolov5", "runs", "train", "exp2", "weights", "best.pt")

    # --------------------------------------------

    #image_capture(int(img_count), int(countdown_time))
    #facial_detection(distance_from_camera, head_width, img, path_to_model)