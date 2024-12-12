# Documentation file for Head Depth Detection with YOLOv5 (HDDY5)

## Introduction:
Welcome to HDDY5! The purpose of this project is to be able to calculate how far a user is from a camera of their choosing. Computer vision is a rapidly developing field and is a vital part to the future of robotics. Assisting humans with tasks is a very important use for robotics, especially when it comes to space operations. The ability of a robot to tell how far the person they are assisting is from them is a vital part to assisting in tasks. This project is a very simple implementation which allows the user to collect their own dataset and train their own YOLOv5 model so that it can find their head and tell how far away it is.

## Necessary Repositories:
LabelImg: https://github.com/tzutalin/labelImg
With images given by the user, we can use the GUI in this library to import them, create rectangles around certain objects, and create “labels” which will be used by yolov5 to train a custom model.

YOLOv5: https://github.com/ultralytics/yolov5.git
Will be used to train our own custom model with the dataset that the user provides.


## How to use:

1.	Lets get started! First let’s clone the repository for this! Use the command:
git clone https://github.com/IshanDutta11/hddy5.git

2.	You should see two empty repositories, a dataset.yaml file, a README.md file, a test image, and a project.py file. Run the following commands. This will clone the necessary repositories and install all the required dependencies for this project.
pip install pyqt5 lxml --upgrade
git clone https://github.com/ultralytics/yolov5
cd yolov5
pip install -r requirements.txt
cd ..
 

3.	Next, we will collect some images for our dataset. Open the project.py file. 

4.	At the very bottom, there is a section indicating where the user must change some variables. Look at the Image Collection section. Specify the amount of images you would like and how long you want between each image capture.

5.	Uncomment only the image_capture function. Leave every other function commented out. 

6.	Before you run this file, Read this procedure:

a)	There is a tkinter gui that displays a 5 second countdown. Pose for the camera. .
b)	Once the image is taken, move to a different spot and pose again.
c)	Do this for the amount of images you want to take

•	Keep in mind, you want to give the model a wide variety of images.
•	Get up close, go far away, look up, look down. You can even face away from the camera if you want the back of your head to be part of the dataset.
•	Get some friends to help if you would like!
•	The more, the better.
 
7.	This will start the process immediately, so once you are ready, run the project file with:
python ./project.py



8.	Now your data/images folder should be populated with a bunch of images of you! Let’s label them now. Run these lines:
git clone https://github.com/tzutalin/labelImg
cd labelImg
pyrcc5 -o libs/resources.py resources.qrc
python labelImg.py

** Make sure the 3rd line of code is compatible with your setup. Reference the repository https://github.com/HumanSignal/labelImg to make sure.


9.	Once the gui is open click Open Dir and use the data/images folder. 

10.	Then click Change Save Dir and use the data/labels folder. 

11.	Finally, on the 4th box to the bottom, switch the toggle to “YOLO”. Now we are ready to label images!

12.	The labelling process:

a.	Hit “W” on your keyboard.
b.	Create a rectangle around your head
c.	Call the label “head”. 
d.	Ctrl+s to save
e.	Hit “D” to move to the next image.
f.	Repeat a-e until all images are labelled properly with head. When you are all done, X out of the gui

13.	You should see our data/labels folder populated with txt files that indicate where the heads are in the image provided. Now copy/move dataset.yaml into the yolov5 folder.

14.	Now we can start training, navigate into the yolov5 folder and train our model with this command:
python train.py --img 320 --batch 16 --epochs 500 --data dataset.yaml --weights yolov5s.pt --workers 2

You may need to edit these values to optimize performance and the model. 


15.	After the training process, we now have a model saved as an experiment in runs/train! BUT we still need to calibrate the focal length. Head back over to project.py, comment out image_capture and uncomment facial_detection. 
16.	There is an image called test.jpg in the home directory. Take a clear picture of your face a certain distance from your camera. Make sure to write this distance down.

17.	Measure the width of your head in inches and write it down.

18.	Place the image in the home directory. You may replace the original image if you like.

19.	Open project.py and navigate to the bottom. Change the focal calibration variables. If you are using a new name for your image, replace that path as well. 

20.	Edit the path_to_model variable so that it is pointing to either best.pt or last.pt in your most recent run.

21.	Once you have saved the document, navigate to the home directory and run this command again:
python ./project.py


If you have completed all the steps correctly and have a model with good data, you will be able to detect the distance of your head from your camera! 
If you want to exit, click on the feed and press q or ctrl + c in the terminal.
