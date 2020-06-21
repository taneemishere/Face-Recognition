# Face Recognition
Python app that first detect face of a person from a folder called known_faces, where there is another folder which is person, in my case I only add one folder aka person, you can add multiple, and then recognize his/her face from other images.

## Requirements:
- [Python](https://www.python.org/downloads/)

- [OpenCV](https://www.opencv.org/)
<br />```pip install python-opencv```

- [Face Recognition](https://pypi.org/project/face-recognition/)
<br />```pip install face-recognition```

<br />Also if the face-recognition installation shows any error, most probably you'll need to install [Visual Studio](https://visualstudio.microsoft.com/downloads/), and then within Visual Studio you have to install Destop Development with C++.

## Directories
There are two directories in this, the "knows_faces" and "unknown_faces", the known_faces contain the person folder which you want to recognize by the program.
The unknown_faces contais the images from which you want to recognize the face of known person.

## Program Workflow
- First import some of the packages you need.
- Specify the directories for known and unknown faces/images.
- Give the Tolerance rate of 0.6 which is somekind of default value.
- Frame and Font Thickness values in px.
- Specify the model, make sure to check both Convolutional Neural Network (cnn), and Histogram of Oriented Gradients (hog). 
- Iterate over the known_faces directory and load all the images, encode them and then add them to the list.
- Then iterate over the unknown_faces directory for the images and compare each of these for each of the known faces.
- Load the image files and find it's location i-e the coordinates of them.
- Encode these images now.
- Convert it by using cv2.
- Now iterate over the locations and encodings for the unknown faces that found in unknown.
- Compare the known faces to the encoded unknown faces.
- Draw the rectangle for the recognizer, green in color.
- Put the label text over the recognizer rectangle with the name of the known face(s)

## Images
All these images are downloaded from web.

## Run
In the main directory run the following command.
<br />```python main.py```
