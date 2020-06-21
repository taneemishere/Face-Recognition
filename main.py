import face_recognition
import os
import cv2

# Knwon faces directory
KNWON_FACES_DIR = "known_faces"
# Unknwon faces directory
UNKNOWN_FACES_DIR = "unknown_faces"
TOLERANCE = 0.6
# For drawing square around the head, value in pixel
FRAME_THICKNESS = 3
# For names to be displayed, value in pix
FONT_THICKNESS = 2
# Our model is Convolutional Neural Network
# If only using the CPU without CUDA and GPU better to use the "hog"
MODEL = "hog"   #hog

print("loading known faces....")

# list for known faces, the encoded faces
known_faces = []
# list for known names, associated with that known faces
known_names = []

# Iterate over all the known faces or images and store the information
for name in os.listdir(KNWON_FACES_DIR):
    # Iterate over the all the images in the Directory
    for filename in os.listdir(f"{KNWON_FACES_DIR}/{name}"):
        # load the image file, actually we're loading the path
        image = face_recognition.load_image_file(f"{KNWON_FACES_DIR}/{name}/{filename}")
        # encode that image
        encoding = face_recognition.face_encodings(image)[0]
        # Add these to the list
        known_faces.append(encoding)
        known_names.append(name)


print("processing unknown faces....")
# Iterate over all the unknown images find all the faces and then for each face,
# compare that face to each of the known faces
for filename in os.listdir(UNKNOWN_FACES_DIR):
    print(filename)
    # load the image files
    image = face_recognition.load_image_file(f"{UNKNOWN_FACES_DIR}/{filename}")
    # finding all the faces inside that image, this is going to do face detection 
    # for us, finding all the locations for faces, acutally it'll find the the 
    # coordinates
    locations = face_recognition.face_locations(image, model=MODEL)
    # Now encode that faces, we're specifying the locations of images
    encodings = face_recognition.face_encodings(image, locations)
    # Convert this image to opencv, cvtColor means convert colors
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Iterate over the locations and encodings for the unknown faces that found
    for face_encoding, faces_location in zip(encodings, locations):
        # compare the known faces to the encoded faces of unknowns 
        # right now we've only one known faces but we can now extend this list
        # and this is the list of booleans whether the comparison is true or not
        results = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)
        match = None
        if True in results:
            match = known_faces[results.index[True]]
            print(f"Match Found: {match}")
            # for the rectangle we need top left and bottom right so that we can 
            # draw rectangle over the faces
            top_left = (faces_location[3], faces_location[0])
            bottom_right = (faces_location[1], faces_location[2])
            # green color rectangle
            color = [0, 255, 0]
            # draw the rectangle
            cv2.rectangle(image, top_left, bottom_right, color, FONT_THICKNESS)

            # now label the rectangle, this is for the text
            top_left = (faces_location[3], faces_location[2])
            bottom_right = (faces_location[1], faces_location[2]+22)
            cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)
            # putting the text
            # the image
            # match, what is the text whatever the match was
            # location for the text
            # specify the font
            # size is 0.5
            # the color, round about off-white
            # the font thinkness, we specify at the top
            cv2.putText(
                image, match, (faces_location[3]+10, faces_location[2]+15), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, (200, 200, 200), FONT_THICKNESS
                )
        
        # show the image, 
        # filename => title of image
        # image is the actual image we want to show
        cv2.imshow(filename, image)
        cv2.waitKey(0)
        cv2.destroyWindow(filename)