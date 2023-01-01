import cv2
import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D

model = Sequential()
model.add(Conv2D(8, (3,3), input_shape=(100,100,3), activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(16,(3,3),activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Flatten())
model.add(Dense(100,activation='relu'))
model.add(Dense(6,activation='softmax'))
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.summary()

model.load_weights('b.h5')

# Define the font and text properties
font = cv2.FONT_HERSHEY_SIMPLEX 
font_scale = 0.5
color = (0, 0, 255)  # BGR
thickness = 2

# Load the Haar cascade classifiers
classifier_face = cv2.CascadeClassifier("./haar_cascade/lib/python3.10/site-packages/cv2/data/haarcascade_frontalface_default.xml")
classifier_smile = cv2.CascadeClassifier("./haar_cascade/lib/python3.10/site-packages/cv2/data/haarcascade_smile.xml")

# Initialize the webcam capture object
capture = cv2.VideoCapture(0)

# Loop until the user presses 'q' to quit
while True:
    # Read the next frame from the webcam
    ret, frame = capture.read()

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect objects using the first classifier
    objects1 = classifier_face.detectMultiScale(gray_frame)

    get_face = frame

    # Loop through the detected objects and draw a rectangle around them
    # for (x, y, w, h) in objects1:
    #     cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    #     get_face = frame[y:y+h, x:x+w]

    # Select the object of interest based on its position in the image
    selected_face = None
    for (x, y, w, h) in objects1:
        if selected_face is None or y < selected_face[1]:
            selected_face = (x, y, w, h)

    # Crop the selected object from the image
    if selected_face is not None:
        x, y, w, h = selected_face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        get_face = frame[y:y+h, x:x+w]

    # Detect objects using the second classifier
    gray_face = cv2.cvtColor(get_face, cv2.COLOR_BGR2GRAY)
    objects2 = classifier_smile.detectMultiScale(gray_face)

    # Loop through the detected objects and draw a rectangle around them
    # for (x, y, w, h) in objects2:
    #     if objects2 is None or y < objects2[1]:
    #         selected_object = (x, y, w, h)
    #         cv2.rectangle(get_face, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
        
    # Select the object of interest based on its lower middle position in the image
    selected_object = None
    for (x, y, w, h) in objects2:
        cx = x + w // 2
        cy = y + h
        if selected_object is None or cy > selected_object[1]:
            selected_object = (cx, cy, w, h)

    object_image = get_face



    # Draw a rectangle around the selected object
    if selected_object is not None:
        cx, cy, w, h = selected_object
        cv2.rectangle(get_face, (cx-w//2, cy-h), (cx+w//2, cy), (0, 255, 0), 2)
        object_image = get_face[cy-h:cy, cx-w//2:cx+w//2]

        # Calculate the position of the label
        text_size = cv2.getTextSize('labels', font, font_scale, thickness)[0]
        x1 = cx-w//2 + 10
        y1 = cy-h - 10 - text_size[1]
        #print(type(x1))
  
    smile = cv2.resize(object_image, (100,100))/255

    y_pred = model.predict(np.asarray([smile]))
    y_pred = np.argmax(y_pred, axis=1)
    print(y_pred)
    # Define a dictionary of labels
    labels = {0: 'none', 1: 'A', 2: 'I', 3: 'U', 4: 'E', 5: 'O'}

    # Give labels to the predicted values
    y_pred_labels = [labels[pred] for pred in y_pred]
    #y_pred_labels = 'pass'

    print(y_pred_labels[0])  # ['cat', 'dog', 'bird', ...]
    #print(type(y_pred_labels))

    # Draw the label on the image
    cv2.putText(get_face, y_pred_labels[0], (x1, y1), font, font_scale, color, thickness)

    # Show the frame with the detected objects
    cv2.imshow('Objects', frame)

    # Check if the user pressed 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam capture object
capture.release()
