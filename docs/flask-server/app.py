import os
import cv2
import base64
import numpy as np
from flask import Flask, render_template, Response, request
from flask_socketio import SocketIO, emit
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Conv1D, MaxPooling1D
from keras.models import Sequential

# Create layer and load lip classification weights
model_image = Sequential()
model_image.add(Conv2D(8, (3, 3), input_shape=(426, 426, 3), activation='relu'))
model_image.add(MaxPooling2D(2, 2))
model_image.add(Conv2D(16, (3, 3), activation='relu'))
model_image.add(MaxPooling2D(2, 2))
model_image.add(Conv2D(32, (3, 3), activation='relu'))
model_image.add(MaxPooling2D(2, 2))
model_image.add(Flatten())
model_image.add(Dense(100, activation='relu'))
model_image.add(Dense(6, activation='softmax'))
model_image.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_image.summary()
model_image.load_weights('model_image.h5')

# Define the font and text properties
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.5
color = (0, 0, 255)  # BGR
thickness = 2

# Load the Haar cascade classifiers
classifier_face = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml")
classifier_smile = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_smile.xml")

# Define Flask app
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")  # Allow requests from all origins

@app.route('/', methods=['POST', 'GET'])
def index():
    return render_template('index.html')

@socketio.on('image') 
def handle_image(image): 
    # Process the received image on the server 
    processed_result = process_image(image)

    # Send the processed result back to the client
    emit('processed_result', processed_result, broadcast=True)

def process_image(image):
    # Placeholder function for image processing
    # Replace this with your actual image processing logic
    # For example, you might analyze the image and return some result

    # Decode base64 image data
    image_bytes = base64.b64decode(image.split(',')[1])
    image_np = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    x1 = 0
    y1 = 0

    while True:
        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect objects using the first classifier
        objects1 = classifier_face.detectMultiScale(gray_frame)

        get_face = image

        # Select the object of interest based on its position in the image
        selected_face = None
        for (x, y, w, h) in objects1:
            if selected_face is None or y < selected_face[1]:
                selected_face = (x, y, w, h)

        # Crop the selected object from the image
        if selected_face is not None:
            x, y, w, h = selected_face
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            get_face = image[y:y + h, x:x + w]

        # Detect objects using the second classifier
        gray_face = cv2.cvtColor(get_face, cv2.COLOR_BGR2GRAY)
        objects2 = classifier_smile.detectMultiScale(gray_face)

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
            cv2.rectangle(get_face, (cx - w // 2, cy - h),
                          (cx + w // 2, cy), (0, 255, 0), 2)
            object_image = get_face[cy - h:cy, cx - w // 2:cx + w // 2]

            # Calculate the position of the label
            text_size = cv2.getTextSize(
                'labels', font, font_scale, thickness)[0]
            x1 = cx - w // 2 + 10
            y1 = cy - h - 10 - text_size[1]

        smile = cv2.resize(object_image, (426, 426)) / 255

        y_pred = model_image.predict(np.asarray([smile]), verbose=0)
        y_pred = np.argmax(y_pred, axis=1)

        # Define a dictionary of labels
        labels = {0: 'none', 1: 'A', 2: 'I', 3: 'U', 4: 'E', 5: 'O'}

        # Give labels to the predicted values
        y_pred_labels = [labels[pred] for pred in y_pred]

        # Draw the label on the image
        cv2.putText(
            get_face, y_pred_labels[0], (x1, y1), font, font_scale, color, thickness)

        # Encode the processed image to base64
        _, encoded_image = cv2.imencode('.jpg', image)
        base64_encoded_image = base64.b64encode(encoded_image.tobytes()).decode('utf-8')

        return base64_encoded_image

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), allow_unsafe_werkzeug=True)
