import cv2
import librosa
import numpy as np
from flask import Flask, render_template, Response, request
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

# Create layer and load voice classification weight
model_voice = Sequential()
model_voice.add(Conv1D(32, 3, input_shape=(40, 128), activation='relu'))
model_voice.add(MaxPooling1D(2, 2))
model_voice.add(Conv1D(64, 3, activation='relu'))
model_voice.add(MaxPooling1D(2, 2))
model_voice.add(Conv1D(128, 3, activation='relu'))
model_voice.add(MaxPooling1D(2, 2))
model_voice.add(Flatten())
model_voice.add(Dense(100, activation='relu'))
model_voice.add(Dense(6, activation='softmax'))

model_voice.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
           metrics=['accuracy'])
model_voice.summary()

model_voice.load_weights('model_voice.h5')

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

app = Flask(__name__)


@app.route("/")
def index():
    return render_template('index.html')


@app.route('/webcam')
def webcam_feed():
    return Response(webcam(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


def webcam():
    cap = cv2.VideoCapture(0)
    x1 = 0
    y1 = 0

    while True:
        ret, frame = cap.read()

        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect objects using the first classifier
        objects1 = classifier_face.detectMultiScale(gray_frame)

        get_face = frame

        # Select the object of interest based on its position in the image
        selected_face = None
        for (x, y, w, h) in objects1:
            if selected_face is None or y < selected_face[1]:
                selected_face = (x, y, w, h)

        # Crop the selected object from the image
        if selected_face is not None:
            x, y, w, h = selected_face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            get_face = frame[y:y + h, x:x + w]

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
        # print(y_pred)

        # Define a dictionary of labels
        labels = {0: 'none', 1: 'A', 2: 'I', 3: 'U', 4: 'E', 5: 'O'}

        # Give labels to the predicted values
        y_pred_labels = [labels[pred] for pred in y_pred]
        # y_pred_labels = 'pass'

        print(y_pred_labels[0])
        # print(type(y_pred_labels))

        # Draw the label on the image
        cv2.putText(
            get_face, y_pred_labels[0], (x1, y1), font, font_scale, color, thickness)

        if ret:
            ret, jpeg = cv2.imencode('.jpg', frame)
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')


dic = {0: 'A', 1: 'I', 2: 'U', 3: 'E', 4: 'O'}


def extract_features_song(f):
    waveform, sr = librosa.load(f)

    # get mfcc
    mfcc = librosa.feature.mfcc(waveform, sr=22050, n_mfcc=40)
    print(mfcc.shape)

    return mfcc


def predict_label(input_audio):
    x_test = extract_features_song(input_audio)
    x_test = [librosa.util.fix_length(i, 128) for i in x_test]
    x_test = np.asarray(x_test)
    x_test = x_test.reshape(1, x_test.shape[0], x_test.shape[1])
    y_pred = model_voice.predict(x_test)
    y_pred = np.argmax(y_pred, axis=1)

    return y_pred


# routes
@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template("index.html")


@app.route("/submit", methods=['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        audio = request.files['upload']

        file_path = "static/" + audio.filename
        audio.save(file_path)
        print('pass_1')
        print(file_path)
        p = predict_label(file_path)
        print('pass_2')
        print(p)
        p = [dic[_] for _ in p]
    return render_template("index.html", prediction=p)


if __name__ == "__main__":
    app.run()
