import cv2

# Load the Haar cascade classifiers
classifier_face = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
classifier_smile = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")

# Initialize the webcam capture object
capture = cv2.VideoCapture(0)

# Loop until the user presses 'q' to quit
while True:
    # Read the next frame from the webcam
    ret, frame = capture.read()
    get_face = frame

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect objects using the first classifier
    objects1 = classifier_face.detectMultiScale(gray_frame)

    # Loop through the detected objects and draw a rectangle around them
    for (x, y, w, h) in objects1:
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

    # Draw a rectangle around the selected object
    if selected_object is not None:
        cx, cy, w, h = selected_object
        cv2.rectangle(get_face, (cx - w // 2, cy - h), (cx + w // 2, cy), (0, 255, 0), 2)

    # Show the frame with the detected objects
    cv2.imshow('Objects', frame)

    # Check if the user pressed 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam capture object
capture.release()
