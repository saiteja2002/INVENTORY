import tensorflow
from tensorflow import keras
from keras_preprocessing import image

from PIL import Image
import numpy as np
#dictionary mapping softmax output and the corresponding feeling
feelings={
 0: 'angry',
1: 'fear',
 2: 'happy',
 3: 'neutral',
 4: 'sad',
5:'suprise'}
import cv2
#loading the trained models
new_model = tensorflow.keras.models.load_model('model location')

#gives the feeling when given the image


def predictor(x):
    classes = new_model.predict(x)

    y_pred = np.argmax(classes, axis=1)
    return feelings[y_pred[0]]
# Load the cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# To capture video from webcam.
cap = cv2.VideoCapture(0)

while True:
    # Read the frame
    _, img = cap.read()
    font = cv2.FONT_HERSHEY_COMPLEX
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #  Detect the faces

    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    gray = Image.fromarray(gray, 'L')

    # Draw the rectangle around each face
    for (x, y, w, h) in faces:
        try:
            gray=gray.crop((x-4,y-50,x+w+11, y + h+20))

        except:
            gray = Image.fromarray(gray1, 'L')
            gray = gray.crop((x-4,y-50,x+w+11, y + h+20))

        gray = gray.resize((48, 48))




        gray = image.img_to_array(gray)
        gray = np.expand_dims(gray, axis=0)
        # predicting using the model
        feeling = predictor(gray)
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(img, feeling, (x + 50, y - 10), font, 1, (200, 255, 255), 2, cv2.LINE_AA)
    # Display
    cv2.imshow('img', img)
    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
#feelings[feelings_array.tolist()[0].index

