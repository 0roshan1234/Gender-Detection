import cv2
import numpy as np

# Load pre-trained face and gender detection models
face_model = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

gender_model = cv2.dnn.readNetFromCaffe('deploy_gender.prototxt', 'gender_net.caffemodel')

# List of gender labels
gender_list = ['Male', 'Female']


def detect_gender(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Detect faces in the image
    faces = face_model.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Extract face region
        face = image[y:y + h, x:x + w]
        # Preprocess face for gender detection
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
        # Feed the blob through the gender detection model
        gender_model.setInput(blob)
        gender_preds = gender_model.forward()
        # Get the predicted gender
        gender = gender_list[gender_preds[0].argmax()]
        # Draw rectangle around the face and label with gender
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(image, gender, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)

    return image


# Load input image
input_image = cv2.imread('shrreekrishna.jpg')
# Detect gender in the image
output_image = detect_gender(input_image)
# Display the result
cv2.imshow('Gender Detection', output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
