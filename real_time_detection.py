import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# 1. Load the trained model and Haar Cascade
model = load_model('model/emotion_model.h5')
face_classifier = cv2.CascadeClassifier('model/haarcascade_frontalface_default.xml')

# Labels assigned alphabetically by Keras based on folder names
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# 2. Start Video Capture (0 is usually your built-in webcam)
cap = cv2.VideoCapture(0)

print("Starting webcam... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    # Convert frame to grayscale for face detection and the model
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_classifier.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)
    
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        
        # Crop the face region
        roi_gray = gray_frame[y:y+h, x:x+w]
        
        # Resize to 48x48 to match the model's input
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
        
        # Normalize the image
        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            
            # Predict the emotion
            prediction = model.predict(roi, verbose=0)[0]
            label = emotion_labels[prediction.argmax()]
            
            # Put text above the rectangle
            label_position = (x, y - 10)
            cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, 'No Faces', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
    # Show the video feed
    cv2.imshow('Emotion Detector', frame)
    
    # Press 'q' to close the window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()