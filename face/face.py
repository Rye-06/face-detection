# Face Detection Open CV project

# Imports open cv library
import cv2

# Load pre-trained face data from open cv
trained_face_data = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Image to detect face
#img = cv2.imread('2Diana.jpg')

# Captures video from webcam
webcam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Iterate forever over frames
while True:
    # Reads current frame
    successful_frame_read,frame = webcam.read()

    # Converts to Grey Scale
    gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detects Faces
    face_coordinates = trained_face_data.detectMultiScale(gray_scale)

    for (x,y,w,h) in face_coordinates:
    # Draws Rectangle around the face(s)
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0),3)

    # Shows Video on Window
    cv2.imshow('Face Detector', frame)
    
    # Shows next frame and don't autoclose of the program
    key = cv2.waitKey(1)

    # Stops code if 'S' key is pressed
    if key == 83 or key == 115:
        break

# Release the webcam once code is executed
webcam.release()

# FINISH CODE
print("Code Completed!")



# Code for Open CV Image Face Detection
 
"""

# Converts Image to Grey Scale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detects Faces
face_coordinates = trained_face_data.detectMultiScale(gray_img)

for (x,y,w,h) in face_coordinates:
    # Draws Rectangle around the face(s)
    cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0),3)

#print(face_coordinates)

# Shows Image on Window
cv2.imshow('Face Detector', img)


# Waits till key is pressed to end program
cv2.waitKey()


# FINISH CODE
print("Code Completed!")

"""
