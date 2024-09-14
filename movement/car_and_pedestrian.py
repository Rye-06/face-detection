# Movement Detection Open CV project

# Imports open cv library
import cv2

# Load pre-trained car data from open cv
#trained_car_data = cv2.CascadeClassifier(r'C:\Users\shaur\OneDrive\Documents\opencvDetection\movement\car.xml')

# Car video to detect cars
#car_video = cv2.VideoCapture(r'C:\Users\shaur\OneDrive\Documents\opencvDetection\movement\dashcam.mp4')

# Load pre-trained pedestrian data from open cv
trained_pedestrian_data = cv2.CascadeClassifier(r'C:\Users\shaur\OneDrive\Documents\opencvDetection\movement\pedestrian.xml')

# Pedestrian video to detect cars
pedestrian_video = cv2.VideoCapture(r'C:\Users\shaur\OneDrive\Documents\opencvDetection\movement\pedestrian.mp4')

# Captures video from webcam
#webcam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Iterate forever over frames till the video is over
while True:
    
    # Reads the current frame
    successful_frame_read,frame = pedestrian_video.read()

    # If there is a frame remaining in the video
    if successful_frame_read:
    # Converts frame gray
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break

    pedestrian_coordinates = trained_pedestrian_data.detectMultiScale(gray)

    for (x,y,w,h) in pedestrian_coordinates:
    # Draws Rectangle around the face(s)
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0),3)

    # Displays the current frame
    cv2.imshow('Pedestrian Detector',frame)

    # Shows next frame every 1ms and don't autoclose of the program
    key = cv2.waitKey(1)

    # Ends progrma is 'Q' or 'q' key is pressed
    if key == ord("q") or key == ord("Q"):
        break


# Ends window
cv2.destroyAllWindows()
pedestrian_video.release()
#webcam.release()


# Finish Code
print("CODE FINISHED")