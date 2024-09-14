# Helps the Blind to know the objects in front of them

# Info: https://www.youtube.com/watch?v=lE9eZ-FGwoE

# Imports Libraries
import cv2
import numpy as np

# Image Path
image_path = 'test_img.JPG'

# Prototxt Path
prototxt = 'deploy.prototxt'

# Model Path 
model = 'mobilenet_iter_73000.caffemodel'

# Classes Dataset
classes = 'coco.names'

# Dont classify image if it is below or equal to this confidence level 
min_confidence = 0.2

# Random generated color for each different object
np.random.seed(543210)
colors = np.random.uniform(0,255,size=(len(classes),3))

# Loads the neural network 
net = cv2.dnn.readNetFromCaffe(prototxt, model)

# Read Image
image = cv2.imread(image_path)
height, width = image.shape[0], image.shape[1]
blob = cv2.dnn.blobFromImage(cv2.resize(image,(300,300)), 0.007, (300,300), 130)

# Displays the output
net.setInput(blob)
detected_obj = net.forward()

print(detected_obj[0],[0],[0])

for i in range(detected_obj.shape[2]):
    # Confidence of the computer
    confidence = int(detected_obj[0][0][i][2])

    # Execute only if the confidence of the computer is higher than the minimum confidence on detecting the object(s)
    if confidence > min_confidence:
        # Class index number
        class_ind = detected_obj[0][0][i[2]]
        
        # Coordinates of Object Detected
        upper_left_x = int(detected_obj[0,0,i,3] * width)
        upper_left_y = int(detected_obj[0,0,i,4] * height)
        lower_right_x = int(detected_obj[0,0,i,5] * width)
        lower_right_y = int(detected_obj[0,0,i,6] * height)

        obj_name = f"{classes[class_ind]} : {confidence:.2f}%"
        cv2.rectangle(image, (upper_left_x,upper_left_y),(lower_right_x,lower_right_y), colors[class_ind],3)
        cv2.putText(image,obj_name,(upper_left_x,
        upper_left_y-15 if upper_left_y > 30 else upper_left_y + 15),
        cv2.FONT_HERSHEY_SIMPLEX,0.6,colors[class_ind],2)

# Shows image on screen
cv2.imshow("Object Detector", image)
cv2.waitKey(0)
cv2.destroyAllWindows