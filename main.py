import cv2
import time
from djitellopy import Tello
import numpy as np
import mediapipe as mp
import tensorflow as tf
from keras.models import load_model
from queue import Queue



dataQueue = Queue(5)        
        
# Set frame height and width
frameWidth = 600
frameHeight = 400

# initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Load the gesture recognizer model
model = load_model('mp_hand_gesture')

# Load class names
file = open('gesture.names', 'r')
classNames = file.read().split('\n')
file.close()
print(classNames)


#Initialize the Drone
tello = Tello()
tello.connect()
tello.streamoff()
tello.streamon()

# Set battery threshold
BAT_THRESH = 20

print(tello.get_battery())

if tello.get_battery() > BAT_THRESH:
    tello.takeoff()
    tello.move_up(50)
while True:
    # Read each frame from the webcam
    frame = tello.get_frame_read().frame
    frame = cv2.resize(frame, (frameWidth, frameHeight))

    x = frameWidth
    y = frameHeight

#     # Flip the frame vertically
    frame = cv2.flip(frame, 1)
    #
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get hand landmark prediction
    result = hands.process(framergb)

    # print(result)
    
    className = ''

    # post process the result
    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                #print(id, lm)
                lmx = int(lm.x * x)
                lmy = int(lm.y * y)

                landmarks.append([lmx, lmy])

            # Drawing landmarks on frames
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

            # Predict gesture
            prediction = model.predict([landmarks])
            # print(prediction)
            classID = np.argmax(prediction)
            className = classNames[classID]

    # show the prediction on the frame
    cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0,0,255), 2, cv2.LINE_AA)

    dataQueue.put(className)
    
    while(dataQueue.full()):
        className = dataQueue.get()
        if(className == "fist"):
            print("This is a fist")
            tello.land()
        elif(className == "peace"):
            print("This is a peace sign")
            tello.flip_back()
        elif(className == "okay"):
            print("This is a okay sign")
            tello.flip_forward()
        elif(className == "thumbs down"):
            print("This is a thumbs down")
            tello.move_down(20)
        elif(className == "thumbs up"):
            print("This is a thumbs up sign")
            if(tello.get_height() > 20):
                tello.move_up(20)
            else:
                tello.takeoff()
        elif(className == "stop"):
            print("This is a stop sign")
            tello.move_back(20)
        elif(className == "rock"):
            print("This is a rock sign")
            tello.flip_left()
            tello.flip_right()
        elif(className == "live long"):
            print("This is a okay sign")
            tello.move_forward(20)
        elif(className == "smile"):
            print("This is a smile sign")
    
        # Show the final output
        cv2.imshow("Output", frame) 
        
        if cv2.waitKey(1) == ord('q'):
            break

tello.land()
tello.streamoff()
tello.end()
cv2.destroyAllWindows()