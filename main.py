import cv2
from djitellopy import Tello 
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model

# Set frame height and width
frameWidth = 960
frameHeight = 720

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
tello.takeoff()
tello.streamoff()
tello.streamon()

# Set battery threshold
BAT_THRESH = 20

print(tello.get_battery())

while tello.get_battery() > BAT_THRESH:
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
    if(className == "fist"):
        print("This is a fist")
        tello.land()
    elif(className == "peace"):
        print("This is a peace sign")
    elif(className == "okay"):
        print("This is a okay sign")
    elif(className == "thumbs up"):
        print("This is a thumbs up")
    elif(className == "thumbs down"):
        print("This is a thumbs down")
    elif(className == "call me"):
        print("This is a call me sign")
    elif(className == "stop"):
        print("This is a stop sign")
    elif(className == "rock"):
        print("This is a rock sign")
    elif(className == "live long"):
        print("This is a okay sign")
    elif(className == "smile"):
        print("This is a smile sign")
   
    # Show the final output
    cv2.imshow("Output", frame) 
    
    if cv2.waitKey(1) == ord('q'):
        tello.land()
        tello.streamoff()
        cv2.destroyAllWindows()
        break

tello.land()
tello.streamoff()
cv2.destroyAllWindows()