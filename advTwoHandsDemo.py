import cv2
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)
detector = HandDetector(detectionCon = 0.8, maxHands = 2)

while True:
    success, img = cap.read()
    
    hands, img = detector.findHands(img, flipType=True) #This draws the hand models to the screen when it detects the hands
    cv2.imshow("Image", img)
    cv2.waitKey(1)
    if cv2.waitKey(1) == ord('q'):
        break

# release the webcam and destroy all active windows
cap.release()

cv2.destroyAllWindows()

