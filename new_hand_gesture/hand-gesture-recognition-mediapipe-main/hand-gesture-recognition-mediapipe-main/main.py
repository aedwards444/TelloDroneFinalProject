#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import copy
import argparse
import itertools
from collections import deque

import cv2 as cv
import mediapipe as mp
from djitellopy import Tello

from model import KeyPointClassifier
from model import PointHistoryClassifier


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=600)
    parser.add_argument("--height", help='cap height', type=int, default=400)

    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)

    args = parser.parse_args()

    return args


def main():
    # Argument parsing #################################################################
    args = get_args()

    cap_width = args.width
    cap_height = args.height

    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    # Camera preparation and Drone activation ###############################################################

    tello = Tello()
    tello.connect()
    tello.streamon()
    print(tello.get_battery())
    

    # Model load #############################################################
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=1,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    keypoint_classifier = KeyPointClassifier()

    point_history_classifier = PointHistoryClassifier()

    # Read labels ###########################################################
    with open('model/keypoint_classifier/keypoint_classifier_label.csv',
              encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ]
    with open(
            'model/point_history_classifier/point_history_classifier_label.csv',
            encoding='utf-8-sig') as f:
        point_history_classifier_labels = csv.reader(f)
        point_history_classifier_labels = [
            row[0] for row in point_history_classifier_labels
        ]


    # Coordinate history #################################################################
    history_length = 16
    point_history = deque(maxlen=history_length)

    # Finger gesture history ################################################
    finger_gesture_history = deque(maxlen=history_length)

    
    #################Take flight ###################
    tello.takeoff()
    tello.move_up(70)



    #################Begin Hand Recognition ##########################
    
    
    
    while True:

        # Process Key (ESC: end) #################################################
        key = cv.waitKey(10)
        if key == 27:  # ESC
            break

        # Camera capture #####################################################

        image = tello.get_frame_read().frame
        image = cv.resize(image, (cap_width, cap_height))
        image = cv.flip(image, 1)  # Mirror display
        debug_image = copy.deepcopy(image)

        # Detection implementation #############################################################
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        #  ####################################################################
        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):
               
                # Landmark calculation
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                # Conversion to relative coordinates / normalized coordinates
                pre_processed_landmark_list = pre_process_landmark(
                    landmark_list)
                pre_processed_point_history_list = pre_process_point_history(
                    debug_image, point_history)
    

                # Hand sign classification
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)        
                    
            
                if hand_sign_id == 0: # Palm                       
                    tello.move_back(40)
                elif hand_sign_id == 4: # Peace
                    tello.land()
                elif hand_sign_id == 5: # BackHand
                    tello.move_forward(40)
                    
                # Flips **********causing issues ***************
                # elif hand_sign_id == 3: # Right OK
                #     tello.flip_left()
                # elif hand_sign_id == 6: # Left OK
                #     tello.flip_right()
                    
                    
                ######### Finger point tracking###########################################
                

                elif hand_sign_id == 2:  # Point gesture
                
                    # Append (x, y) location of point finger to a deque
                    point_history.append(landmark_list[8]) 
                    
                    while hand_sign_id == 2 and len(point_history) > 1:
                        
                        first_pop = point_history.popleft()
                        second_pop = point_history.popleft()
                        
                        # Ensure the movement range will be within boundaries
                        if 50 > (first_pop[0] - second_pop[0]) > 20: 
                            print('moving right:', first_pop[0] - second_pop[0])
                            tello.move_right(first_pop[0] - second_pop[0])
                            
                        elif 50 > (second_pop[0] - first_pop[0]) > 20:
                            print('moving left:', second_pop[0] - first_pop[0])
                            tello.move_left(second_pop[0] - first_pop[0])
                            
                            
                        elif 50 > (first_pop[1] - second_pop[1]) > 20:
                            print('moving up:', first_pop[1] - second_pop[1])
                            tello.move_up(first_pop[1] - second_pop[1])
                            
                        elif 50 > (second_pop[1] - first_pop[1]) > 20:
                            print('moving down:', second_pop[1] - first_pop[1])
                            tello.move_down(second_pop[1] - first_pop[1])
                        else:
                            point_history.clear()
                    
                ########## End finger point tracking ##########################################
                
                
                
           
                finger_gesture_id = 0
                point_history_len = len(pre_processed_point_history_list)
                if point_history_len == (history_length * 2):
                    finger_gesture_id = point_history_classifier(
                        pre_processed_point_history_list)

                # Calculates the gesture IDs in the latest detection
                finger_gesture_history.append(finger_gesture_id)
                

        # Screen reflection #############################################################
        cv.imshow('Hand Gesture Recognition', debug_image)

    tello.streamoff()
    tello.end()
    cv.destroyAllWindows()





############ End hand recognition ####################



def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


def pre_process_point_history(image, point_history):
    image_width, image_height = image.shape[1], image.shape[0]

    temp_point_history = copy.deepcopy(point_history)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, point in enumerate(temp_point_history):
        if index == 0:
            base_x, base_y = point[0], point[1]

        temp_point_history[index][0] = (temp_point_history[index][0] -
                                        base_x) / image_width
        temp_point_history[index][1] = (temp_point_history[index][1] -
                                        base_y) / image_height

    # Convert to a one-dimensional list
    temp_point_history = list(
        itertools.chain.from_iterable(temp_point_history))

    return temp_point_history




if __name__ == '__main__':
    main()
