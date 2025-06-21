import cv2
import mediapipe as mp
import numpy as np


class handPaint:
    def __init__(self, width, height):
        self.canvas = np.zeros((width, height), dtype=np.uint8)
        self.xi, self.yi = None, None
    def check_index_finger_up(self, hand_landmark):
        finger_index_8 = hand_landmark.landmark[8]
        finger_index_6 = hand_landmark.landmark[6]
        finger_index_12 = hand_landmark.landmark[12]
        finger_index_10 = hand_landmark.landmark[10]
        finger_index_16 = hand_landmark.landmark[16]
        finger_index_14 = hand_landmark.landmark[14]
        finger_index_20 = hand_landmark.landmark[20]
        finger_index_17 = hand_landmark.landmark[17]
        return finger_index_8.y < finger_index_6.y and finger_index_12.y > finger_index_10.y and \
               finger_index_16.y > finger_index_14.y and finger_index_20.y > finger_index_17.y
    def check_hold_hand(self, hand_landmark):
        finger_index_8 = hand_landmark.landmark[8]
        finger_index_5 = hand_landmark.landmark[5]
        finger_index_12 = hand_landmark.landmark[12]
        finger_index_9 = hand_landmark.landmark[9]
        finger_index_16 = hand_landmark.landmark[16]
        finger_index_13 = hand_landmark.landmark[13]
        finger_index_20 = hand_landmark.landmark[20]
        finger_index_17 = hand_landmark.landmark[17]
        return finger_index_8.y > finger_index_5.y and finger_index_12.y > finger_index_9.y and \
            finger_index_16.y > finger_index_13.y and finger_index_20.y > finger_index_17.y