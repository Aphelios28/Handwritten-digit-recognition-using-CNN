import cv2
import mediapipe as mp
import numpy as np
class handPaint:
    def __init__(self, width, height):
        self.canvas = np.zeros((height, width, 3), dtype=np.uint8)
        self.xi, self.yi = None, None

    def check_index_finger_up(self, hand_landmark):
        return (hand_landmark.landmark[8].y < hand_landmark.landmark[6].y and
                hand_landmark.landmark[12].y > hand_landmark.landmark[10].y and
                hand_landmark.landmark[16].y > hand_landmark.landmark[14].y and
                hand_landmark.landmark[20].y > hand_landmark.landmark[17].y)

    def check_hold_hand(self, hand_landmark):
        return (hand_landmark.landmark[8].y > hand_landmark.landmark[5].y and
                hand_landmark.landmark[12].y > hand_landmark.landmark[9].y and
                hand_landmark.landmark[16].y > hand_landmark.landmark[13].y and
                hand_landmark.landmark[20].y > hand_landmark.landmark[17].y)

    def check_status(self, hand_landmark):
        if self.check_index_finger_up(hand_landmark):
            return "finger_up"
        elif self.check_hold_hand(hand_landmark):
            return "hold_hand"
        else:
            return "nothing"

    def draw(self, hand_landmark, image):
        h, w, _ = image.shape
        cx, cy = int(hand_landmark.landmark[8].x * w), int(hand_landmark.landmark[8].y * h)

        if self.xi is None or self.yi is None:
            self.xi, self.yi = cx, cy

        cv2.line(self.canvas, (self.xi, self.yi), (cx, cy), (255, 255, 255), 5)
        self.xi, self.yi = cx, cy

    def clear(self):
        self.canvas.fill(0)
        self.xi, self.yi = None, None

    def action(self, image, hand_landmark):
        status = self.check_status(hand_landmark)

        if status == "hold_hand":
            self.clear()
        elif status == "finger_up":
            self.draw(hand_landmark, image)
        else:
            self.xi, self.yi = None, None

        return cv2.addWeighted(image, 1, self.canvas, 0.5, 0)
