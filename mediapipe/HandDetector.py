import cv2
import mediapipe as mp
class handDetector():
    def __init__(self, mode = False, maxHands = 1, detection_confidence=0.5, trackCon = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detection_confidence = detection_confidence
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode= self.mode,max_num_hands= self.maxHands,
                                        min_detection_confidence= self.detection_confidence,min_tracking_confidence= self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
    def findHands(self, image, draw=True):
        imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = self.hands.process(imgRGB)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # for id, lm in enumerate(hand_landmarks.landmark):
                #     print(id, lm)
                #     h, w, c = image.shape
                #     cx, cy = int(lm.x * w), int(lm.y + h)
                #     cv2.circle(image, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
                if draw:
                    self.mpDraw.draw_landmarks(image, hand_landmarks, self.mpHands.HAND_CONNECTIONS)
        return image