import argparse
import time
import cv2
from mediapipe_hand_tracking.HandDetector import handDetector
from mediapipe_hand_tracking.HandPaint import handPaint

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=480)
    parser.add_argument("--height", help='cap height', type=int, default=540)

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
    arg = get_args()
    weight_cam, height_cam = 640, 480
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    painter = handPaint(weight_cam, height_cam)

    cap.set(3, weight_cam)
    cap.set(3, height_cam)

    while True:
        success, image = cap.read()
        image = cv2.flip(image, 1)
        img = detector.findHands(image)
        resutls = detector.hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        if resutls.multi_hand_landmarks:
            status = f"You are "
            for hand_landmarks in resutls.multi_hand_landmarks:
                if painter.check_hold_hand(hand_landmarks):
                    status += "hold your hand"
                elif painter.check_index_finger_up(hand_landmarks):
                    status += "pointing with your index finger"
                else:
                    status += "doing nothing"
            cv2.putText(img, status, (10, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
        cv2.imshow("Image", image)
        k = cv2.waitKey(1) & 0xff
        if k == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
if __name__ == '__main__':
    main()

