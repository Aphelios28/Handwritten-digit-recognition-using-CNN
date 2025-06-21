import argparse
import time
import cv2
from mediapipe.HandDetector import handDetector
from mediapipe.HandPaint import handPaint

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

    cap = cv2.VideoCapture(0)
    detector = handDetector()

    weight_cam, height_cam = 640, 480

    cap.set(3, weight_cam)
    cap.set(3, height_cam)

    cTime, pTime = 0, 0
    while True:
        success, image = cap.read()
        image = cv2.flip(image, 1)
        img = detector.findHands(image)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv2.imshow("Image", image)
        k = cv2.waitKey(1) & 0xff
        if k == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
if __name__ == '__main__':
    main()

