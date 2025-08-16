import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from mediapipe_hand_tracking.HandDetector import handDetector
from mediapipe_hand_tracking.HandPaint import handPaint

model = load_model('saved_model/digit_recog.model')

def preprocess_canvas(canvas):
    # Chuyển sang grayscale
    img = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    # Đảo màu (nền đen, nét trắng)
    img = cv2.bitwise_not(img)
    # Làm nét trắng tinh
    _, img = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
    # Làm dày nét
    img = cv2.dilate(img, np.ones((2, 2), np.uint8), iterations=1)
    # Resize về 28x28
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
    # Chuẩn hóa [0,1]
    img = img.astype('float32') / 255.0
    # Thêm chiều cho đúng input
    img = img.reshape(1, 28, 28, 1)
    return img

def main():
    width_cam, height_cam = 640, 640
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    painter = handPaint(width_cam, height_cam)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width_cam)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height_cam)

    while True:
        success, image = cap.read()
        if not success:
            break

        image = cv2.flip(image, 1)
        image = detector.findHands(image)
        results = detector.hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                painter.check_status(hand_landmarks)
                image = painter.action(image, hand_landmarks)

        predict_img = preprocess_canvas(painter.canvas)

        predict = model.predict(predict_img)
        label = int(np.argmax(predict))
        conf = float(np.max(predict)) * 100

        answer = f"this is {label} ({conf:.2f}%)"
        cv2.putText(image, answer, (10, 70), cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2)

        cv2.imshow("Image", image)
        cv2.imshow("Canvas", painter.canvas)  # Xem nét vẽ gốc

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
