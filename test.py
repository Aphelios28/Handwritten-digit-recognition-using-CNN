import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
import cv2
from train import x_test, y_test


model = tf.keras.models.load_model('digit_recog.model')

image_number = 1
while os.path.isfile(f"test_image/digit{image_number}.png"):
    try:
        img = cv2.imread(f"test_image/digit{image_number}.png")[:,:,0]
        img = np.invert(np.array([img]))
        pred = model.predict(img)
        print (f"the number is: {np.argmax(pred)}")
        plt.imshow(img[0], cmap = plt.cm.binary)
        plt.show()
    except:
        print ("Error")
    finally:
        image_number += 1