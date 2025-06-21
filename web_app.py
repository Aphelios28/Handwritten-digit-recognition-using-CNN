from flask import Flask, render_template, request, redirect, url_for
import os
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = load_model('digit_recog.model')

IMAGE_SIZE = (28, 28)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods = ['POST'])
def predict():
    if 'image' not in request.files:
        return redirect(url_for('index'))

    file = request.files['image']
    if file.filename == "":
        return redirect(url_for('index'))

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    # Xử lý ảnh
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (28, 28))
    img = 255 - img
    img = img / 255.0
    img = img.reshape(1, 28, 28)

    prediction = model.predict(img)
    result = int(np.argmax(prediction))
    confidence_score = float(np.max(prediction))*100

    return render_template('index.html', result=result, confidence = round(confidence_score, 2), img_path=file_path)

if __name__ == '__main__':
    app.run(debug=True)

