from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np

model = cv2.ml.SVM_load("hog_svm.xml")

WIN_SIZE = (64, 128)
CELL_SIZE = (8, 8)
BLOCK_SIZE = (16, 16)
BLOCK_STRIDE = (8, 8)
HISTOGRAM_BINS = 9


def extract_hog_features(img):
    hog = cv2.HOGDescriptor(
        WIN_SIZE, BLOCK_SIZE, BLOCK_STRIDE, CELL_SIZE, HISTOGRAM_BINS
    )
    return hog.compute(img).flatten()


app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    img = cv2.resize(img, WIN_SIZE)
    features = extract_hog_features(img)

    # Predict
    _, pred = model.predict(np.array([features]))
    result = "Pedestrian detected" if pred[0][0] == 1 else "No pedestrian"

    return jsonify({"result": result})


@app.route('/')
def home():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)