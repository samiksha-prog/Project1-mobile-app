from flask import Flask,request,jsonify
import numpy as np
import pickle
import cv2
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.models import model_from_json
from keras.applications.vgg16 import preprocess_input


model = pickle.load(open('model.pkl','rb'))
# load json and create model
json_file = open('vgg_model.json', 'r')
vgg_model_json = json_file.read()
json_file.close()
vgg_model = model_from_json(vgg_model_json)
# load weights into new model
vgg_model.load_weights("vgg_model.h5")

app = Flask(__name__)

@app.route('/')
def index():
    return "Hello world"

@app.route('/predict',methods=['POST'])
def predict():
    f = request.files['file']
    f.save('input_file.jpg')

    img = cv2.imread('input_file.jpg')
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray_img = cv2.cvtColor(hsv_img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Set minimum area threshold
    min_area = 50

    # Iterate through contours
    for contour in contours:
        # Calculate area
        area = cv2.contourArea(contour)

        # Remove contour if area is below threshold
        if area < min_area:
            cv2.drawContours(thresh, [contour], 0, 0, -1)

    # Apply mask to original image
    result = cv2.bitwise_and(img, img, mask=thresh)
    cv2.imwrite('./test.jpg', result)

    def get_features(img_path):
        img = load_img(img_path, target_size=(224, 224))
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        flatten = vgg_model.predict(x)
        return list(flatten[0])

    features_t = []
    features_t.append(get_features('test.jpg'))

    predicted = model.predict(features_t)[0]
    return jsonify({'status':str(predicted)})

if __name__ == '__main__':
    app.run(debug=True)