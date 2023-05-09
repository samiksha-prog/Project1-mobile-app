from flask import Flask,request,jsonify
import numpy as np
import pickle
from PIL import Image
from tensorflow.keras.utils import img_to_array, array_to_img
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
    img = Image.open(f)
    img_array = img_to_array(img)

    def get_features(img_path):
        img = array_to_img(img_path)
        resized_img = img.resize((224, 224))
        x = img_to_array(resized_img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        flatten = vgg_model.predict(x)
        return list(flatten[0])

    features_t = []
    features_t.append(get_features(img_array))

    predicted = model.predict(features_t)[0]
    return jsonify({'status':str(predicted)})

if __name__ == '__main__':
    app.run(debug=True)