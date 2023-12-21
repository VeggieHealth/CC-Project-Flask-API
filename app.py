from flask import Flask, request, jsonify
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import io
from flask_cors import CORS
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

model = load_model('veggiehealth_model_tf_origin.h5')

def preprocess_image(img_data):
    img = image.load_img(io.BytesIO(img_data), target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array/255.
    return img_array


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'status': False, 'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'status': False, 'error': 'No selected file'}), 400

    if file:
        filename = secure_filename(file.filename)

        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            return jsonify({'status': False, 'error': 'Invalid file type. Allowed types: JPG, JPEG, PNG'}), 400

        img_data = file.read()
        img = preprocess_image(img_data)

        if img is None:
            return jsonify({'status': False, 'error': 'Error processing image'}), 500

        prediction = model.predict(img)
        highest_probability = np.max(prediction)
        result = np.argmax(prediction)

        classes = [
            {'id': 1, 'name': 'Bean'},
            {'id': 2, 'name': 'Bitter_Groud'},
            {'id': 3, 'name': 'Bottle_Gourd'},
            {'id': 4, 'name': 'Brinjal'},
            {'id': 5, 'name': 'Broccoli'},
            {'id': 6, 'name': 'Cabbage'},
            {'id': 7, 'name': 'Capiscium'},
            {'id': 8, 'name': 'Carrot'},
            {'id': 9, 'name': 'Cauliflower'},
            {'id': 10, 'name': 'Cucumber'},
            {'id': 11, 'name': 'Papaya'},
            {'id': 12, 'name': 'Potato'},
            {'id': 13, 'name': 'Pumpkin'},
            {'id': 14, 'name': 'Radish'},
            {'id': 15, 'name': 'Tomato'},
        ]
        accuracy = float(highest_probability * 100)

        predicted_class_info = next((item for item in classes if item['id'] == result + 1), None)
        if accuracy > 0:
            predicted_class = predicted_class_info['name']
            predicted_id = predicted_class_info['id']
            return jsonify({'prediction': predicted_class, 'vegetableId': predicted_id, 'accuracy': accuracy, 'status': True}), 200
        else:
            return jsonify({'status': False, 'error': 'Vegetable not found'}), 500
    else:
        return jsonify({'status': False, 'error': 'File not found'}), 400

if __name__ == '__main__':
    app.run(debug=True, port=8080)
