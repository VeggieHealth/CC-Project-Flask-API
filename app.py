from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import io
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

model = load_model('veggiehealth_model_5.h5') 


def preprocess_image(img_data):
    img = image.load_img(io.BytesIO(img_data), target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    global model  # Use the loaded model
    if model is None:
        return jsonify({'status': False, 'error': 'Model not loaded'}), 500

    if 'file' not in request.files:
        return jsonify({'status': False, 'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'status': False, 'error': 'No selected file'}), 400

    img_data = file.read()
    img = preprocess_image(img_data)

    prediction = model.predict(img)
    highest_probability = np.max(prediction)

    if highest_probability < 0.5:
        return jsonify({'prediction': 'Tidak terdeteksi', 'status': True}), 200

    result = np.argmax(prediction)
    classes = ['Bitter_Gourd', 'Brinjal', 'Broccoli', 'Cabbage', 'Carrot', 'Cassava_leaves', 'Cauliflower', 'Cucumber', 'Enoki_mushrooms', 'Gnetum_gnemon', 'Potato', 'Tomato', 'bean_sprouts', 'sawi_hijau', 'sawi_putih', 'spinach', 'string_bean', 'water_spinach']  # Update with your actual classes
    predicted_class = classes[result]
    return jsonify({'prediction': predicted_class, 'akurasi': float(highest_probability), 'status': True}), 200


@app.route('/classes', methods=['GET'])
def get_classes():
    classes = ['Bitter_Gourd', 'Brinjal', 'Broccoli', 'Cabbage', 'Carrot', 'Cassava_leaves', 'Cauliflower', 'Cucumber', 'Enoki_mushrooms', 'Gnetum_gnemon', 'Potato', 'Tomato', 'bean_sprouts', 'sawi_hijau', 'sawi_putih', 'spinach', 'string_bean', 'water_spinach']  # Update with your actual classes
    return jsonify({'classes': classes, 'status': True}), 200


if __name__ == '__main__':
    app.run(debug=True,port=8080)
