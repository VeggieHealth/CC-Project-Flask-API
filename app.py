from flask import Flask, request, jsonify
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# Load the model
model = load_model('veggiehealth_model_1.h5')

def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'No selected file'})

        if file:
            temp_path = 'temp_img.jpg'
            file.save(temp_path)

            img = preprocess_image(temp_path)

            prediction = model.predict(img)

            highest_probability = np.max(prediction)

            if highest_probability < 0.5:
                return jsonify({'prediction': 'Tidak terdeteksi'})

            result = np.argmax(prediction)
            classes = ['Bitter_Gourd', 'Brinjal', 'Broccoli', 'Cabbage', 'Carrot', 'Cassava_leaves', 'Cauliflower', 'Cucumber', 'Gnetum_gnemon', 'Potato', 'Tomato', 'bean_sprouts', 'string_bean']  # Update with your actual classes
            predicted_class = classes[result]

            return jsonify({'prediction': predicted_class})

    elif request.method == 'GET':
        return 'Send a POST request to this endpoint with an image file.'

if __name__ == '__main__':
    app.run(debug=True)
