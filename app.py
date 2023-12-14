from flask import Flask, request, jsonify
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# Load the model
model = load_model('veggiehealth_model_1.h5')
model._make_predict_function()  # Required for multi-threaded environments like Flask

# Function to preprocess the image
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# API endpoint to predict
@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        # Check if the POST request has a file part
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})

        file = request.files['file']

        # If the user does not select a file, the browser may send an empty file without filename
        if file.filename == '':
            return jsonify({'error': 'No selected file'})

        if file:
            # Save the uploaded image to a temporary file
            temp_path = 'temp_img.jpg'
            file.save(temp_path)

            # Preprocess the image
            img = preprocess_image(temp_path)

            # Make prediction
            prediction = model.predict(img)
            result = np.argmax(prediction)

            # Example: Mapping predicted index to class labels
            classes = ['class1', 'class2', 'class3']  # Define your class labels here
            predicted_class = classes[result]

            return jsonify({'prediction': predicted_class})

    elif request.method == 'GET':
        return 'Send a POST request to this endpoint with an image file.'

if __name__ == '__main__':
    app.run(debug=True)
