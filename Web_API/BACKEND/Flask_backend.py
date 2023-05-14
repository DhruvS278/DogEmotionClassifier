from flask import Flask, request, jsonify
from PIL import Image
import io
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from flask import render_template_string
from keras.applications.mobilenet import MobileNet, preprocess_input, decode_predictions
from flask import send_from_directory
import os

app = Flask(__name__)
model = load_model('C://Users//DHRUV.DESKTOP-KRBVT38//OneDrive//Desktop//dogemotion//DogEmotionClassifier//EmotionClassfierModel//dog_emotion_classifier1.h5')
class_labels = ['Angry', 'Sad', 'Relaxed', 'Happy']
@app.route('/')
def home():
    return render_template_string('''
        <html>
            <body>
                <h1>Dog Emotion Classification API</h1>
                <form method="POST" action="/predict" enctype="multipart/form-data">
                    <input type="file" name="image">
                    <input type="submit" value="Predict">
                </form>
            </body>
        </html>
    ''')

# Define API endpoint for image prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Check if file is uploaded
    if 'image' not in request.files:
        return 'No image file found', 400

    # Load image file from request
    image = request.files['image'].read()
    image = Image.open(io.BytesIO(image))

    # Preprocess image for prediction
    image = image.resize((96, 96))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image.astype('float32')
    image /= 255.0

    # Make prediction using model
    predictions = model.predict(image)
    predicted_classes = np.argmax(predictions, axis=1)

    # Format prediction result
    result = [{'class': class_labels[predicted_class], 'probability': float(predictions[0][predicted_class])}
              for predicted_class in predicted_classes]

    return jsonify({'predictions': result})

if __name__ == '__main__':
    app.run(debug=True)