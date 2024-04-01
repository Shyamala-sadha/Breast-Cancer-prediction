from flask import Flask, request, render_template
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from werkzeug.utils import secure_filename

# load model
model_path = 'save2.h5'
model = load_model(model_path)

app = Flask(__name__)

# Define the directory where uploaded files will be stored
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['UPLOAD'] = os.path.join('')
# Allow file uploads of up to 16 megabytes
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Ensure that file extensions are limited to images only
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

def predictions(img_path, model):
    img = load_img(img_path, target_size=(100, 100, 3))

    x = image.img_to_array(img)
    x = x/255
    x = np.expand_dims(x, axis=0)
    y = model.predict(x)
    pred = np.argmax(y[0], axis=-1)
    print(pred)

    if pred == 0:
        preds = 'Cancer Benign Stage'
    elif pred == 1:
        preds = 'Cancer Malignant Stage'
    else:
        preds = 'Cancer Normal Stage'

    return preds, y

@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the uploaded image file from the form
        uploaded_file = request.files['imagefile']
        
        # Save the file to a temporary directory
        filename = secure_filename(uploaded_file.filename)
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Create the 'uploads' directory if it doesn't exist
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])
        
        uploaded_file.save(img_path)
        
        img_upload = mpimg.imread(img_path)
        img_upload = np.expand_dims(img_upload, axis=2)
        rgb_image = np.repeat(img_upload, 3, axis=2)
        
        # Save the image to the 'static' directory
        img = os.path.join('static/savingimg', filename)
        mpimg.imsave(img, rgb_image)
        
        # Get the prediction for the uploaded image
        prediction, y = predictions(img_path, model)
        
        return render_template('index.html', prediction=prediction, img=img)

    return render_template('index.html')

if __name__ == '__main__':
    app.run()