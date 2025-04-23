# app.py
import os
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from tensorflow.keras import layers
from tensorflow.keras.models import load_model


# Custom Patches Layer
# Custom Patches Layer
class Patches(layers.Layer):
    def __init__(self, patch_size, **kwargs):
        super(Patches, self).__init__(**kwargs)  # Pass **kwargs to the parent class
        self.patch_size = patch_size

    def call(self, images):
        import tensorflow as tf  # Import inside the method
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding='SAME'
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches


# # Custom PatchEncoder Layer
# Custom PatchEncoder Layer
class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim, **kwargs):
        super(PatchEncoder, self).__init__(**kwargs)  # Pass **kwargs to the parent class
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patches):
        import tensorflow as tf  # Import inside the method
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patches) + self.position_embedding(positions)
        return encoded

# Load the model with custom objects
model = load_model(
    'my_hybrid_model.h5',
    custom_objects={'Patches': Patches, 'PatchEncoder': PatchEncoder}
)

def make_prediction(image_path):
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import load_img, img_to_array
    import numpy as np

    # Load and preprocess the image
    IMG_SIZE = 224  # Use the same image size as during training
    image = load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    image_array = img_to_array(image)
    image_array = image_array / 255.0  # Rescale as during training
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

    # Make prediction
    prediction = model.predict(image_array)

    # Since it's a binary classification, get the probability
    probability = prediction[0][0]

    # Interpret the result
    if probability >= 0.5:
        result = 'Melanoma Detected (Probability: {:.2f}%)'.format(probability * 100)
    else:
        result = 'No Melanoma Detected (Probability: {:.2f}%)'.format((1 - probability) * 100)

    return result


app = Flask(__name__)

# Configure upload folder and allowed extensions
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return (
        '.' in filename and
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    )


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload_and_predict():
    if request.method == 'POST':
        # Check if a file is present in the request
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file
        if file.filename == '':
            return redirect(request.url)
        # If the file is valid
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            # Save the file to the upload folder
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            # Make prediction
            prediction = make_prediction(filepath)
            # prediction = "Positive"
            # Render result template
            return render_template('result.html', prediction=prediction, image_url=filepath)
    return redirect(url_for('index'))

if __name__ == '__main__':
    # Ensure the upload folder exists
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)


