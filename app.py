from flask import Flask, request, render_template, redirect, url_for
import numpy as np
from PIL import Image
from keras.models import load_model

app = Flask(__name__, template_folder='templates')
global model,prediction

# Loading the trained model from h5 file
model = load_model("mnist.h5")

# route for the landing page or home page
@app.route('/')
def upload():
    return render_template('home.html')

# route for handling the uploaded image and predicting
@app.route('/results', methods = ['POST'])
def upload_image():
    if request.method == 'POST':
        # fetching the uploaded image and preprocessing it
        img = Image.open(request.files['Img'].stream).convert("L")
        img = img.resize((28,28))
        img_arr = np.array(img).reshape(1,28,28,1)

        # predict the digit
        prediction = str(np.argmax(model.predict(img_arr)))

        # rendering the results page with the predicted digit
        return render_template('results.html',predicted = prediction)

# route for the "go home" button on the results page
@app.route('/home')
def home():
    return render_template('home.html')
    
if __name__ == '__main__':
    app.run(debug=True, port=8000)