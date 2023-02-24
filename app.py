from flask import Flask, request, render_template, redirect, url_for
#import pandas as pd
#from scipy.misc import imread, imresize
import base64
from keras.models import load_model
import numpy as np
import cv2
from werkzeug.utils import secure_filename
import os
#init_Base64 = 21

global model
inp = None

app = Flask(__name__, template_folder='templates')

#@app.route('/')
#def index_view():
#    return render_template('home.html')

#def convertImage(imgData1):
#	imgstr = re.search(b'base64,(.*)',imgData1).group(1)
#	with open('output.png','wb') as output:
#	    output.write(base64.b64decode(imgstr))

#import our trained CNN from the h5 file
model = load_model("mnist.h5")

app.config['IMAGE_UPLOADS'] = './static/Images'

@app.route('/home',methods=['POST',"GET"])
def upload_image():
    if request == "POST":
        inp = request.files['file']
        if inp.filename == "":
            print("Invalid File name")
            return redirect(request.url)
        filename = secure_filename(inp.filename)

        basedir = os.path.abspath(os.path.dirname(__file__))
        inp.save(os.path.join(basedir, app.config["IMAGE_UPLOADS"],filename))
        return render_template("home.html",filename=filename)
    
    return render_template("home.html")

@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static',filename = "/Images" + filename), code = 888)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        prediction = None

        #img = request.form['url']
        #Removing the extra part of the url that was added for the encoding process
        #img = img[init_Base64:]

        #Decoding for processing
        #img_decoded = base64.b64decode(img)
        #image = np.asarray(bytearray(img_decoded), dtype="uint8")
        image = inp
        image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)

        #Resizing and reshaping the user's image.
        resized = cv2.resize(image, (28,28), interpolation = cv2.INTER_AREA)
        vect = np.asarray(resized, dtype="uint8")
        vect = vect.reshape(1, 1, 28, 28).astype('float32')
        #Predicting the digit
        y_pred = model.predict(vect)
        #To filter out the number the cnn has the most confidence in
        prediction = np.argmax(y_pred)

    return render_template('results.html', Prediction=prediction)
    
if __name__ == '__main__':
    app.run(debug=True, port=8000)