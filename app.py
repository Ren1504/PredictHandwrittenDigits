from flask import Flask, render_template, request
#import pandas as pd
from scipy.misc import imsave, imread, imresize
import base64
from keras.models import load_model
import numpy as np
import cv2

init_Base64 = 21

global model

app = Flask(__name__, template_folder='templates')

@app.route('/')
def index_view():
    return render_template('home.html')

#def convertImage(imgData1):
#	imgstr = re.search(b'base64,(.*)',imgData1).group(1)
#	with open('output.png','wb') as output:
#	    output.write(base64.b64decode(imgstr))

#import our trained CNN from the h5 file
model = load_model("mnist.h5")

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        prediction = None

        img = request.form['url']
        #Removing the extra part of the url that was added for the encoding process
        img = img[init_Base64:]

        #Decoding for processing
        img_decoded = base64.b64decode(img)
        image = np.asarray(bytearray(img_decoded), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)

        #Resizing and reshaping the user's image.
        resized = cv2.resize(image, (28,28), interpolation = cv2.INTER_AREA)
        vect = np.asarray(resized, dtype="uint8")
        vect = vect.reshape(1, 1, 28, 28).astype('float32')
        #Predicting the digit
        y_pred = model.predict(vect)
        #To filter out the number the cnn has the most confidence in
        prediction = np.argmax(y_pred) + 1

    return render_template('results.html', prediction=prediction)
    
if __name__ == '__main__':
    app.run(debug=True, port=8000)