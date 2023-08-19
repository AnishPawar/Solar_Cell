from email.mime import image
from re import L
from xml.sax.saxutils import prepare_input_source
from flask import Flask, request, render_template, jsonify, json
from flask_cors import CORS
import flask_ngrok
import tensorflow as tf
import cv2
import json

import os
import numpy as np

from preproc import make_preds, make_gradcam_heatmap

app = Flask(__name__)
# CORS(app)
# flask_ngrok.run_with_ngrok(app)

currentPred = ['', 0.0]


def emptydir():
    dirx = 'static/received_images'
    if os.listdir(dirx):
        for f in os.listdir(dirx):
            os.remove(os.path.join(dirx, f))


@app.route('/', methods=['GET', 'POST'])
def lander():

    if request.form:
        if request.form.getlist('foox'):
            indexVal = request.form.getlist('foox')
            print("IVAL")
            print(indexVal)
            if indexVal[0] == '1':
                print(f"It's {currentPred}")
                return render_template("index.html", path='0', prediction=f"It's {currentPred[0]}", probability=f"Probability: {currentPred[1]}%")

    return render_template("index.html", path='0', prediction=f"It's {currentPred[0]}", probability=f"Probability: {currentPred[1]*100}%")


@app.route('/upload_static_file', methods=['POST'])
def upload_static_file():
    global filePath, currentPred
    print("Got request in static files")
    print(request.files)

    emptydir()

    files = request.files.getlist('static_file')

    for f in files:
        f.save(f'static/received_images/'+f.filename)

    imageNames = os.listdir('static/received_images')
    preds = []
    for i in os.listdir('static/received_images'):
        img = cv2.imread(f'static/received_images/{i}')
        img = cv2.resize(img, (224, 224))
        modelPreds = make_preds(model, img)
        heatmap = make_gradcam_heatmap(img_array=np.expand_dims(
            img, axis=0), model=model, last_conv_layer_name='block5_pool', inner_model=model.get_layer("vgg19"))
        cv2.imwrite('static/processed_images/heatmap.png', heatmap)
        cv2.imwrite('static/processed_images/og.png', img)
        print(f"Model Preds Are: {modelPreds}")
        currentPred = modelPreds
        preds.append(modelPreds)

    retDict = {'imageName': imageNames, 'preds': preds}
    with open("static/outputData.json", "w") as outfile:
        json.dump(retDict, outfile)
    print(retDict)
    return render_template("index.html", path='1')


if __name__ == "__main__":

    emptydir()
    model = tf.keras.models.load_model('assets/FT_96.h5')
    app.run()
