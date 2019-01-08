from flask import Flask
from flask import request
from flask import redirect
from flask import session
from keras.models import load_model
import os
import test_network
import train_network

app = Flask(__name__)

@app.route("/test")
def hello():
    return train_network.dataTraining()

@app.route("/classify-image", methods=['GET','POST'])
def classifyImage():
    if request.method=='POST':
        imagefile = request.files.get('image', '')
        imagefile.save(os.path.join("uploads", imagefile.filename))
        image_path = "uploads/"+imagefile.filename
        return test_network.imageClassification(image_path)
        # return image_path

if __name__ == '__main__':
    app.debug = True
    app.run()
