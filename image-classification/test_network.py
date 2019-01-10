from keras.preprocessing.image import img_to_array
from keras.models import load_model
from flask import jsonify
import keras
import numpy as np
import argparse
import imutils
import cv2
import tensorflow as tf
from flask import session
import numpy as np
import os
from random import randint

def random_with_N_digits(n):
    range_start = 10**(n-1)
    range_end = (10**n)-1
    return randint(range_start, range_end)

def imageClassification(image):

	model = "resnetfood.model"

	burger=0
	pizza=0
	sushi=0
	friedchicken=0
	pasta=0
	sate=0

	# switch function to convert value to label
	def label_switch(x):
		return {
	        burger: "Burger",
	        pizza: "Pizza",
			sushi: "Sushi",
			icecream: "Ice Cream",
			pasta: "Pasta",
			sate: "Sate",
	    }[x]

	# load the image
	image = cv2.imread(image)
	orig = image.copy()
	image_name = str(random_with_N_digits(8)) + ".jpg"
	cv2.imwrite(image_name, image)
	# pre-process the image for classification
	image = cv2.resize(image, (128, 128))
	image = image.astype("float") / 255.0
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)

	# load the trained convolutional neural network
	print("[INFO] loading network...")
	with keras.backend.get_session().graph.as_default():
		model = load_model(model)

	# classify the input image
	with keras.backend.get_session().graph.as_default():
		(burger, pizza, sushi, icecream, pasta, sate) = model.predict(image)[0]

	# build the label
	label = label_switch(max([burger,pizza,sushi,icecream,pasta,sate]))
	proba = max([burger,pizza,sushi,icecream,pasta,sate])
	proba = int(proba*100)

	if(proba < 95):
		os.remove(image_name)
		print("image removed")
	else:
		print(image_name)
		print(label)
		os.rename(image_name, "images/" + label + "/" + image_name)
		print("image moved")

	return jsonify(status=200, food=label, probability=proba)
