# USAGE
# python test_network.py --model santa_not_santa.model --image images/examples/santa_01.png

# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import keras
import numpy as np
import argparse
import imutils
import cv2
import tensorflow as tf
from flask import session


def imageClassification(image):
# construct the argument parse and parse the arguments
	# ap = argparse.ArgumentParser()
	# ap.add_argument("-m", "--model", required=True,
	# 	help="path to trained model model")
	# ap.add_argument("-i", "--image", required=True,
	# 	help="path to input image")
	# args = vars(ap.parse_args())

	model = "food.model"

	burger=0
	pizza=0
	sushi=0
	friedchicken=0
	pasta=0
	sate=0

	def label_switch(x):
		return {
	        burger: "Burger",
	        pizza: "Pizza",
			sushi: "Sushi",
			friedchicken: "Fried Chicken",
			pasta: "Pasta",
			sate: "Sate",
	    }[x]

	# load the image
	image = cv2.imread(image)
	orig = image.copy()

	# pre-process the image for classification
	image = cv2.resize(image, (128, 128))
	image = image.astype("float") / 255.0
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)

	# load the trained convolutional neural network
	print("[INFO] loading network...")
	with keras.backend.get_session().graph.as_default():
		model = load_model(model)
	# model = load_model(model)

	# classify the input image
	with keras.backend.get_session().graph.as_default():
		(burger, pizza, sushi, friedchicken, pasta, sate) = model.predict(image)[0]

	# build the label
	label = label_switch(max([burger,pizza,sushi,friedchicken,pasta,sate]))
	proba = max([burger,pizza,sushi,friedchicken,pasta,sate])
	label = "{}: {:.2f}%".format(label, proba * 100)

	# print("burger:",burger)
	# print("pizza:",pizza)
	# print("sushi:",sushi)
	# print("chicken:",friedchicken)
	# print("pasta:",pasta)
	# print("sate:",sate)
	#
	# # label = "Burger" if burger > pizza else "Pizza"
	# # proba = burger if burger > pizza else pizza
	# # label = "{}: {:.2f}%".format(label, proba * 100)
	#
	# # draw the label on the image
	# output = imutils.resize(orig, width=400)
	# cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,
	# 	0.7, (0, 255, 0), 2)
	#
	# # show the output image
	# cv2.imshow("Output", output)
	# cv2.waitKey(0)

	return label
