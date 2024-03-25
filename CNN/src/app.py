from flask import Flask, request
from flask_cors import CORS
from keras.models import load_model
from cnn_model.custom_objects import layers
from cnn_model.custom_objects import metrics
import json
from preprocess import prepare_image
import sys
import numpy as np
import keras
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
<<<<<<< HEAD
=======
import os
>>>>>>> 0f3f6fb (Initial commit)

app = Flask(__name__)
CORS(app)
model = None


def load_cnn_model():
	global model
<<<<<<< HEAD
=======
	global graph
	graph = tf.compat.v1.get_default_graph()
>>>>>>> 0f3f6fb (Initial commit)
	model = load_model('./cnn_model/calista_comparison_based.h5', \
	    custom_objects = {
			'LRN': layers.LRN,
			'euclidean_distance_loss': metrics.euclidean_distance_loss,
			'rmse': metrics.rmse
		})
	model.summary()
	print('[*] Model loaded')
<<<<<<< HEAD
	global graph
	graph = tf.get_default_graph()
=======

>>>>>>> 0f3f6fb (Initial commit)

@app.route('/')
def index():
	return "Flask server"

<<<<<<< HEAD
@app.route('/run_cnn', methods = ['POST'])
def postdata():
=======
graph = tf.compat.v1.get_default_graph()

@app.route('/run_cnn', methods = ['POST'])
def postdata():


	# model_path = './cnn_model/calista_comparison_based.h5'
	# print("Checking if model file exists:", os.path.exists(model_path))

	# # Check file existence
	# if os.path.exists(model_path):
	# 	# Load the model
	# 	model = load_model(model_path)
		
	# 	# Print model summary
	# 	print("Model loaded successfully. Summary:", file=sys.stderr)
	# 	model.summary()
	# else:
	# 	print("Model file not found at the specified path:", file=sys.stderr)



>>>>>>> 0f3f6fb (Initial commit)
	data = request.get_json()

	imagePath = data.get('imagePath')

	input_image = prepare_image(imagePath)
	print('Evaluating webpage ...', file=sys.stderr)

	test_datagen = ImageDataGenerator(rescale = 1./255)
	test_data = test_datagen.flow(input_image, batch_size=1, shuffle=False).next()
<<<<<<< HEAD

	with graph.as_default():
		score = model.predict(test_data)
		score = float(score)
=======
	with graph.as_default():
		score = model.predict(test_data)
		score = float(score)
		

>>>>>>> 0f3f6fb (Initial commit)

		# score bound protection
		score = np.minimum(score, 10.0)
		score = np.maximum(score, 1.0)

		print('CNN score: ' + str(score), file=sys.stderr)

	return json.dumps({"score": score})

if __name__ == "__main__":

    load_cnn_model()
    app.run(host='0.0.0.0', port=5000, debug=True)
