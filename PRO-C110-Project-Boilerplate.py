# To Capture Frame
import cv2

# To process image array
import numpy as np


# import the tensorflow modules and load the model
import os
import tensorflow as tf
from tensorflow import keras

loaded_model = tf.keras.models.load_model('keras_model.h5')



# Attaching Cam indexed as 0, with the application software
camera = cv2.VideoCapture(0)

# Infinite loop
while True:

	# Reading / Requesting a Frame from the Camera 
	status , frame = camera.read()

	# if we were sucessfully able to read the frame
	if status:

		# Flip the frame
		frame = cv2.flip(frame , 1)
		
		
		
		#resize the frame
		resized_frame = cv2.resize(frame, (640, 480))

		
		# expand the dimensions
		expanded_frame = np.expand_dims(resized_frame, axis=0)

		
		# normalize it before feeding to the model
		normalized_frame = expanded_frame / 255.0

		
		# get predictions from the model
		predictions = loaded_model.predict(normalized_frame)

		
		
		
		# displaying the frames captured
		cv2.imshow('feed' , frame)

		# waiting for 1ms
		code = cv2.waitKey(1)
		
		# if space key is pressed, break the loop
		if code == 32:
			break

# release the camera from the application software
camera.release()

# close the open window
cv2.destroyAllWindows()
