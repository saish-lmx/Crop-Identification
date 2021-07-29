#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import Flask, render_template, request, redirect
from werkzeug.utils import secure_filename
from werkzeug.datastructures import  FileStorage
from flask_uploads import UploadSet, configure_uploads, IMAGES
from flask import flash
from flask import session
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras import backend as K
import os
import glob
import shutil

from keras.applications.vgg19 import VGG19

import numpy as np
import sklearn
#import Keras packages
import tensorflow as tf
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import load_img
import random
from keras.applications.imagenet_utils import preprocess_input, decode_predictions

import numpy as np
from keras import applications
from keras.layers import Input
from keras.models import Model,load_model
from keras import optimizers
from keras.utils import get_file


src_dir = "/static/img"
dst_dir = "/.static/img1"

app = Flask(__name__)
app.secret_key = "super secret key"

photos = UploadSet('photos', IMAGES)

app.config['UPLOADED_PHOTOS_DEST'] = 'static/img'
configure_uploads(app, photos)

@app.route('/', methods=['GET', 'POST'])
def upload():
    flash('')
    if request.method == 'POST' and 'photo' in request.files:
        	filename = photos.save(request.files['photo'])
        	for jpgfile in glob.iglob(os.path.join(src_dir, ".")):
        	  shutil.copy(jpgfile, dst_dir)
        
        	classifier = Sequential()

        	classifier.add(Convolution2D(64, 3, 3, input_shape = (128, 128, 3), activation = 'relu'))
        	classifier.add(MaxPooling2D(pool_size = (2, 2)))
        	classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
        	classifier.add(MaxPooling2D(pool_size = (2, 2)))
        	classifier.add(Convolution2D(16, 3, 3, activation = 'relu'))
        	classifier.add(MaxPooling2D(pool_size = (2, 2)))
        
        
        
        	classifier.add(Flatten())
        
        	#hidden layer
        	classifier.add(Dense(128, activation = 'relu'))
        	classifier.add(Dropout(0.5))
        
        	#output layer
        	classifier.add(Dense(32, activation = 'softmax'))
        
        	classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

        
        	model = classifier
        	#loading saved weights
        	model.load_weights("keras_plant_trained_model_weights_3000_augm.h5")


        	from keras.preprocessing.image import ImageDataGenerator
        
        	test_datagen = ImageDataGenerator(rescale=1./255)
        
        
        	itr1 = test_set1 = test_datagen.flow_from_directory(
                'static',
                target_size=(128, 128),
                batch_size=377,
                class_mode='categorical')
        
        	X1, y1 = itr1.next()
        	arr = model.predict(X1, batch_size=377, verbose=1)
        
        	arr = np.argmax(arr, axis=1)
			
			

           
        	"""
        	i = 0
        	j = 0
        	while(i < len(arr)):
        	  if(arr[i] == 1):
        	    j += 1
        	  i += 1
        	"""
	 
        
        	# flash(str(arr[0]))
        	if(arr[0] == 0):
				
        	  flash('Leaf is Damaged')
        	elif(arr[0] == 1):
        	  flash('Its a Apple Leaf having Black rots')        	      
        	elif(arr[0] == 2):
        	  flash('Leaf is a Apple Leaf Looks Little Rusty Might be Cedar')        	      
        	elif(arr[0] == 3):
        	  flash('Its a Healthy Apple Leaf')        	      
        	elif(arr[0] == 4):
        	  flash('Leaf is having4______________')        	      
        	elif(arr[0] == 5):
        	  flash('Its a corn Leaf with some Common Rust')        	      
        	elif(arr[0] == 6):
        	  flash('Its a corn Leaf with Grey Spots can be Cercospora')        	      
        	elif(arr[0] == 7):
        	  flash('Its a Healthy Corn Leaf')        	      
        	elif(arr[0] == 8):
        	  flash('Leaf is having8______________')        	      
        	elif(arr[0] == 9):
        	  flash('Grape Leaf with some Black rots')        	      
        	elif(arr[0] == 10):
        	  flash('Its a Grape Leaf, spots are Visible can be Isariopsis')        	      
        	elif(arr[0] == 11):
        	  flash('Its a Healthy Grape Leaf!')        	      
        	elif(arr[0] == 12):
        	  flash('Its a Orange Leaf Haunglongbing Citrus Greening')        	      
        	elif(arr[0] == 13):
        	  flash('Its a Peach Leaf with some bacterial Spots')        	      
        	elif(arr[0] == 14):
        	  flash('Its a Healthy Peach Leaf')        	      
        	elif(arr[0] == 15):
        	  flash('Its a bell Pepper Leaf with Bacterial Sopts')        	      
        	elif(arr[0] == 16):
        	  flash('Its a Healthy bell Pepper')        	      
        	elif(arr[0] == 17):
        	  flash('Leaf is a Apple Leaf with Scubs')        	      
        	elif(arr[0] == 18):
        	  flash('Its a poptato Leaf which is Late Bright')        	      
        	elif(arr[0] == 19):
        	  flash('Leaf is having19______________')        	      
        	elif(arr[0] == 20):
        	  flash('Its a Healthy Potato Leaf')        	      
        	elif(arr[0] == 21):
        	  flash('Its a poptato Leaf which is Early Bright')        	      
        	elif(arr[0] == 22):
        	  flash('Its a Healthy Strawberry Leaf')        	      
        	elif(arr[0] == 23):
        	  flash('Its a Tomato Leaf with some bacterial Spots')        	      
        	elif(arr[0] == 24):
        	  flash('Its a Tomato Leaf which is Early Bright')        	      
        	elif(arr[0] == 25):
        	  flash('Its a Tomato Leaf Which is Late Blight')        	      
        	elif(arr[0] == 26):
        	  flash('Leaf is having26______________')        	      
        	elif(arr[0] == 27):
        	  flash('Its a Tomato Leaf with Septoria Leaf Spots')        	      
        	elif(arr[0] == 28):
        	  flash('Leaf is having28______________')        	      
        	elif(arr[0] == 29):
        	  flash('Its a Tomato Leaf with Yellow Leaf Curl Virus')        	      
        	elif(arr[0] == 30):
        	  flash('Its a Tomato Leaf with Mosaic Virus')        	      
        	else:
			  	
        	  flash('Leaf is Healthy')
        
        	#K.clear_session()
        
        	K.clear_session()
        
        	os.remove('static/img/' + filename)
        	return render_template('index.html', user_image = 'static/img1/' + filename)
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)




