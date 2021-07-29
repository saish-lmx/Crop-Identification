#!/usr/bin/env python
# coding: utf-8

# In[5]:


#!/usr/bin/env python
# coding: utf-8

# In[5]:


#!/usr/bin/env python
# coding: utf-8

# In[9]:





# In[10]:


#!/usr/bin/env python
# coding: utf-8

# In[5]:


# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 11:29:48 2018

@author: mohit123
"""
import numpy as np
from keras import applications
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Input
from keras.models import Model
from keras import optimizers


np.random.seed(1337)
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



from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'D:\\Z-Mini project\\rabi\\train',
        target_size=(128, 128),
        batch_size=64,
        class_mode='categorical' )
label_map = (training_set.class_indices)

print(label_map)

itr = test_set = test_datagen.flow_from_directory(
        'D:\\Z-Mini project\\rabi\\testing\\train1',
        target_size=(128, 128),
        batch_size=377,
        class_mode='categorical')

X, y = itr.next()
k = 0
classifier.load_weights('D:\\Z-Mini project\\rabi\\keras_plant_trained_model_weights_3000_augm.h5')

#for layer in classifier.layers:
#    g=layer.get_config()
#    h=layer.get_weights()
#    print (g)
#    print (h)

#scores = classifier.evaluate_generator(test_set,62/32)

arr = classifier.predict_classes(X, batch_size=377, verbose=1)
#arr = np.argmax(arr, axis=1)
i = 0
j = 0
print(arr)
while i < len(arr):
   
        j += 1
        k += 1
        i += 1

print ('Correct Early Blight predictions')
print(j/len(arr))

# Healthy test cases
itr = test_set = test_datagen.flow_from_directory(
        'D:\\Z-Mini project\\rabi\\testing\\train2',
        target_size=(128, 128),
        batch_size=377,
        class_mode='categorical')

X, y = itr.next()

classifier.load_weights('D:\\Z-Mini project\\rabi\\keras_plant_trained_model_weights_3000_augm.h5')

#for layer in classifier.layers:
#    g=layer.get_config()
#    h=layer.get_weights()
#    print (g)
#    print (h)

#scores = classifier.evaluate_generator(test_set,62/32)
arr = classifier.predict_classes(X, batch_size=377, verbose=1)
#arr = np.argmax(arr, axis=1)
i = 0
j = 0
print(arr)
while i < len(arr):
    
        j += 1
        k += 1
        i += 1

print ('Correct healthy predictions:')
print(j/len(arr))

itr = test_set = test_datagen.flow_from_directory(
        'D:\\Z-Mini project\\rabi\\testing\\train3',
        target_size=(128, 128),
        batch_size=377,
        class_mode='categorical')

X, y = itr.next()

classifier.load_weights('D:\\Z-Mini project\\rabi\\keras_plant_trained_model_weights_3000_augm.h5')

#for layer in classifier.layers:
#    g=layer.get_config()
#    h=layer.get_weights()
#    print (g)
#    print (h)

#scores = classifier.evaluate_generator(test_set,62/32)
arr = classifier.predict_classes(X, batch_size=377, verbose=1)
#arr = np.argmax(arr, axis=1)

i = 0
j = 0
print(arr)
while i < len(arr):
   
        j += 1
        k += 1
        i += 1

print ('Correct Late blight predictions:')
print(j/len(arr))

itr = test_set = test_datagen.flow_from_directory(
        'D:\\Z-Mini project\\rabi\\testing\\train4',  #crop name folder
        target_size=(128, 128),
        batch_size=377,
        class_mode='categorical')

X, y = itr.next()

classifier.load_weights('D:\\Z-Mini project\\rabi\\keras_plant_trained_model_weights_3000_augm.h5')

#for layer in classifier.layers:
#    g=layer.get_config()
#    h=layer.get_weights()
#    print (g)
#    print (h)

#scores = classifier.evaluate_generator(test_set,62/32)
arr = classifier.predict_classes(X, batch_size=377, verbose=1)
#arr = np.argmax(arr, axis=1)
i = 0
j = 0
print(arr)
while i < len(arr):
   
        j += 1
        k += 1
        i += 1

print ('Correct Late blight predictions:')
print(j/len(arr))

itr = test_set = test_datagen.flow_from_directory(
        'D:\\Z-Mini project\\rabi\\testing\\train5',
        target_size=(128, 128),
        batch_size=377,
        class_mode='categorical')

X, y = itr.next()

classifier.load_weights('D:\\Z-Mini project\\rabi\\keras_plant_trained_model_weights_3000_augm.h5')

#for layer in classifier.layers:
#    g=layer.get_config()
#    h=layer.get_weights()
#    print (g)
#    print (h)

#scores = classifier.evaluate_generator(test_set,62/32)
arr = classifier.predict_classes(X, batch_size=377, verbose=1)
#arr = np.argmax(arr, axis=1)
i = 0
j = 0
print(arr)
while i < len(arr):
    
        j += 1
        k += 1
        i += 1

print ('Correct Late blight predictions:')
print(j/len(arr))

itr = test_set = test_datagen.flow_from_directory(
        'D:\\Z-Mini project\\rabi\\testing\\train6',
        target_size=(128, 128),
        batch_size=377,
        class_mode='categorical')

X, y = itr.next()

classifier.load_weights('D:\\Z-Mini project\\rabi\\keras_plant_trained_model_weights_3000_augm.h5')

#for layer in classifier.layers:
#    g=layer.get_config()
#    h=layer.get_weights()
#    print (g)
#    print (h)

#scores = classifier.evaluate_generator(test_set,62/32)
arr = classifier.predict_classes(X, batch_size=377, verbose=1)
#arr = np.argmax(arr, axis=1)
i = 0
j = 0
print(arr)
while i < len(arr):
   
        j += 1
        k += 1
        i += 1

print ('Correct Late blight predictions:')
print(j/len(arr))

itr = test_set = test_datagen.flow_from_directory(
        'D:\\Z-Mini project\\rabi\\testing\\train7',
        target_size=(128, 128),
        batch_size=377,
        class_mode='categorical')

X, y = itr.next()

classifier.load_weights('D:\\Z-Mini project\\rabi\\keras_plant_trained_model_weights_3000_augm.h5')

#for layer in classifier.layers:
#    g=layer.get_config()
#    h=layer.get_weights()
#    print (g)
#    print (h)

#scores = classifier.evaluate_generator(test_set,62/32)
arr = classifier.predict_classes(X, batch_size=377, verbose=1)
#arr = np.argmax(arr, axis=1)
i = 0
j = 0
print(arr)
while i < len(arr):
   
        j += 1
        k += 1
        i += 1

print ('Correct Late blight predictions:')
print(j/len(arr))

itr = test_set = test_datagen.flow_from_directory(
        'D:\\Z-Mini project\\rabi\\testing\\train8',
        target_size=(128, 128),
        batch_size=377,
        class_mode='categorical')

X, y = itr.next()

classifier.load_weights('D:\\Z-Mini project\\rabi\\keras_plant_trained_model_weights_3000_augm.h5')

#for layer in classifier.layers:
#    g=layer.get_config()
#    h=layer.get_weights()
#    print (g)
#    print (h)

#scores = classifier.evaluate_generator(test_set,62/32)
arr = classifier.predict_classes(X, batch_size=377, verbose=1)
#arr = np.argmax(arr, axis=1)
i = 0
j = 0
print(arr)
while i < len(arr):
    
        j += 1
        k += 1
        i += 1

print ('Correct Late blight predictions:')
print(j/len(arr))

itr = test_set = test_datagen.flow_from_directory(
        'D:\\Z-Mini project\\rabi\\testing\\train9',
        target_size=(128, 128),
        batch_size=377,
        class_mode='categorical')

X, y = itr.next()

classifier.load_weights('D:\\Z-Mini project\\rabi\\keras_plant_trained_model_weights_3000_augm.h5')

#for layer in classifier.layers:
#    g=layer.get_config()
#    h=layer.get_weights()
#    print (g)
#    print (h)

#scores = classifier.evaluate_generator(test_set,62/32)
arr = classifier.predict_classes(X, batch_size=377, verbose=1)
#arr = np.argmax(arr, axis=1)
i = 0
j = 0
print(arr)
while i < len(arr):
    
        j += 1
        k += 1
        i += 1

print ('Correct Late blight predictions:')
print(j/len(arr))

itr = test_set = test_datagen.flow_from_directory(
        'D:\\Z-Mini project\\rabi\\testing\\train10',
        target_size=(128, 128),
        batch_size=377,
        class_mode='categorical')

X, y = itr.next()

classifier.load_weights('D:\\Z-Mini project\\rabi\\keras_plant_trained_model_weights_3000_augm.h5')

#for layer in classifier.layers:
#    g=layer.get_config()
#    h=layer.get_weights()
#    print (g)
#    print (h)

#scores = classifier.evaluate_generator(test_set,62/32)
arr = classifier.predict_classes(X, batch_size=377, verbose=1)
#arr = np.argmax(arr, axis=1)
i = 0
j = 0
print(arr)
while i < len(arr):
   
        j += 1
        k += 1
        i += 1

print ('Correct Late blight predictions:')
print(j/len(arr))

itr = test_set = test_datagen.flow_from_directory(
        'D:\\Z-Mini project\\rabi\\testing\\train11',
        target_size=(128, 128),
        batch_size=377,
        class_mode='categorical')

X, y = itr.next()

classifier.load_weights('D:\\Z-Mini project\\rabi\\keras_plant_trained_model_weights_3000_augm.h5')

#for layer in classifier.layers:
#    g=layer.get_config()
#    h=layer.get_weights()
#    print (g)
#    print (h)

#scores = classifier.evaluate_generator(test_set,62/32)
arr = classifier.predict_classes(X, batch_size=377, verbose=1)
#arr = np.argmax(arr, axis=1)
i = 0
j = 0
print(arr)
while i < len(arr):
    
        j += 1
        k += 1
        i += 1

print ('Correct Late blight predictions:')
print(j/len(arr))


itr = test_set = test_datagen.flow_from_directory(
        'D:\\Z-Mini project\\rabi\\testing\\train12',
        target_size=(128, 128),
        batch_size=377,
        class_mode='categorical')

X, y = itr.next()

classifier.load_weights('D:\\Z-Mini project\\rabi\\keras_plant_trained_model_weights_3000_augm.h5')

#for layer in classifier.layers:
#    g=layer.get_config()
#    h=layer.get_weights()
#    print (g)
#    print (h)

#scores = classifier.evaluate_generator(test_set,62/32)
arr = classifier.predict_classes(X, batch_size=377, verbose=1)
#arr = np.argmax(arr, axis=1)
i = 0
j = 0
print(arr)
while i < len(arr):
   
        j += 1
        k += 1
        i += 1

print ('Correct Late blight predictions:')
print(j/len(arr))


itr = test_set = test_datagen.flow_from_directory(
        'D:\\Z-Mini project\\rabi\\testing\\train13',
        target_size=(128, 128),
        batch_size=377,
        class_mode='categorical')

X, y = itr.next()

classifier.load_weights('D:\\Z-Mini project\\rabi\\keras_plant_trained_model_weights_3000_augm.h5')

#for layer in classifier.layers:
#    g=layer.get_config()
#    h=layer.get_weights()
#    print (g)
#    print (h)

#scores = classifier.evaluate_generator(test_set,62/32)
arr = classifier.predict_classes(X, batch_size=377, verbose=1)
#arr = np.argmax(arr, axis=1)
i = 0
j = 0
print(arr)
while i < len(arr):
    if(arr[i] == 11):
        j += 1
        k += 1
    i += 1

print ('Correct Late blight predictions:')
print(j/len(arr))


itr = test_set = test_datagen.flow_from_directory(
        'D:\\Z-Mini project\\rabi\\testing\\train14',
        target_size=(128, 128),
        batch_size=377,
        class_mode='categorical')

X, y = itr.next()

classifier.load_weights('D:\\Z-Mini project\\rabi\\keras_plant_trained_model_weights_3000_augm.h5')

#for layer in classifier.layers:
#    g=layer.get_config()
#    h=layer.get_weights()
#    print (g)
#    print (h)

#scores = classifier.evaluate_generator(test_set,62/32)
arr = classifier.predict_classes(X, batch_size=377, verbose=1)
#arr = np.argmax(arr, axis=1)
i = 0
j = 0
print(arr)
while i < len(arr):
    
        j += 1
        k += 1
        i += 1

print ('Correct Late blight predictions:')
print(j/len(arr))


itr = test_set = test_datagen.flow_from_directory(
        'D:\\Z-Mini project\\rabi\\testing\\train15',
        target_size=(128, 128),
        batch_size=377,
        class_mode='categorical')

X, y = itr.next()

classifier.load_weights('D:\\Z-Mini project\\rabi\\keras_plant_trained_model_weights_3000_augm.h5')

#for layer in classifier.layers:
#    g=layer.get_config()
#    h=layer.get_weights()
#    print (g)
#    print (h)

#scores = classifier.evaluate_generator(test_set,62/32)
arr = classifier.predict_classes(X, batch_size=377, verbose=1)
#arr = np.argmax(arr, axis=1)
i = 0
j = 0
print(arr)
while i < len(arr):
    
        j += 1
        k += 1
        i += 1

print ('Correct Late blight predictions:')
print(j/len(arr))


itr = test_set = test_datagen.flow_from_directory(
        'D:\\Z-Mini project\\rabi\\testing\\train16',
        target_size=(128, 128),
        batch_size=377,
        class_mode='categorical')

X, y = itr.next()

classifier.load_weights('D:\\Z-Mini project\\rabi\\keras_plant_trained_model_weights_3000_augm.h5')

#for layer in classifier.layers:
#    g=layer.get_config()
#    h=layer.get_weights()
#    print (g)
#    print (h)

#scores = classifier.evaluate_generator(test_set,62/32)
arr = classifier.predict_classes(X, batch_size=377, verbose=1)
#arr = np.argmax(arr, axis=1)
i = 0
j = 0
print(arr)
while i < len(arr):
   
        j += 1
        k += 1
        i += 1

print ('Correct Late blight predictions:')
print(j/len(arr))


itr = test_set = test_datagen.flow_from_directory(
        'D:\\Z-Mini project\\rabi\\testing\\train17',
        target_size=(128, 128),
        batch_size=377,
        class_mode='categorical')

X, y = itr.next()

classifier.load_weights('D:\\Z-Mini project\\rabi\\keras_plant_trained_model_weights_3000_augm.h5')

#for layer in classifier.layers:
#    g=layer.get_config()
#    h=layer.get_weights()
#    print (g)
#    print (h)

#scores = classifier.evaluate_generator(test_set,62/32)
arr = classifier.predict_classes(X, batch_size=377, verbose=1)
#arr = np.argmax(arr, axis=1)
i = 0
j = 0
print(arr)
while i < len(arr):
   
        j += 1
        k += 1
        i += 1

print ('Correct Late blight predictions:')
print(j/len(arr))


itr = test_set = test_datagen.flow_from_directory(
        'D:\\Z-Mini project\\rabi\\testing\\train18',
        target_size=(128, 128),
        batch_size=377,
        class_mode='categorical')

X, y = itr.next()

classifier.load_weights('D:\\Z-Mini project\\rabi\\keras_plant_trained_model_weights_3000_augm.h5')

#for layer in classifier.layers:
#    g=layer.get_config()
#    h=layer.get_weights()
#    print (g)
#    print (h)

#scores = classifier.evaluate_generator(test_set,62/32)
arr = classifier.predict_classes(X, batch_size=377, verbose=1)
#arr = np.argmax(arr, axis=1)
i = 0
j = 0
print(arr)
while i < len(arr):
   
        j += 1
        k += 1
        i += 1

print ('Correct Late blight predictions:')
print(j/len(arr))


itr = test_set = test_datagen.flow_from_directory(
        'D:\\Z-Mini project\\rabi\\testing\\train19',
        target_size=(128, 128),
        batch_size=377,
        class_mode='categorical')

X, y = itr.next()

classifier.load_weights('D:\\Z-Mini project\\rabi\\keras_plant_trained_model_weights_3000_augm.h5')

#for layer in classifier.layers:
#    g=layer.get_config()
#    h=layer.get_weights()
#    print (g)
#    print (h)

#scores = classifier.evaluate_generator(test_set,62/32)
arr = classifier.predict_classes(X, batch_size=377, verbose=1)
#arr = np.argmax(arr, axis=1)
i = 0
j = 0
print(arr)
while i < len(arr):
    
        j += 1
        k += 1
        i += 1

print ('Correct Late blight predictions:')
print(j/len(arr))


itr = test_set = test_datagen.flow_from_directory(
        'D:\\Z-Mini project\\rabi\\testing\\train20',
        target_size=(128, 128),
        batch_size=377,
        class_mode='categorical')

X, y = itr.next()

classifier.load_weights('D:\\Z-Mini project\\rabi\\keras_plant_trained_model_weights_3000_augm.h5')

#for layer in classifier.layers:
#    g=layer.get_config()
#    h=layer.get_weights()
#    print (g)
#    print (h)

#scores = classifier.evaluate_generator(test_set,62/32)
arr = classifier.predict_classes(X, batch_size=377, verbose=1)
#arr = np.argmax(arr, axis=1)
i = 0
j = 0
print(arr)
while i < len(arr):
   
        j += 1
        k += 1
        i += 1

print ('Correct Late blight predictions:')
print(j/len(arr))


itr = test_set = test_datagen.flow_from_directory(
        'D:\\Z-Mini project\\rabi\\testing\\train21',
        target_size=(128, 128),
        batch_size=377,
        class_mode='categorical')

X, y = itr.next()

classifier.load_weights('D:\\Z-Mini project\\rabi\\keras_plant_trained_model_weights_3000_augm.h5')

#for layer in classifier.layers:
#    g=layer.get_config()
#    h=layer.get_weights()
#    print (g)
#    print (h)

#scores = classifier.evaluate_generator(test_set,62/32)
arr = classifier.predict_classes(X, batch_size=377, verbose=1)
#arr = np.argmax(arr, axis=1)
i = 0
j = 0
print(arr)
while i < len(arr):
   
        j += 1
        k += 1
        i += 1

print ('Correct Late blight predictions:')
print(j/len(arr))


itr = test_set = test_datagen.flow_from_directory(
        'D:\\Z-Mini project\\rabi\\testing\\train22',
        target_size=(128, 128),
        batch_size=377,
        class_mode='categorical')

X, y = itr.next()

classifier.load_weights('D:\\Z-Mini project\\rabi\\keras_plant_trained_model_weights_3000_augm.h5')

#for layer in classifier.layers:
#    g=layer.get_config()
#    h=layer.get_weights()
#    print (g)
#    print (h)

#scores = classifier.evaluate_generator(test_set,62/32)
arr = classifier.predict_classes(X, batch_size=377, verbose=1)
#arr = np.argmax(arr, axis=1)
i = 0
j = 0
print(arr)
while i < len(arr):
    
        j += 1
        k += 1
        i += 1

print ('Correct Late blight predictions:')
print(j/len(arr))


itr = test_set = test_datagen.flow_from_directory(
        'D:\\Z-Mini project\\rabi\\testing\\train23',
        target_size=(128, 128),
        batch_size=377,
        class_mode='categorical')

X, y = itr.next()

classifier.load_weights('D:\\Z-Mini project\\rabi\\keras_plant_trained_model_weights_3000_augm.h5')

#for layer in classifier.layers:
#    g=layer.get_config()
#    h=layer.get_weights()
#    print (g)
#    print (h)

#scores = classifier.evaluate_generator(test_set,62/32)
arr = classifier.predict_classes(X, batch_size=377, verbose=1)
#arr = np.argmax(arr, axis=1)
i = 0
j = 0
print(arr)
while i < len(arr):
    
        j += 1
        k += 1
        i += 1

print ('Correct Late blight predictions:')
print(j/len(arr))


itr = test_set = test_datagen.flow_from_directory(
        'D:\\Z-Mini project\\rabi\\testing\\train24',
        target_size=(128, 128),
        batch_size=377,
        class_mode='categorical')

X, y = itr.next()

classifier.load_weights('D:\\Z-Mini project\\rabi\\keras_plant_trained_model_weights_3000_augm.h5')

#for layer in classifier.layers:
#    g=layer.get_config()
#    h=layer.get_weights()
#    print (g)
#    print (h)

#scores = classifier.evaluate_generator(test_set,62/32)
arr = classifier.predict_classes(X, batch_size=377, verbose=1)
#arr = np.argmax(arr, axis=1)
i = 0
j = 0
print(arr)
while i < len(arr):
    
        j += 1
        k += 1
        i += 1

print ('Correct Late blight predictions:')
print(j/len(arr))


itr = test_set = test_datagen.flow_from_directory(
        'D:\\Z-Mini project\\rabi\\testing\\train25',
        target_size=(128, 128),
        batch_size=377,
        class_mode='categorical')

X, y = itr.next()

classifier.load_weights('D:\\Z-Mini project\\rabi\\keras_plant_trained_model_weights_3000_augm.h5')

#for layer in classifier.layers:
#    g=layer.get_config()
#    h=layer.get_weights()
#    print (g)
#    print (h)

#scores = classifier.evaluate_generator(test_set,62/32)
arr = classifier.predict_classes(X, batch_size=377, verbose=1)
#arr = np.argmax(arr, axis=1)
i = 0
j = 0
print(arr)
while i < len(arr):
    
        j += 1
        k += 1
        i += 1

print ('Correct Late blight predictions:')
print(j/len(arr))


itr = test_set = test_datagen.flow_from_directory(
        'D:\\Z-Mini project\\rabi\\testing\\train26',
        target_size=(128, 128),
        batch_size=377,
        class_mode='categorical')

X, y = itr.next()

classifier.load_weights('D:\\Z-Mini project\\rabi\\keras_plant_trained_model_weights_3000_augm.h5')

#for layer in classifier.layers:
#    g=layer.get_config()
#    h=layer.get_weights()
#    print (g)
#    print (h)

#scores = classifier.evaluate_generator(test_set,62/32)
arr = classifier.predict_classes(X, batch_size=377, verbose=1)
#arr = np.argmax(arr, axis=1)
i = 0
j = 0
print(arr)
while i < len(arr):
    
        j += 1
        k += 1
        i += 1

print ('Correct Late blight predictions:')
print(j/len(arr))


itr = test_set = test_datagen.flow_from_directory(
        'D:\\Z-Mini project\\rabi\\testing\\train27',
        target_size=(128, 128),
        batch_size=377,
        class_mode='categorical')

X, y = itr.next()

classifier.load_weights('D:\\Z-Mini project\\rabi\\keras_plant_trained_model_weights_3000_augm.h5')

#for layer in classifier.layers:
#    g=layer.get_config()
#    h=layer.get_weights()
#    print (g)
#    print (h)

#scores = classifier.evaluate_generator(test_set,62/32)
arr = classifier.predict_classes(X, batch_size=377, verbose=1)
#arr = np.argmax(arr, axis=1)
i = 0
j = 0
print(arr)
while i < len(arr):
   
        j += 1
        k += 1
        i += 1

print ('Correct Late blight predictions:')
print(j/len(arr))


print(k/(50*27))

#print(j)
    
#print(arr)
#print(len(scores))
#print(scores)
#print('Test loss:', scores[0])
#print('Test accuracy:', scores[1])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:






