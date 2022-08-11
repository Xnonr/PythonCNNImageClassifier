#!/usr/bin/env python
# coding: utf-8

# In[ ]:


'''
Based off of the 'CNN Image Classifier' tutorial by Nicholas Renotte
Tutorial Video Link: 'https://www.youtube.com/watch?v=jztwpsIzEGc'
'''


# In[ ]:


# Installing Dependencies
#!pip3 install matplotlib
#!pip3 install opencv-python
#!pip3 install tensorflow
#!pip3 install tensorflow-gpu


# In[ ]:


# Displaying Available Pip Libraries
#!pip3 list


# In[ ]:


# Importing Dependencies
import cv2
import imghdr
import numpy as np
import os
import tensorflow as tf

from matplotlib import pyplot as plt

from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import MaxPooling2D

from tensorflow.keras.metrics import BinaryAccuracy
from tensorflow.keras.metrics import Precision
from tensorflow.keras.metrics import Recall

from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential


# In[ ]:


# Removing Dodgy Images
image_data_directory = '../../Data/ImageRecognition'


# In[ ]:


valid_image_extensions = ['bmp','jpeg', 'jpg', 'png']


# In[ ]:


os.listdir(image_data_directory)


# In[ ]:


os.listdir(os.path.join(image_data_directory, 'Boats'))


# In[ ]:


for image_class in os.listdir(image_data_directory):
    for image in os.listdir(os.path.join(image_data_directory, image_class)):
        image_path = os.path.join(image_data_directory, image_class, image)
        try:
            current_image = cv2.imread(image_path)
            current_image_extension = imghdr.what(image_path)
            if(current_image_extension not in valid_image_extensions):
                print('This image has an extension of {} which s invalid and not included within the current valid headers.'.format(image_path))
                os.remove(image_path)
        except Exception as exception:
            print('Issue with image {}'.format(image_path))


# In[ ]:


# Displaying an example sample test image
test_image = cv2.imread(os.path.join(image_data_directory, 'Boats', 'TS_Harbor_Sunrise.jpg'))


# In[ ]:


test_image.shape


# In[ ]:


plt.imshow(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))
plt.show


# In[ ]:


# Loading In Data
image_dataset = tf.keras.utils.image_dataset_from_directory(image_data_directory)


# In[ ]:


image_dataset


# In[ ]:


image_dataset_iterator = image_dataset.as_numpy_iterator()


# In[ ]:


image_dataset_iterator


# In[ ]:


# Retrives a batch of images as data from the iterator
batch = image_dataset_iterator.next()


# In[ ]:


# Images represented as numpy arrays
batch[0].shape


# In[ ]:


# Class 1 = Boats | Class 2 = Planes
batch[1]


# In[ ]:


figure, subplot = plt.subplots(ncols = 4, figsize = (20, 20))
for index, image in enumerate(batch[0][:4]):
    subplot[index].imshow(image.astype(int))
    subplot[index].title.set_text(batch[1][index])


# In[ ]:


# Preprocessing Data

# Scaling Data

# Cutting down
image_dataset = image_dataset.map(lambda x, y: (x / 255, y))


# In[ ]:


scaled_image_dataset_iterator = image_dataset.as_numpy_iterator()


# In[ ]:


batch = scaled_image_dataset_iterator.next()


# In[ ]:


figure, subplot = plt.subplots(ncols = 4, figsize = (20, 20))
for index, image in enumerate(batch[0][:4]):
    subplot[index].imshow(image)
    subplot[index].title.set_text(batch[1][index])


# In[ ]:


# Splitting Data

training_ds_size = int(len(image_dataset) * 0.7)
validation_ds_size = int(len(image_dataset) * 0.2) + 1
testing_ds_size = int(len(image_dataset) * 0.1) + 1


# In[ ]:


training_ds = image_dataset.take(training_ds_size)
validation_ds = image_dataset.skip(training_ds_size).take(validation_ds_size)
testing_ds = image_dataset.skip(training_ds_size + validation_ds_size).take(testing_ds_size)


# In[ ]:


model = Sequential()


# In[ ]:


'''
First layer must have an input
First layer has 16 filters of 3 by 3 pixels in size with a stride of 1, moving over by
    1 pixel each time, these values being called architectural decisions
A relu activation takes the output of the convolutional layer and transforms it via a function,
    setting all previously negative values to 0 but preserving postive values as is, allowing
    for the taking into consideration of non linear patterns

'''
model.add(Conv2D(16, (3,3), 1, activation = 'relu', input_shape = (256, 256, 3)))

'''
Takes in the maximum value after the relu activation and returns said value, scanning across the output
    and condensing the information over a set region and is not simply returning a singular value
'''
model.add(MaxPooling2D())

# Convolutional Block 2
model.add(Conv2D(32, (3,3), 1, activation = 'relu'))
model.add(MaxPooling2D())

# Convolutional Block 3
model.add(Conv2D(16, (3,3), 1, activation = 'relu'))
model.add(MaxPooling2D())

'''
Flattening the data down
When applying convolutional layers, the filters act as the last channel
Condense the rows and width and the number of filters form the channel value
This is undersired when passing into the dense layer where only a signle value is the goal

'''
model.add(Flatten())

# Dense layers are fully connected layers
model.add(Dense(256, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))


# In[ ]:


'''
When compiling the model, the paramaters passed in include the type of optimizer, the loss, 
    and the metrics desired to be tracked
'''
model.compile('adam', loss = tf.losses.BinaryCrossentropy(), metrics = ['accuracy'])


# In[ ]:


model.summary()


# In[ ]:


# Training The Model


# In[ ]:


logs_directory = 'ImageClassifierLogs'


# In[ ]:


tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logs_directory)


# In[ ]:


history = model.fit(training_ds, 
                    epochs = 20, 
                    validation_data = validation_ds, 
                    callbacks = [tensorboard_callback])


# In[ ]:


# Plotting Model Performance

# Loss
loss_figure = plt.figure()
plt.plot(history.history['loss'], color = 'teal', label = 'Loss')
plt.plot(history.history['val_loss'], color = 'orange', label = 'Validation Loss')
loss_figure.suptitle('Loss', fontsize = 20)
plt.legend(loc = 'upper left')
plt.show()


# In[ ]:


# Accuracy
accuracy_figure = plt.figure()
plt.plot(history.history['accuracy'], color = 'teal', label = 'Accuracy')
plt.plot(history.history['val_accuracy'], color = 'orange', label = 'Validation Accuracy')
accuracy_figure.suptitle('Accuracy', fontsize = 20)
plt.legend(loc = 'upper left')
plt.show()


# In[ ]:


# Evaluating Model Performance

model_precision = Precision()
model_recall = Recall()
model_accuracy = BinaryAccuracy()


# In[ ]:


len(testing_ds)


# In[ ]:


for batch in testing_ds.as_numpy_iterator():
    x, y = batch
    yhat = model.predict(x)
    model_precision.update_state(y, yhat)
    model_recall.update_state(y, yhat)
    model_accuracy.update_state(y, yhat)


# In[ ]:


print(f'Precision: {model_precision.result().numpy()}')
print(f'Recall: {model_recall.result().numpy()}')
print(f'Accuracy: {model_accuracy.result().numpy()}')


# In[ ]:


# Testing The Model
test_image = cv2.imread(os.path.join(image_data_directory, 'Boats', 'TS_Harbor_Sunrise.jpg'))
plt.imshow(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))
plt.show()


# In[ ]:


resized_test_image = tf.image.resize(test_image, (256, 256))
plt.imshow(resized_test_image.numpy().astype(int))
plt.show()


# In[ ]:


'''
The model expects a batch of images, not a single individual one, so in order to test this
    the image itself must be encapsulated inside another set of parantheses or placed inside
    of a list
'''
test_image_prediction = model.predict(np.expand_dims(resized_test_image / 255, 0))


# In[ ]:


resized_test_image.shape


# In[ ]:


np.expand_dims(resized_test_image, 0).shape


# In[ ]:


yhat = model.predict(np.expand_dims(resized_test_image / 255, 0))


# In[ ]:


yhat


# In[ ]:


if(yhat > 0.5):
    print('The image is predicted to represent a boat.')
else:
    print('The image is predicted to represent a plane.')


# In[ ]:


# Saving The Model

model.save(os.path.join('models', 'imageClassifierModelBoatPlanes.h5'))


# In[ ]:


loaded_model = load_model(os.path.join('models', 'imageClassifierModelBoatPlanes.h5'))

