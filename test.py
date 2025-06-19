from keras import applications
from keras.layers import Input
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json
import os
from keras.preprocessing import image
import numpy as np
from keras.layers import Convolution2D
import cv2
import pickle

input_tensor = Input(shape=(64, 64, 3))
vgg_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(64, 64, 3)) #VGG16 transfer learning code here
vgg_model.summary()
layer_dict = dict([(layer.name, layer) for layer in vgg_model.layers])
x = layer_dict['block2_pool'].output
x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(4, activation='softmax')(x)
custom_model = Model(input=vgg_model.input, output=x)
for layer in custom_model.layers[:7]:
    layer.trainable = False
custom_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
train_datagen = ImageDataGenerator(rescale = 1./255., rotation_range = 40, width_shift_range = 0.2, height_shift_range = 0.2,shear_range = 0.2,
                                   zoom_range = 0.2, horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1.0/255.)
training_set = train_datagen.flow_from_directory('Dataset',target_size = (64, 64), batch_size = 2, class_mode = 'categorical', shuffle=True)
test_set = test_datagen.flow_from_directory('Dataset',target_size = (64, 64), batch_size = 2, class_mode = 'categorical', shuffle=False)
hist = custom_model.fit_generator(training_set,samples_per_epoch = 500,nb_epoch = 10,validation_data = test_set,nb_val_samples = 125)
custom_model.save_weights('model/vgg_weights.h5')
model_json = custom_model.to_json()
with open("model/vgg_model.json", "w") as json_file:
    json_file.write(model_json)
print(training_set.class_indices)
print(custom_model.summary())
f = open('model/vgg_history.pckl', 'wb')
pickle.dump(hist.history, f)
f.close()
