import tensorflow as tf
import model
import numpy as np
import create_shapes
import cv2
import backgrounds
classes_to_use = [1,3,4,5]

train_images = []
train_labels = []
#TODO:
#Generate the dataset
data, labels, val_data, val_labels = create_shapes.get_train_data("data/caltech101_silhouettes_16_split1.mat", classes_to_use)
#Generate the testing set

#Create the model
the_model = model.model()
#Train train model
the_model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
the_model.fit(data, labels, batch_size=32, epochs=20, validation_data=(val_data, val_labels))
#Test the model

#Show results
