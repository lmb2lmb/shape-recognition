import tensorflow as tf
import model
import create_shapes
classes_to_use = [7,21,27,82]
data_shape = (25,25)
#TODO:
#Generate the dataset
data, labels, val_data, val_labels, test_data, test_labels = create_shapes.get_train_data("data/caltech101_silhouettes_16_split1.mat", classes_to_use, data_shape)
#Generate the testing set
#Create the model
the_model = model.model()
#Train train model
the_model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
the_model.fit(data, labels, batch_size=32, epochs=10, validation_data=(val_data, val_labels))
the_model.summary()
#Test the model
predictions = the_model.predict(test_data)
predictions = tf.argmax(predictions, axis=1)
accuracy = tf.keras.metrics.Accuracy()
accuracy.update_state(test_labels, predictions)
result = accuracy.result().numpy()
#Show results
print("accuracy:", result)
