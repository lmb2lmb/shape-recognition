import tensorflow as tf

class model(tf.keras.Model):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.the_model = tf.keras.Sequential()
        self.the_model.add(tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same'))
        self.the_model.add(tf.keras.layers.MaxPool2D())
        self.the_model.add(tf.keras.layers.BatchNormalization())
        self.the_model.add(tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same'))
        self.the_model.add(tf.keras.layers.MaxPool2D())
        self.the_model.add(tf.keras.layers.BatchNormalization())
        self.the_model.add(tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same'))
        self.the_model.add(tf.keras.layers.MaxPool2D())
        self.the_model.add(tf.keras.layers.BatchNormalization())
        self.the_model.add(tf.keras.layers.Flatten())
        self.the_model.add(tf.keras.layers.Dense(128, activation='relu'))
        self.the_model.add(tf.keras.layers.Dense(4, activation='softmax'))

    def call(self, data):
        return self.the_model(data)
