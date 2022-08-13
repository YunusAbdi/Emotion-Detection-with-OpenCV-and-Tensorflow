import tensorflow as tf
import keras.backend as kb
import keras



def model():
    model = tf.keras.Sequential([
        keras.layers.Conv2D(64,(3,3), activation='relu',input_shape=(48,48,3)),
        keras.layers.MaxPool2D(2,2),
        keras.layers.Conv2D(64,(3,3), activation='relu'),
        keras.layers.MaxPool2D(2,2),
        keras.layers.Flatten(),
        keras.layers.Dense(128,activation='relu'),
        keras.layers.Dense(7,activation='softmax')
    ])

    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

    return model