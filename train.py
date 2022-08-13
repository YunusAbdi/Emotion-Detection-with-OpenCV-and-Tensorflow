import tensorflow as tf
import keras
import keras.backend as kb
import model
import read_data
import my_callbacks
import numpy as np

SHUFFLE_BUFFER_SIZE = 1000
BATCH_SIZE = 64
x_train,y_train,x_val,y_val = read_data.import_data()
train_dataset = tf.data.Dataset.from_tensor_slices((x_train,y_train))
train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE).repeat(1000)
validation_dataset = tf.data.Dataset.from_tensor_slices((x_val,y_val))
validation_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE).repeat(1000)
my_model = model.model()
callback = my_callbacks.myCallback()
model_save_path = "D:/Machine learning projects/EMOTION DETECTION/saves/model-BL-epoch-{epoch:02d}- Acc-{accuracy:02f}.hdf5"
save_model_callback = tf.keras.callbacks.ModelCheckpoint(
  monitor='val_accuracy',
  mode='auto',
  filepath=model_save_path,
  save_weights_only=True,
  save_freq=2800,
  verbose=1)


my_model.fit(x_train,y_train,
steps_per_epoch=280,
batch_size=64,
validation_steps=30,
epochs=200,verbose=1,
callbacks=[callback,save_model_callback],
)
