import keras
import tensorflow as tf
import cv2
from PIL import Image
import io
import numpy as np
from io import BytesIO
import tensorflow as tf
import keras
from skimage import transform
import gc
from matplotlib import cm
def loadImage(filename):
   np_image = Image.open(filename)
   width, height = np_image.size
   np_image = np.array(np_image).astype('float32')/255
   np_image = transform.resize(np_image, (48,48,3))
   np_image = np.expand_dims(np_image, axis=0)
   return np_image,width,height

model = tf.keras.Sequential([
        keras.layers.Conv2D(64,(3,3), activation='relu',input_shape=(48,48,3)),
        keras.layers.MaxPool2D(2,2),
        keras.layers.Conv2D(64,(3,3), activation='relu'),
        keras.layers.MaxPool2D(2,2),
        keras.layers.Flatten(),
        keras.layers.Dense(128,activation='relu'),
        keras.layers.Dense(7,activation='softmax')
    ])

#LOAD WHOLE MODEL
#model = keras.models.load_model("saves/95+ acc")

model.load_weights("saves/model-BL-epoch-130- Acc-0.991685.hdf5")

img,w,h = loadImage("../Datasets/Face Expression/images/validation/disgust/19734.jpg")
s = img[0,:,:,:].astype("float32") * 255 
im = Image.fromarray(s.astype(np.uint8))
#im.show()
pred = model.predict(img)[0]
print("%.2f %.2f %.2f %.2f %.2f %.2f %.2f" %(pred[0],pred[1],pred[2],pred[3],pred[4],pred[5],pred[6]))