import keras
import tensorflow as tf
base_dir = "D:/Machine learning projects/EMOTION DETECTION/saves/"
class myCallback(keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs={}):
        if(logs.get('accuracy') >=0.999):
            print("\nReached 99'%' accuracy so cancelling training!")
            self.model.save(base_dir + "95+ acc")
            self.model.stop_training = True





