import numpy as np
import tensorflow as tf

# load full model (likely to cause lag)
class KeyPointClassifierFull(object):
    def __init__(self, model_path='model/keypoint_classifier/keypoint_classifier.hdf5'):
        self.model = tf.keras.models.load_model(model_path, compile=False)

    def __call__(self, landmark_list):
        result = self.model.predict(np.array([landmark_list], dtype=np.float32))
        result_index = np.argmax(np.squeeze(result))
        return result_index