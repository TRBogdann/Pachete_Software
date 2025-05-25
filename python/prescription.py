import keras
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from keras.src.saving import register_keras_serializable

@register_keras_serializable()
def ctc_loss(y_true, y_pred):
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

    return tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)

def load_and_preprocess_image(filepath, target_size=(256, 128)):
    image = tf.io.read_file(filepath)
    image = tf.image.decode_png(image, channels=1)
    image = tf.image.resize_with_pad(image, target_size[1], target_size[0])
    image = image / 255.0
    return image

img_path = './dataIN/488.png'
img = load_and_preprocess_image(img_path)
img = tf.expand_dims(img, axis=0)

model = load_model('./dataOUT/raw_prescription_model.keras')
prediction = model.predict(img)
print(prediction[0].shape)
print((np.argmax(prediction[0],axis=-1)).shape)
print("Predictions:", np.argmax(prediction[0],axis=-1))