
import tensorflow_transform as tft
import tensorflow as tf

IMAGE_KEY = 'image_raw'
LABEL_KEY = 'label'
IMAGE_SIZE = 32

def transformed_name(name):
  return name + '_xf'

def _image_parser(image_str):
    image = tf.image.decode_image(image_str, channels=3)
    image = tf.reshape(image, (IMAGE_SIZE, IMAGE_SIZE, 3))
    #image = tf.cast(image, tf.float32) / 255.
    return image

def _label_parser(label_id):
    label = tf.one_hot(label_id, 10)
    return label

def preprocessing_fn(inputs):
    outputs = {transformed_name(IMAGE_KEY): tf.compat.v2.map_fn(_image_parser, tf.squeeze(inputs[IMAGE_KEY], axis=1),
                                                                  dtype=tf.float32),
               transformed_name(LABEL_KEY): tf.compat.v2.map_fn(_label_parser, tf.squeeze(inputs[LABEL_KEY], axis=1),
                                                                  dtype=tf.float32)
               }
    return outputs
