import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

directory = './data'
train_data = directory + '/train'
IMG_SIZE = 32

def visualize(original, augmented):
  fig = plt.figure()
  plt.subplot(1,2,1)
  plt.title('Original image')
  plt.imshow(original)

  plt.subplot(1,2,2)
  plt.title('Augmented image')
  plt.imshow(augmented)

def resize_and_rescale(image, label):
  image = tf.cast(image, tf.float32)
  image = tf.image.resize(image, [IMG_SIZE,IMG_SIZE])
  image = (image / 255.0)
  return image, label


def augment(image,label):
  image, label = resize_and_rescale(image, label)
  # Add 6 pixels of padding
   # Random crop back to the original size
  image = tf.image.random_crop(image, size=[IMG_SIZE, IMG_SIZE, 3])
  image = tf.image.random_brightness(image, max_delta=0.5) # Random brightness
  image = tf.clip_by_value(image, 0, 1)
  return image, label

train = tf.keras.preprocessing.image_dataset_from_directory(
        train_data ,
        labels="inferred",
        label_mode="categorical",
        class_names=["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"],
        shuffle=True,
        seed=123,
        batch_size=batch_size,
        image_size=(32, 32),
    )

    for images, labels in train.take(1):
        for i in range(1):
            visualize(image,augment)
