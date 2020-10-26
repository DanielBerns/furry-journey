# https://www.tensorflow.org/tutorials/images/classification

# Setup
import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds

print(tf.__version__)

# Download flowers_photos dataset
# flowers_photos/
#   daisy/
#   dandelion/
#   roses/
#   sunflowers/
#   tulips/

import pathlib
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file(origin=dataset_url, 
                                   fname='flower_photos', 
                                   untar=True)
data_dir = pathlib.Path(data_dir)

image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)

assert(image_count == 3670)

roses = list(data_dir.glob('roses/*'))
PIL.Image.open(str(roses[0]))

roses = list(data_dir.glob('roses/*'))
PIL.Image.open(str(roses[1]))

# Load using keras.preprocessing

# Create a dataset
# Define some parameters for the loader:

batch_size = 32
img_height = 180
img_width = 180

# It's good practice to use a validation split when developing your model. 
# We will use 80% of the images for training, and 20% for validation.

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

# You can find the class names in the class_names attribute on these datasets.
class_names = train_ds.class_names
print(class_names)

# Visualize the data

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")

# If you like, you can also manually iterate over the dataset and retrieve batches of images:
for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break

# Standardize the data
# The RGB channel values are in the [0, 255] range. This is not ideal for a neural network; 
# in general you should seek to make your input values small. 
# Here, we will standardize values to be in the [0, 1] by using a Rescaling layer.

from tensorflow.keras import layers

normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)

# There are two ways to use this layer. You can apply it to the dataset by calling map:

normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Notice the pixels values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image))

# Or, you can include the layer inside your model definition to simplify deployment. We will use the second approach here.

# Note: If you would like to scale pixel values to [-1,1] you can instead write Rescaling(1./127.5, offset=-1)
# Note: we previously resized images using the image_size argument of image_dataset_from_directory. 
# If you want to include the resizing logic in your model, you can use the Resizing layer instead.

# Configure the dataset for performance
# Let's make sure to use buffered prefetching so we can yield data from disk without having I/O become blocking. 
# These are two important methods you should use when loading data.

# .cache() keeps the images in memory after they're loaded off disk during the first epoch. 
# This will ensure the dataset does not become a bottleneck while training your model. 
# If your dataset is too large to fit into memory, you can also use this method to 
# create a performant on-disk cache.

# .prefetch() overlaps data preprocessing and model execution while training.

# Interested readers can learn more about both methods, as well as how to cache data to disk 
# in the data performance guide.
# https://www.tensorflow.org/guide/data_performance#prefetching

AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Train a model
# For completeness, we will show how to train a simple model using the datasets we just prepared. 
# This model has not been tuned in any way - the goal is to show you the mechanics using 
# the datasets you just created. To learn more about image classification, visit this tutorial.
# https://www.tensorflow.org/tutorials/images/classification

num_classes = 5

model = tf.keras.Sequential([
  layers.experimental.preprocessing.Rescaling(1./255),
  layers.Conv2D(32, 3, activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

model.compile(
  optimizer='adam',
  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])

model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=3
)

# You may notice the validation accuracy is low to the compared to the training accuracy, 
# indicating our model is overfitting. You can learn more about overfitting and how to reduce it in this tutorial.
# https://www.tensorflow.org/tutorials/keras/overfit_and_underfit



