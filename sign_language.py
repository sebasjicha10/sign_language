import csv
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from os import getcwd


def get_data(filename):
    with open(filename) as training_file:
        raw_labels = []
        raw_images = []

        # Delete spaces, create lists, extract images and labels
        next(training_file)
        for row in training_file:

            deleted_space = row.strip()
            string_to_list_row = deleted_space.split(",")
            label = string_to_list_row[0]

            raw_image = string_to_list_row[1:785]
            numpy_image = np.array(raw_image).astype(float)
            final_image = np.array(np.array_split(numpy_image, 28))

            raw_labels.append(label)
            raw_images.append(final_image)

        # Turn labels into Numpy
        labels = np.array(raw_labels).astype(int)

        # Turn images into Numpy
        images = np.array(raw_images)

    return images, labels

path_sign_mnist_train = f"{getcwd()}/../tmp2/sign_mnist_train.csv"
path_sign_mnist_test = f"{getcwd}/../tmp2/sign_mnist_test.csv"
training_images, training_labels = get_data(path_sign_mnist_train)
testing_images, testing_labels = get_data(path_sign_mnist_test)

print(training_images.shape)
print(training_labels.shape)
print(testing_images.shape)
print(testing_labels.shape)

training_images = np.expand_dims(training_images, axis=4)
testing_images = np.expand_dims(testing_images, axis=4)

train_datagen = ImageDataGenerator(
    rescale = 1/255,
    rotation_range = 40,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True,
    fill_mode = "nearest"
)

validation_datagen = ImageDataGenerator(rescale=1/255)


train_generator = train_datagen.flow(training_images, training_labels)
validation_generator = validation_datagen.flow(testing_images, testing_labels)

model = tf.keras.models.Sequential([
    # First Convolution
    tf.keras.layers.Conv2D(64, (3,3), activation="relu", input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2,2),
    # Second Convolution
    tf.keras.layers.Conv2D(128, (3,3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2,2),
    # Flatten
    tf.keras.layers.Flatten(),
    # Hidden + Output Layers
    tf.keras.layers.Dense(512, activation="relu"),
    tf.keras.layers.Dense(26, activation="softmax")
])

model.summary()

model.compile(
    loss="categorical_crossentropy",
    optimizer="rmsprop",
    metrics=["accuracy"]
)

history = model.fit(
    train_generator,
    epochs=2,
    validation_data=validation_generator,
    verbose=2,
)

%matplotlib inline
import matplotlib.pyplot as plt
acc = history.history["accuracy"]
val_acc= history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]

epochs = range(len(acc))

plt.plot(epochs, acc, "r", label="Training accuracy")
plt.plot(epochs, val_acc, "b", label="Validation accuracy")
plt.title("Training and validation accuray")
plt.legend()
plt.figure()

plt.plot(epochs, loss, "r", label="Trainin accuracy")
plt.plot(epochs, val_acc, "b", label="Validation accuracy")
plt.title("Training and validation loss")
plt.legend()

plt.show()