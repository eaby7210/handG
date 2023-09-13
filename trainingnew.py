import tensorflow as tf
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)


training_set = train_datagen.flow_from_directory(r'C:\Users\miche\PycharmProjects\os_controlling\dataSet\training_data',
                                                 target_size = (128, 128),
                                                 batch_size = 10,
                                                 color_mode = 'grayscale',
                                                 class_mode = 'categorical')


test_set = test_datagen.flow_from_directory(r'C:\Users\miche\PycharmProjects\os_controlling\dataSet\training_data',
                                            target_size = (128, 128),
                                            batch_size = 10,
                                            color_mode = 'grayscale',
                                            class_mode = 'categorical')

classifier = tf.keras.models.Sequential()

classifier.add(tf.keras.layers.Conv2D(filters=32,
                                     kernel_size=3,
                                     padding="same",
                                     activation="relu",
                                     input_shape=[128, 128, 1]))


classifier.add(tf.keras.layers.MaxPool2D(pool_size=2,
                                         strides=2,
                                         padding='valid'))

classifier.add(tf.keras.layers.Conv2D(filters=32,
                                      kernel_size=3,
                                      padding="same",
                                      activation="relu"))

classifier.add(tf.keras.layers.MaxPool2D(pool_size=2,
                                         strides=2,
                                         padding='valid'))

classifier.add(tf.keras.layers.Flatten())


classifier.add(tf.keras.layers.Dense(units=128,
                                     activation='relu'))
classifier.add(tf.keras.layers.Dropout(0.40))
classifier.add(tf.keras.layers.Dense(units=96, activation='relu'))
classifier.add(tf.keras.layers.Dropout(0.40))
classifier.add(tf.keras.layers.Dense(units=64, activation='relu'))
classifier.add(tf.keras.layers.Dense(units=3, activation='softmax')) # softmax for more than 2


classifier.compile(optimizer = 'adam',
                   loss = 'categorical_crossentropy',
                   metrics = ['accuracy'])


classifier.summary()


classifier.fit(training_set,
                  epochs = 10,
                  validation_data = test_set)

model_json = classifier.to_json()
with open("model_new.json", "w") as json_file:
    json_file.write(model_json)
print('Model Saved')
classifier.save('model_new.h5')
# print('Weights saved')

