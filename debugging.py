# Load pickled data
import pickle
import numpy as np

# TODO: Fill this in based on where you saved the training and testing data

training_file = 'traffic-signs-data/train.p'
validation_file = 'traffic-signs-data/valid.p'
testing_file = 'traffic-signs-data/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

# TODO: Number of training examples
n_train = X_train.shape[0]

# TODO: Number of validation examples
n_validation = X_valid.shape[0]

# TODO: Number of testing examples.
n_test = X_test.shape[0]

# TODO: What's the shape of an traffic sign image?
image_shape = X_train[0].shape
#print(X_train['labels'])
# TODO: How many unique classes/labels there are in the dataset.
n_classes = len(set(train['labels']))

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
import cv2
import cv2


### Preprocess the data here. It is required to normalize the data. Other preprocessing steps could include
### converting to grayscale, etc.
### Feel free to use as many code cells as needed.
def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140]).reshape((32,32,1))


def augment_brightness_camera_images(image):
    image1 = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    random_bright = .25 + np.random.uniform()

    image1[:, :, 2] = image1[:, :, 2] * random_bright
    image1 = cv2.cvtColor(image1, cv2.COLOR_HSV2RGB)
    return image1


def preprocess_image(img):
    img = augment_brightness_camera_images(img)
    img = rgb2gray(img)
    return img


X_train = np.array([preprocess_image(i) for i in X_train])
X_valid = np.array([preprocess_image(i) for i in X_valid])
X_test =  np.array([preprocess_image(i) for i in X_test])

y_train = np.expand_dims(y_train,axis=1)
y_valid = np.expand_dims(y_valid,axis=1)
y_test = np.expand_dims(y_test,axis=1)


datagen = ImageDataGenerator(
    samplewise_center=True,
    samplewise_std_normalization=True,
    rescale=1. / 255,
    zca_whitening=True
)

import keras
from keras import layers
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.optimizers import Adagrad
### Define your architecture here.
### Feel free to use as many code cells as needed.
model = keras.Sequential()

model.add(layers.Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=(32,32,1)))
model.add(layers.AveragePooling2D())
model.add(layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
model.add(layers.AveragePooling2D())
model.add(layers.Flatten())
model.add(layers.Dense(units=120, activation='relu'))
model.add(layers.Dense(units=84, activation='relu'))
model.add(layers.Dense(units=n_classes, activation = 'softmax'))
model.summary()

import datetime
### Train your model here.
### Calculate and report the accuracy on the training and validation set.
### Once a final model architecture is selected,
### the accuracy on the test set should be calculated and reported as well.
### Feel free to use as many code cells as needed.
# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)
new_dir = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

batch_size=32
epochs=1
load_checkpoint=False

#datagen.fit(X_train)

callbacks = [
    EarlyStopping(patience=10, verbose=1),
    ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1),
    ModelCheckpoint('./logs/' + new_dir + '/checkpoints/lenet.h5', verbose=1,
                    save_best_only=True, save_weights_only=True),
    TensorBoard(log_dir='./logs/' + new_dir + '/tensorboard')
]

#model = multi_gpu_model(model, gpus=2)
model.compile(optimizer=Adagrad(), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

if load_checkpoint:
    model.load_weights("")

results = model.fit_generator(
    datagen.flow(X_train, y_train, batch_size=batch_size),
    epochs=epochs,
    steps_per_epoch=(X_train.shape[0] // batch_size),
    validation_data=datagen.flow(X_train, y_train, batch_size=batch_size),
    validation_steps=(X_valid.shape[0] // batch_size),
    callbacks=callbacks
)