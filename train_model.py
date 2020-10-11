# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pyimagesearch.preprocessing import ImageToArrayPreprocessor
from pyimagesearch.preprocessing import AspectAwarePreprocessor
from pyimagesearch.datasets import SimpleDatasetLoader
from pyimagesearch.nn.conv import AlexNet
from pyimagesearch.nn.conv import MiniVGGNet
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import tensorflow as tf
from sklearn.utils import shuffle


ap=argparse.ArgumentParser()
ap.add_argument("-d","--dataset",required=True,
                help="path to input dataset")
args=vars(ap.parse_args())

# grab the list of images that we’ll be describing, then extract
# the class label names from the image paths
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
classNames = [pt.split(os.path.sep)[-2] for pt in imagePaths]
classNames = [str(x) for x in np.unique(classNames)]
print(len(classNames))

# initialize the image preprocessors
aap = AspectAwarePreprocessor(128, 128)
iap = ImageToArrayPreprocessor()

# load the dataset from disk then scale the raw pixel intensities
 # to the range [0, 1]
sdl = SimpleDatasetLoader(preprocessors=[aap, iap])
(data, labels) = sdl.load(imagePaths, verbose=500)
data = data.astype("float") / 255.0
data,labels=shuffle(data,labels)

(trainX, testX, trainY, testY)=train_test_split(data,labels,test_size=0.25,random_state=42,shuffle=True)
print(trainY[:5])
trainY=LabelBinarizer().fit_transform(trainY)
testY=LabelBinarizer().fit_transform(testY)
print(trainY[:5])

# #construct the image generator for data augmentation
# aug = ImageDataGenerator(rotation_range=10, width_shift_range=0.1,
# height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
# horizontal_flip=False, fill_mode="nearest")


# # initialize the optimizer and model
# print("[INFO] compiling model...")
# opt = SGD(lr=0.05)
# model = MiniVGGNet.build(width=64, height=64, depth=3,
# classes=len(classNames))

# initialize the optimizer and model
print("[INFO] compiling model...")
opt = SGD(lr=0.05)
model = AlexNet.build(width=128, height=128, depth=3,
classes=len(classNames))


model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=["accuracy"])

# # Construct the callback to save only the 'best' model to disk based on the validation loss
filepath = "saved-model-{epoch:02d}-{val_accuracy:.2f}.h5"
#checkpoint = ModelCheckpoint(filepath, monitor="val_loss", mode="min", verbose=1,save_freq=2)
checkpoint = ModelCheckpoint(filepath,verbose=1, period=1,monitor='val_accuracy')
callbacks = [checkpoint]

# train the network
print("[INFO] training network...")
# H = model.fit_generator(aug.flow(trainX, trainY, batch_size=32),validation_data=(testX, testY), steps_per_epoch=len(trainX) // 32,
# epochs=3,verbose=1)
H= model.fit(trainX, trainY, validation_data=(testX, testY),batch_size=32, epochs=5, verbose=1,callbacks=callbacks)

#tf.keras.models.save_model(model,'./model')
#tf.keras.models.save_model(model,'model.h5')
model.save('model.h5')
print("model saved")

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
predictions.argmax(axis=1), target_names=classNames))

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 5), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 5), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 5), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, 5), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()
