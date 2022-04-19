"""
NAME - modeltf.py contains all models used to predict landmark ids for the google landmarks dataset

FILE - Users/ahldenbrough/documents/HCL/modeltf.py

FUNCTIONS:
    none

MODELS:
    VGG16, ResNet50, InceptionV3, EfficientNtB7

"""
import warnings
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Flatten, Dense
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import EfficientNetB7
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from data_loading import load_data
from data_cleaning import clean
from preprocessing import drop_faulty
from preprocessing import get_subset_of_landmark

warnings.filterwarnings("ignore")

#load in kaggle datasets and clean
DF_BOXES = pd.DataFrame()
DF_TRAIN, DF_TEST, DF_BOXES_1, DF_BOXES_2 = load_data()
print("loaded")
DF_TRAIN, DF_TEST, DF_BOXES = clean(DF_TRAIN, DF_TEST, DF_BOXES_1, DF_BOXES_2)
print("cleaned")

#read in csvs for each landmark

def make_dataset_30_landmarks():
    LANDMARK_IDS = DF_TRAIN.landmark_id.value_counts().head(30)
    ID_DICT = LANDMARK_IDS.to_dict()

    for key in ID_DICT:
       get_subset_of_landmark(200, DF_TRAIN, key)

    x1 = pd.read_csv('/Users/ahldenbrough/Documents/HCL/8429.csv')
    x2 = pd.read_csv('/Users/ahldenbrough/Documents/HCL/1553.csv')
    x3 = pd.read_csv('/Users/ahldenbrough/Documents/HCL/5376.csv')
    x4 = pd.read_csv('/Users/ahldenbrough/Documents/HCL/13526.csv')
    x5 = pd.read_csv('/Users/ahldenbrough/Documents/HCL/2743.csv')
    x6 = pd.read_csv('/Users/ahldenbrough/Documents/HCL/4352.csv')
    x7 = pd.read_csv('/Users/ahldenbrough/Documents/HCL/6651.csv')
    x8 = pd.read_csv('/Users/ahldenbrough/Documents/HCL/6696.csv')
    x9 = pd.read_csv('/Users/ahldenbrough/Documents/HCL/5554.csv')
    xa = pd.read_csv('/Users/ahldenbrough/Documents/HCL/2061.csv')
    xb = pd.read_csv('/Users/ahldenbrough/Documents/HCL/9779.csv')
    xc = pd.read_csv('/Users/ahldenbrough/Documents/HCL/6599.csv')
    xd = pd.read_csv('/Users/ahldenbrough/Documents/HCL/9633.csv')
    xe = pd.read_csv('/Users/ahldenbrough/Documents/HCL/6051.csv')
    xf = pd.read_csv('/Users/ahldenbrough/Documents/HCL/9029.csv')
    xg = pd.read_csv('/Users/ahldenbrough/Documents/HCL/428.csv')
    xh = pd.read_csv('/Users/ahldenbrough/Documents/HCL/12172.csv')
    xi = pd.read_csv('/Users/ahldenbrough/Documents/HCL/3924.csv')
    xj = pd.read_csv('/Users/ahldenbrough/Documents/HCL/2338.csv')
    xk = pd.read_csv('/Users/ahldenbrough/Documents/HCL/10045.csv')
    xl = pd.read_csv('/Users/ahldenbrough/Documents/HCL/12718.csv')
    xm = pd.read_csv('/Users/ahldenbrough/Documents/HCL/7092.csv')
    xn = pd.read_csv('/Users/ahldenbrough/Documents/HCL/10184.csv')
    xo = pd.read_csv('/Users/ahldenbrough/Documents/HCL/3804.csv')
    xp = pd.read_csv('/Users/ahldenbrough/Documents/HCL/2949.csv')
    xq = pd.read_csv('/Users/ahldenbrough/Documents/HCL/11784.csv')
    xr = pd.read_csv('/Users/ahldenbrough/Documents/HCL/4987.csv')
    xs = pd.read_csv('/Users/ahldenbrough/Documents/HCL/12220.csv')
    xt = pd.read_csv('/Users/ahldenbrough/Documents/HCL/8063.csv')
    xu = pd.read_csv('/Users/ahldenbrough/Documents/HCL/10900.csv')

    X = pd.concat([x1, x2, x3, x4, x5, x6, x7, x8, x9, xa, xb, xc, xd, xe, xf, xg, xh, xi, xj, xk, xl, xm, xn, xo, xp, xq, xr, xs, xt, xu])
    X["landmark_id"] = X["landmark_id"].astype(str)
    X = X[X['images'].notna()]
    X["faulty"] = X.apply(drop_faulty, axis='columns')
    X = X[X['faulty'].notna()]
    X.to_csv("first30.csv")

def make_dataset():
    LANDMARK_3283 = pd.read_csv('/Users/ahldenbrough/Documents/HCL/3283.csv')
    LANDMARK_7172 = pd.read_csv('/Users/ahldenbrough/Documents/HCL/7172.csv')
    LANDMARK_13653 = pd.read_csv('/Users/ahldenbrough/Documents/HCL/13653.csv')
    LANDMARK_2870 = pd.read_csv('/Users/ahldenbrough/Documents/HCL/2870.csv')
    LANDMARK_13170 = pd.read_csv('/Users/ahldenbrough/Documents/HCL/13170.csv')
    LANDMARK_6231 = pd.read_csv('/Users/ahldenbrough/Documents/HCL/6231.csv')
    LANDMARK_8169 = pd.read_csv('/Users/ahldenbrough/Documents/HCL/8169.csv')
    LANDMARK_7661 = pd.read_csv('/Users/ahldenbrough/Documents/HCL/7661.csv')
    LANDMARK_10033 = pd.read_csv('/Users/ahldenbrough/Documents/HCL/10033.csv')
    LANDMARK_1847 = pd.read_csv('/Users/ahldenbrough/Documents/HCL/1847.csv')

    #create dataframe from all image csvs
    X = pd.concat([LANDMARK_3283, LANDMARK_7172, LANDMARK_13653, \
        LANDMARK_2870, LANDMARK_13170, LANDMARK_6231, LANDMARK_8169, \
        LANDMARK_7661, LANDMARK_10033, LANDMARK_1847])
    X["landmark_id"] = X["landmark_id"].astype(str)
    X = X[X['images'].notna()]

    for df in [LANDMARK_3283, LANDMARK_7172, LANDMARK_13653, \
        LANDMARK_2870, LANDMARK_13170, LANDMARK_6231, LANDMARK_8169, \
        LANDMARK_7661, LANDMARK_10033, LANDMARK_1847]:
        print(df.url.head(5))
        print(df.landmark_id.head(5))

    #dropping all images that cause PIL.UnidentifiedImageError
    X["faulty"] = X.apply(drop_faulty, axis='columns')
    X = X[X['faulty'].notna()]
    X.to_csv("firstten.csv")

X = pd.read_csv("/Users/ahldenbrough/Documents/HCL/firstten.csv")
X["landmark_id"] = X["landmark_id"].astype(str)
#saving classes
Y = X["landmark_id"]

#split into train and validation sets
X_TRAIN, X_VAL, Y_TRAIN, Y_VAL = \
train_test_split(X, Y, test_size=0.2, random_state=2, stratify=X['landmark_id'])

X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = train_test_split(X, Y, test_size=0.4, random_state=2, stratify=X['landmark_id'])

X_TEST, X_VAL, Y_TEST, Y_VAL = train_test_split(X_TEST, Y_TEST, test_size=0.5, random_state=2, stratify=X_TEST['landmark_id'])

#create image generator and add data augmentation
TRAIN_DATAGEN = ImageDataGenerator(
    rescale=1./255.,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

#create training set
BATCH_SIZE = 32

TRAIN_GENERATOR = TRAIN_DATAGEN.flow_from_dataframe(
    dataframe=X_TRAIN,
    directory=None,
    x_col="images",
    y_col="landmark_id",
    subset="training",
    BATCH_SIZE=BATCH_SIZE,
    seed=42,
    shuffle=True,
    class_mode="categorical",
    target_size=(112, 112))

#create validation set
VAL_GENERATOR = TRAIN_DATAGEN.flow_from_dataframe(
    dataframe=X_VAL,
    directory=None,
    x_col="images",
    y_col="landmark_id",
    subset="training",
    BATCH_SIZE=BATCH_SIZE,
    seed=42,
    shuffle=True,
    class_mode="categorical",
    target_size=(112, 112))

#VGG16 model

#creating base model from vgg16 pretrained base
VGG16_BASE = VGG16(weights='imagenet',
                   include_top=False,
                   input_shape=(112, 112, 3),
                   pooling=None)

#freezing all base layers so we cannot train them
for layer in VGG16_BASE.layers:
    layer.trainable = False

print(VGG16_BASE.output_shape)

#adding head to pretrained base model
ADD_LAYER = VGG16_BASE.output
ADD_LAYER = Flatten()(ADD_LAYER) #makes previous layers output 1 dimensional
ADD_LAYER = Dense(64, activation='relu')(ADD_LAYER) #all noise reduced to zero
ADD_LAYER = Dense(10, activation='softmax')(ADD_LAYER) #prediction layer

VGG16_FINAL = Model(inputs=VGG16_BASE.input, outputs=ADD_LAYER)
print(VGG16_FINAL.summary())

#compile model with adam optimizer
VGG16_FINAL.compile(loss='categorical_crossentropy',
                    optimizer=Adam(),
                    metrics=['accuracy'])

print("VGG16_FINAL compiles")

VGG16_HISTORY = VGG16_FINAL.fit(TRAIN_GENERATOR,
                                steps_per_epoch=X_TRAIN.shape[0]//BATCH_SIZE,
                                epochs=100,
                                validation_data=VAL_GENERATOR,
                                validation_steps=X_VAL.shape[0]//BATCH_SIZE)

VGG16_FINAL.save("VGG16_FINAL.h5")

#plot accuracy and loss of VGG16 model
#  "Accuracy"
plt.plot(VGG16_HISTORY.history['accuracy'])
plt.plot(VGG16_HISTORY.history['val_accuracy'])
plt.title('model accuracy VGG16')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'])
plt.show()

# "Loss"
plt.plot(VGG16_HISTORY.history['loss'])
plt.plot(VGG16_HISTORY.history['val_loss'])
plt.title('model loss InceptionV3')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'])
plt.show()


#InceptionV3 model

#load in pretrained base
INCEPTIONV3_BASE = InceptionV3(input_shape=(112, 112, 3), include_top=False, weights='imagenet')

#freese base layers so we can't train on them
for layer in INCEPTIONV3_BASE.layers:
    layer.trainable = False

#add our own head
ADD_LAYER = layers.Flatten()(INCEPTIONV3_BASE.output) #turn base model output 1D
ADD_LAYER = layers.Dense(1024, activation='relu')(ADD_LAYER) #remove noise
ADD_LAYER = layers.Dropout(0.2)(ADD_LAYER)#prevents overfitting by setting 20 percent of inputs to 0
ADD_LAYER = layers.Dense(10, activation='softmax')(ADD_LAYER) #prediction layer

INCEPTIONV3_FINAL = tf.keras.models.Model(INCEPTIONV3_BASE.input, ADD_LAYER)

#complile model with Adam optimizer
INCEPTIONV3_FINAL.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
print(INCEPTIONV3_FINAL.summary())

#train model
INV3_HISTORY = INCEPTIONV3_FINAL.fit_generator(TRAIN_GENERATOR,
                                               validation_data=VAL_GENERATOR,
                                               steps_per_epoch=X_TRAIN.shape[0]//BATCH_SIZE,
                                               epochs=10)

#plot accuracy and loss

#  "Accuracy"
plt.plot(INV3_HISTORY.history['accuracy'])
plt.plot(INV3_HISTORY.history['val_accuracy'])
plt.title('model accuracy InceptionV3')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'])
plt.show()

# "Loss"
plt.plot(INV3_HISTORY.history['loss'])
plt.plot(INV3_HISTORY.history['val_loss'])
plt.title('model loss InceptionV3')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'])
plt.show()


#ResNet50 model

#load in ResNet50 model
RESNET50_BASE = ResNet50(input_shape=(112, 112, 3), include_top=False, weights='imagenet')

#freeze base layers
for layer in RESNET50_BASE.layers:
    layer.trainable = False

#add head
ADD_LAYER = layers.Flatten()(RESNET50_BASE.output) #turn base model output 1D
ADD_LAYER = layers.Dense(1024, activation='relu')(ADD_LAYER) #remove noise
ADD_LAYER = layers.Dropout(0.2)(ADD_LAYER)#prevents overfitting by setting 20 percent of inputs to 0
ADD_LAYER = layers.Dense(10, activation='softmax')(ADD_LAYER) #prediction layer

RESNET50_FINAL = tf.keras.models.Model(RESNET50_BASE.input, x)

#compile model with SGD optimizer
RESNET50_FINAL.compile(optimizer=tf.keras.optimizers.SGD(lr=0.0001), \
    loss='categorical_crossentropy', metrics=['accuracy'])

print(RESNET50_FINAL.summary())

#train model
RS50_HISTORY = RESNET50_FINAL.fit(TRAIN_GENERATOR, validation_data=VAL_GENERATOR, \
    steps_per_epoch=X_TRAIN.shape[0]//BATCH_SIZE, epochs=5)

RESNET50_FINAL.save("RESNET50_FINAL.h5")
#plot accuracy and loss
#  "Accuracy"
plt.plot(RS50_HISTORY.history['accuracy'])
plt.plot(RS50_HISTORY.history['val_accuracy'])
plt.title('model accuracy ResNet50')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'])
plt.show()

# "Loss"
plt.plot(RS50_HISTORY.history['loss'])
plt.plot(RS50_HISTORY.history['val_loss'])
plt.title('model loss ResNet50')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'])
plt.show()

#EfficientNetB7 model

#load in EfficientNetB7 model
EFFICIENTNETB7_BASE = EfficientNetB7(input_shape=(112, 112, 3), include_top=False, weights='imagenet')

#freeze base layers
for layer in EFFICIENTNETB7_BASE.layers:
    layer.trainable = False

#add head
ADD_LAYER = layers.Flatten()(EFFICIENTNETB7_BASE.output) #turn base model output 1D
ADD_LAYER = layers.Dense(1024, activation='relu')(ADD_LAYER) #remove noise
ADD_LAYER = layers.Dropout(0.2)(ADD_LAYER)#prevents overfitting by setting 20 percent of inputs to 0
ADD_LAYER = layers.Dense(10, activation='softmax')(ADD_LAYER) #prediction layer

EFFICIENTNETB7_FINAL = tf.keras.models.Model(EFFICIENTNETB7_BASE.input, x)

#compile model with SGD optimizer
EFFICIENTNETB7_FINAL.compile(optimizer=tf.keras.optimizers.SGD(lr=0.0001), \
    loss='categorical_crossentropy', metrics=['accuracy'])

#train model
EFFICIENTNETB7_HISTORY = EFFICIENTNETB7_FINAL.fit(TRAIN_GENERATOR, validation_data=VAL_GENERATOR, \
    steps_per_epoch=X_TRAIN.shape[0]//BATCH_SIZE, epochs=5)

EFFICIENTNETB7_FINAL.save("EFFICIENTNETB7_FINAL.h5")

#plot accuracy and loss
#  "Accuracy"
plt.plot(EFFICIENTNETB7_HISTORY.history['accuracy'])
plt.plot(EFFICIENTNETB7_HISTORY.history['val_accuracy'])
plt.title('model accuracy EfficientNetB7')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'])
plt.show()

# "Loss"
plt.plot(EFFICIENTNETB7_HISTORY.history['loss'])
plt.plot(EFFICIENTNETB7_HISTORY.history['val_loss'])
plt.title('model loss EfficientNetB7')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'])
plt.show()
