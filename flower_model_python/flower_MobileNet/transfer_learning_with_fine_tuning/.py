# importing all the libraries required for building my model
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,MaxPool2D,Flatten
from tensorflow.keras.preprocessing import image

# from keras applications i am using MobileNet to build my model
mobile_net=tf.keras.applications.MobileNet(input_shape=(128,128,3))

mobile_net.summary()
# Total params: 4,253,864
# Trainable params: 4,231,976
# Non-trainable params: 21,888
# from the summary , we can see the number of trainable params

# we will see the base model, which shows all the convolutional layers
base_model=tf.keras.applications.MobileNet(input_shape=(128,128,3),include_top=False) 

len(base_model.layers) # displays no of layers in base model

# freezinf most of the convolutional layers
for layer in base_model.layers[:71]:
    layer.trainable=False
# just checking of the layers have been changed to non-trainable ones
for layer in base_model.layers:
    print(layer.name,layer.trainable)

# model architecture
transfer_model_t=Sequential()
transfer_model_t.add(base_model)
# transfer_model.add(Flatten()) # increases weights and time consuming
transfer_model_t.add(GlobalAveragePooling2D())
transfer_model_t.add(Dense(units=512,activation='relu'))
transfer_model_t.add(Dropout(.2))
transfer_model_t.add(Dense(units=64,activation='relu'))
transfer_model_t.add(Dropout(.1))
transfer_model_t.add(Dense(units=5,activation='softmax'))

# preparing data
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen=ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input,
                  rotation_range=45,
                  width_shift_range=0.2,
                  height_shift_range=0.2,
                  shear_range=0.2, # shrink
                  zoom_range=0.2,
                  horizontal_flip=True,# becareful with flip dont flip numbers
                  fill_mode='reflect'
                   
                  )
test_datagen=ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input)

# train and test data

train_set=train_datagen.flow_from_directory("C:\\Users\\warda\\flower_photos\\Training",
                                 target_size=(128,128),
                                 batch_size=128,
                                 class_mode='categorical') # for binary give class_mode=binary

test_set=test_datagen.flow_from_directory("C:\\Users\\warda\\flower_photos\\Testing",
                                 target_size=(128,128),
                                 batch_size=128,
                                 class_mode='categorical')# images are taken in random way not sequential

# compile the data
transfer_model_t.compile(optimizer=ada(learning_rate=.0001),loss='categorical_crossentropy',metrics=['accuracy'])

# creating model checkpoint
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint

keras_callback=[EarlyStopping(monitor='val_loss',mode='min',patience=5,min_delta=.01),
ModelCheckpoint('C:\\Users\\warda\\Flowerbest_transfertt',monitor='val_loss',save_best_only=True)] 
transfer_model_t.fit(train_set,
                      steps_per_epoch=2736//128, # how many times in one epoch weights need to be updated?
                      epochs=5, # try 20
                      validation_data=test_set,
                      validation_steps=934//128,
                  callbacks=keras_callback)

# in order to generate a confusion matrix
test_datagen = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input)

test_set=test_datagen.flow_from_directory('C:\\Users\\warda\\flower_photos\\Testing',
                                 target_size=(128,128),
                                 shuffle=False,
                                 batch_size=1,     
                                 class_mode='categorical') # binary for covid-noncovid(2 classes)

filenames = test_set.filenames
nb_samples = len(filenames)

predict = transfer_model_t.predict_generator(test_set,steps = nb_samples)

ypred=predict.argmax(axis=1)
yact=test_set.classes
from sklearn.metrics import confusion_matrix
import seaborn as sb
plt.figure(figsize=(10,6))
sb.heatmap(confusion_matrix(yact,ypred),annot=True,fmt='.0f')
                         
