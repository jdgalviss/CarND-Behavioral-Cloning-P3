import os
import csv
import numpy as np
import shutil
from time import time

#============================Load Data=======================
#Remove flipped files folder
path_prefix = './my_data/'
try:
    shutil.rmtree(path_prefix + 'flipped/IMG/')
except:
    print("no folder for flipped images was found")

#Create folder to store flipped images
try: 
    os.mkdir(path_prefix + 'flipped/IMG/')  
except OSError:  
    print ("Creation of the directory %s failed" % path_prefix + 'flipped/IMG/')

#Load data from csv file
samples = []
with open(path_prefix + 'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        line[0] = line[0].strip().replace('/home/kiwibot/jdgalviss/Self_driving_cars_tools/15Project_BehavioralCloning/CarND-Behavioral-Cloning-P3/my_data/','')
        line[1] = line[1].strip().replace('/home/kiwibot/jdgalviss/Self_driving_cars_tools/15Project_BehavioralCloning/CarND-Behavioral-Cloning-P3/my_data/','')
        line[2] = line[2].strip().replace('/home/kiwibot/jdgalviss/Self_driving_cars_tools/15Project_BehavioralCloning/CarND-Behavioral-Cloning-P3/my_data/','')
        samples.append(line)
samples.pop(0)
print('Data Loaded')
print("number of samples: ", np.array(samples).shape)

#============================Generate flipped images=======================
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

#Test flipping
img=mpimg.imread(path_prefix + samples[0][0])
image_flipped = np.fliplr(img)

#Flip and save images to folder
for sample in samples:
    img=mpimg.imread(path_prefix + sample[0])
    image_flipped = np.fliplr(img)
    cv2.imwrite(path_prefix + 'flipped/' + sample[0], cv2.cvtColor(image_flipped, cv2.COLOR_BGR2RGB))
print('Flipped Images saved')

#============================Divide data in train and test samples======================
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)
print('Data Splitted')
print("training samples: ", np.array(train_samples).shape)
print("validation samples: ", np.array(validation_samples).shape)

#============================Augment Data by using left and right images=======================
steering_correction_factor = 0.15
img_paths_train = []
angles_train = []

for sample in train_samples:
    steering_center = float(sample[3])
    
    # create adjusted steering measurements for the side camera images
    steering_left = steering_center + steering_correction_factor
    steering_right = steering_center - steering_correction_factor
    
    # define center, right and left paths
    img_paths_train.append(path_prefix + sample[0]) # center
    img_paths_train.append(path_prefix + sample[1]) # left
    img_paths_train.append(path_prefix + sample[2]) # right
    angles_train.append(steering_center)
    angles_train.append(steering_left)
    angles_train.append(steering_right)
print('Data Augmented: Left and Right')   
print("number of samples: ", np.array(img_paths_train).shape)
print("number of samples: ", np.array(angles_train).shape)

#============================Augment Data using flipped images=======================
for sample in train_samples:
    img_paths_train.append(path_prefix+'flipped/' + sample[0].strip())
    angles_train.append(-float(sample[3]))
    #print(sample[1])
print('Data Augmented flipped')
print("number of samples: ", np.array(img_paths_train).shape)
print("number of samples: ", np.array(angles_train).shape)
print(img_paths_train[0])

#=================Put Validation data into img_path and angles lists==================
# Validation imgs - create vectors
img_paths_validation = []
angles_validation = []
for sample in validation_samples:
    img_paths_validation.append(path_prefix + sample[0].strip())
    angles_validation.append(float(sample[3]))
print('Validation Data')    
print("number of samples: ", np.array(img_paths_validation).shape)
print("number of samples: ", np.array(angles_validation).shape)

#======================Generator for image processing=================
import sklearn
from sklearn.utils import shuffle
def generator(images_path, angles, batch_size=32):
    num_samples = len(images_path)
    while 1: # Loop forever so the generator never terminates
        images_path, angles = shuffle(images_path, angles)
        for offset in range(0, num_samples, batch_size):
            batch_images = images_path[offset:offset+batch_size]
            batch_angles = angles[offset:offset+batch_size]
            
            images = []
            angles_batch = []
            for batch_image, angle in zip(batch_images, batch_angles):
                center_image = mpimg.imread(batch_image)
                center_angle = float(angle)
                images.append(center_image)
                angles_batch.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles_batch)
            yield sklearn.utils.shuffle(X_train, y_train)
            
# Set our batch size
batch_size=32

# compile and train the model using the generator function
train_generator = generator(img_paths_train, angles_train, batch_size=batch_size)
validation_generator = generator(img_paths_validation, angles_validation, batch_size=batch_size)

#=======================Model architecture===============
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import BatchNormalization, Lambda, Cropping2D, MaxPooling2D, Conv2D, Dense, Activation, Flatten, Dropout
from tensorflow.keras.callbacks import TensorBoard

ch, row, col = 3, 75, 320  # Trimmed image format
keep_prob = 0.5
def pilotNet(train = True, keep_prob = 0.5):
    model = Sequential()
    #------------Preprocess incoming data------------
    # Crop Image
    model.add(Cropping2D(cropping=((65,20), (0,0)), input_shape=(160,320,3)))
    #centered around zero with small standard deviation 
    model.add(Lambda(lambda x: x/127.5 - 1.,
            input_shape=(ch, row, col),
            output_shape=(ch, row, col)))
    #--------------Model Architecture: PilotNet-------------------
    model.add(Conv2D(32, (3, 3), input_shape=(32, 32, 3)))

    model.add(Conv2D(24, (5, 5), strides = 2))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(rate = keep_prob))
    
    model.add(Conv2D(36, (5, 5), strides = 2))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(rate = keep_prob))

    model.add(Conv2D(48, (5, 5), strides = 2))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(rate = keep_prob))

    model.add(Conv2D(64, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(rate = keep_prob))

    model.add(Conv2D(64, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(rate = keep_prob))
    
    model.add(Flatten())
    model.add(Dense(200))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(rate = keep_prob))
    model.add(Dense(1))
    return model
model = pilotNet()

#============================Train model=======================
from math import ceil
model.compile(loss='mse', optimizer='adam')
tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
history_object = model.fit_generator(train_generator,
            steps_per_epoch=ceil(len(img_paths_train)/batch_size),
            validation_data=validation_generator,
            validation_steps=ceil(len(img_paths_validation)/batch_size),
            epochs=4, verbose=2, callbacks=[tensorboard])

#============================Save model=======================
model.save('model_mine3.h5')
print('model saved')


#============================Plot=======================
### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()