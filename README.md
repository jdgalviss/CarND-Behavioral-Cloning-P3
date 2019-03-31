# **Behavioral Cloning** 

### In this project, a behavioral cloning approach to self driving on udacity's simulator is presented

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road


[//]: # (Image References)

[image1]: ./output_imgs/uncropped.jpg "uncropped image"
[image2]: ./output_imgs/cropped.jpg "cropped"
[image3]: ./output_imgs/unflipped.jpg "unflipped"
[image4]: ./output_imgs/flipped.jpg "flipped"
[image5]: ./output_imgs/architecture.png "architecture"
[image6]: ./output_imgs/epoch_loss.png "epoch_loss"
[image7]: ./output_imgs/epoch_val_loss.png "epoch_val_loss"
[image8]: ./output_imgs/Figure_3.png "loss"
[image9]: ./output_imgs/loss_batches.png "loss_batches"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* my_model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py my_model.h5
```

The model.py file contains the code for loading data, preprocessing it, defininf, training and saving the convolution neural network. The file shows the pipeline used for training and validating the model, and it contains comments to explain how the code works.

### Data Capture
Using [udacity's simulator](https://github.com/udacity/self-driving-car-sim) data on both tracks is captured taking into account:

* Data of center-line driving is captured, making several laps on the circuit
* Data is captured clockwise and counter-clockwise in order to avoid overfitting.
* Data is captured on both sample tracks.
* Data of recovery behavior (i.e. when the robot is getting out of the track and comes back to the center) is captured.

### Image preprocessing:
## 1. Data Augmentation
In order to train over more data and have a model that generalizes better and avoids overfitting, data is augmented using two processes:
* Flip Images: Images get flipped and a negative value of the steering angle is taken.
* 
![alt text][image3] ![alt text][image4]

```python
image_flipped = np.fliplr(img)
```

* The simulator not only captures data from a centered camera, but also from cameras located to the sides. These left and right images are given a steering value with an offset respect to the center image's steering value.
```python
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
```

## 2. Image cropping
In order to remove information that is not useful to the model, i.e. the hood of the car, the sky, trees and hills, image are cropped:
![alt text][image1] ![alt text][image2]

In order to perform image cropping, a cropping layer is added to the model:
```python
model.add(Cropping2D(cropping=((65,20), (0,0)), input_shape=(160,320,3)))
```

## 3. Normalization
The data is normalized in the model using a Keras lambda layer. 
```python
model.add(Lambda(lambda x: x/127.5 - 1.,
            input_shape=(ch, row, col),
            output_shape=(ch, row, col)))
```

### Model Architecture and Training Strategy

After experimenting with a couple model architectures, a model architecture is implemented based on the work of [Nvidia's ent=d to end learning for self-driving cars](https://arxiv.org/pdf/1604.07316.pdf). The implementation can be found in model.py

```python
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
```


Here is a visualization of the architecture

![alt text][image5]

The model contains dropout layers and batch normalization layers in order to reduce overfitting.

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 188).

```python
model.compile(loss='mse', optimizer='adam')
```

### Results
The model was trained for 4 epochs in batches of 32 images. Data was divided in training(80%) and validation(20%):

```python
train_samples, validation_samples = train_test_split(samples, test_size=0.2)
```

 Getting the following result in training and validation loss:

![alt text][image8]
![alt text][image9]

In the following links you can find the videos corresponding to the performance of the model on both tracks (challenge was completed after recording a little of data on the second track):
* [Project](https://youtu.be/0f6uDj9qVY8)
* [Challenge](https://arxiv.org/pdf/1604.07316.pdf)