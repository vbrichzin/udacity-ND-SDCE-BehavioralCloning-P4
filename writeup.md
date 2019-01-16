# **Behavioral Cloning** 

## Writeup

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/model.png                   "Model Visualization"
[image2]: ./examples/right-bias.jpg              "Recovery Image"
[image3]: ./examples/corrected-right-bias.jpg    "Recovery Image"
[image4]: ./examples/left-bias.jpg               "Recovery Image"
[image5]: ./examples/corrected-left-bias.jpg     "Recovery Image"
[image6]: ./examples/image.jpg                   "Normal Image"
[image7]: ./examples/image-flipped.jpg           "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* `model.py` containing the script to create and train the model
* `drive.py` for driving the car in autonomous mode (no changes to original file)
* `model.h5` containing a trained convolution neural network 
* `writeup.md` summarizing the results (this document)
* `run1.mp4` video of a little more than a lap of autonomous driving on track 1

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

I did so and recorded a lap of driving on track 1 in the file `run1.mp4`.

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of 5 convolution neural network layers with 5x5 resp. 3x3 filter sizes and depths between 24 and 64.
These convolutional layers are followed by 4 flat layers.
This architecture is the one used by the NVIDIA team and explained in the instructor video.

The model includes RELU layers to introduce nonlinearity in the convolution steps as the activation function, and the data is before normalized and centered and the images are cropped that the top and bottom part are discarded.


#### 2. Attempts to reduce overfitting in the model

The model has commented out dropout layers that could be used in order to reduce overfitting.
Since I don't think I had an issue with overfitting I didn't use them in the end.

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track (track 1).

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road and additional training data for the difficult tighter curves.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use the NVIDIA model as this seemed to provide good results in the instructor video.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. From the start the training and validation sets provided low mean square errors.


The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track especially the tighter curves. To improve the driving behavior in these cases, I recorded additional training data especially for the curves and especially at the entrance of the curve to have a bigger steering angle.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road, see also the `run1.mp4`video.

#### 2. Final Model Architecture

The final model architecture consisted of the 5 convolution neural network layers with subsequent dense layers.

Here is a visualization of the model (created with the `plot_model` function from `keras`:

![alt text][image1]


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded a little over two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image6]

To augment the data and generalize the model I flipped the images and steering angle measurements accordingly. Here is an example of the flipped image data to the above center lane driving:

![alt text][image7]

As it was suggested I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to correct from being off-center to the middle of the road. 
These images show what a recovery looks like starting from the right and coming back to the center:

![alt text][image2]
![alt text][image3]

Similarly for a recovery starting from the left and ending in the center:
![alt text][image4]
![alt text][image5]

After the collection process, I had X number of data points. I then preprocessed this data by normalizing and centering it and cropping the top part (with the trees and sky) and the bottom part (with the hood of the car).


I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 2 as evidenced by observing the training mean square error that already came to fluctuation in the second epoch. 

I used an adam optimizer so that manually training the learning rate wasn't necessary.
