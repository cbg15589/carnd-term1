# **Behavioral Cloning** 
---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./examples/nvidia_model.JPG "NVIDIA model"
[image3]: ./examples/e.jpg "Centre driving"
[image4]: ./examples/9_cameras_sample.JPG "9 Cameras"
[image5]: ./examples/radio.JPG "Radio variation"
[image6]: ./examples/angle_offset.JPG "angle offsets"
[image7]: ./examples/offset_calculation.JPG "Offsets Calculation"
[image8]: ./examples/sample_images.png "Sample Images"
[image9]: ./examples/histogram_orig.png "Original histogram"
[image10]: ./examples/histogram_augmented.png "Augmented histogram"
[image11]: ./examples/13_cameras.JPG "Original histogram"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* model.ipynb notebook based on model.py
* drive_track1.py for driving the car in autonomous mode on track1
* drive_track2.py for driving the car in autonomous mode on track2
* model.10-0.01.h5 containing a trained convolution neural network 
* writeup_report.md or summarizing the results
* track1_video.mp4 containing a video of the car around the first track
* track2_video.mp4 containing a video of the car around the second track
* Third person video of track 1 (https://www.youtube.com/watch?v=WMAliBD9yyE)
* Third person video of track 2 (https://www.youtube.com/watch?v=bchC4b4chMw)

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around first track by executing :
```sh
python drive_track1.py model.10-0.01.h5
```

For second track (slower set speed): 
```sh
python drive_track2.py model.10-0.01.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works. Additionally it contains the code for the data exploration and examples of the applied image augmentation.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

As model architecture I used the [NVIDIA](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf "NVIDIA") model, which has been proven capable of driving a car autonomously on real environments.

Taking into account that the simulator is simpler, this should be a safe choice. For the implementation I used Keras, which makes the model code very simple and the use of tools such as data generators very easy.

The model consists in 5 convolutional layers followed by three fully connected layers. 2x2 strides are used for the 5x5 kernel and non-stride for the 3x3 kernel. The model includes RELU layers to introduce nonlinearity. It is preceded by a lambda layer which normalises the images and a cropping layer to focus in the area of interest. The model also includes two dropout layers between the three fully connected layers. These help to avoid overfitting.

#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting. The data split between training and validation was 80-20 respectively. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

As explained before, also two dropout layers were included be between the three fully connected layers, these helped to avoid overfitting. As a result both validation and training loss were very similar.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually. 

I didn't need to tune the number of epochs, using Keras "early stop" and "save checkpoint" callbacks, I saved the model every time it improved validation loss, and also if the loss didn't decrease for a couple of epochs, training would stop automatically.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used one lap on each track, driving on the centre of the road.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to find a model which could be able to generalise enough to be able to drive on both tracks.

My first step was to use the simplest model I could make, just a fully connected layer. This way, although the driving was quite bad, I was sure that all the wiring between the model and the data was working.

Later I needed to add a more complex model, after some research on both literature and experiences from other students, I decided to go with the model proved by NVIDIA. This model has been proven capable to drive in a real environment in very different situations, so with the correct training data, it should be able to cope with both tracks in the simulator.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that although the mean squared error was low in both training and validation sets, the vehicle slowly drifted out of the track, so I needed better training data (for detailed explanation see next section).

After improving the training data, I started seeing signs of overfitting in the model, as the validation loss was more than double of the training loss at the end of the training. Adding two dropout layers between the three fully connected layers solved this issue.

Then I trained the model on the first track again, and the vehicle was able to drive autonomously around the track without leaving the road.

Finally I trained the model on both tracks, the vehicle can drive on both tracks, although the behaviour in the first track was a bit worse, showing steering angle oscillations in some parts of the track.

#### 2. Final Model Architecture

As explained before, I used the [NVIDIA](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf "NVIDIA") model. Below you can see a visualization of the model:

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behaviour, I first recorded one lap on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image3]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to go back to the center of the lane. 

Getting good training data was a challenge, controlling the vehicle with the keyboard or the mouse made it very difficult. Ideally I would have used a real steering wheel, but as I didn't have access to one, I used the mobile phones gyroscope as controller. Still, I wasn't getting the desired data for the recoveries, so I tried to find some alternatives. 

First I tried augmenting the data, where posts from other students such as this one from [Vivek Yadak](https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9 "Vivek Yadak"), helped a lot. To mimic the recoveries I tried different image transformations such image shifting and more complex transformations.

NVIDIA, in the previously referenced paper, has proven that they could successfully generate recovery data with image transformations. On the other hand, I wasn't achieving the desired results, so I tried a different approach.

In the real life, you have limitations on how many cameras and how far you can install them in the vehicle, but as we are running a simulator, these limitations don't exist. What I did next, is to clone the simulator Github repository, download Unity and have a look at the simulator's project. 

After some investigations (I had never used Unity), I found the location for the vehicle cameras, and as they are "free", I duplicated them to a total of 13 (I finally only used 9)  cameras at different distances from the centre. Additionally I had to modify a script to save the images from the new cameras and include them in the .csv file.

![alt text][image11]

Below you can see an image of each camera at a corner in track number one:

![alt text][image4]


Then I needed to calculate the steering angle offset for each camera. After reading [Andrew's](https://hoganengineering.wixsite.com/randomforest/single-post/2017/03/13/Alright-Squares-Lets-Talk-Triangles "Andrew's") approach, I used a similar concept for the variability of the distance to the point where the car is heading, after some tuning I ended up with the equation R = C/(angle^6 + a).

Then using some trigonometry, I calculated the steering angles for all the cameras. Below you can find images for the process:

![alt text][image5]
![alt text][image6]
![alt text][image7]


To further augment the data sat, I also flipped images and angles thinking that this would normalise the distribution of recorded angles, specially on the first track were most of the corners  are left handed. This will spare me the need of recording a lap in the opposite direction. 

Finally, I added brightness augmentation, the tracks have different ilumination, and this will help to generalize between them.

After the collection process, I had 4404 number of data points. With the additional cameras, these turned into 9 times more, to 39636.
Here we can see, the histogram before and after data augmentation. With the latter we get a something more similar to a normal distribution

![alt text][image9]
![alt text][image10]

Because I don't have enough memory to fit all the images, I used a data generator, which selects randomy an image from any of the cameras and augment it "on the fly". I finally randomly shuffled the original data, and fitted it in two generators, 20% of the original data was fitted to the validation generator.

Below you can see some examples of the output of the generator plus the normalization and cropping layers:

![alt text][image8]

For the training I used Keras callbacks to save the model every time it improved validation loss, and paused the training after 2 epochs without any improvement. This way I could use the best model.

To finis, after training the model with just one lap on each track, the vehicle was able to drive autonomously without problem on both tracks. Below you can see videos on both tracks:

Track1: (https://www.youtube.com/watch?v=WMAliBD9yyE)
Track2: (https://www.youtube.com/watch?v=bchC4b4chMw)


