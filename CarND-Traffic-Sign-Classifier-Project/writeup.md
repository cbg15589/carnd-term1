# **Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/original_dataset.JPG "Original Dataset"
[image2]: ./examples/original_histogram.JPG "Original Dataset Histogram"
[image3]: ./examples/grayscaled_dataset.JPG "Grayscale Dataset"
[image4]: ./examples/normalized_dataset.JPG "Normalized Dataset"
[image5]: ./examples/modified_histogram.JPG "Modified Dataset Histogram"
[image6]: ./examples/data_augmentation.JPG "Image Augmentation"
[image7]: ./examples/LeNet_arquitecture.JPG "Lenet Architecture"
[image8]: ./test_images/a.JPG "Traffic Sign 35"
[image9]: ./test_images/b.JPG "Traffic Sign 17"
[image10]: ./test_images/c.JPG "Traffic Sign 36"
[image11]: ./test_images/d.JPG "Traffic Sign 2"
[image12]: ./test_images/e.JPG "Traffic Sign 13"
[image13]: ./test_images/f.JPG "Traffic Sign 16"
[image14]: ./test_images/g.jpg "Traffic Sign 12"
[image15]: ./test_images/h.jpg "Traffic Sign 14"
[image16]: ./test_images/i.jpg "Traffic Sign 0"
[image17]: ./test_images/j.jpg "Traffic Sign 23"
[image18]: ./test_images/k.jpg "Traffic Sign 29"
[image19]: ./test_images/l.jpg "Traffic Sign 40"
[image20]: ./examples/precision_recall.JPG "Precision & Recall"
[image21]: ./examples/softmax.jpg "Precision & Recall"
[image22]: ./examples/conv_original.JPG "Original Image"
[image23]: ./examples/conv_spatial.JPG "Spatial Transformer"
[image24]: ./examples/conv_conv1.JPG "Conv1"
[image25]: ./examples/conv_conv1_activation.JPG "Conv1 Activation"
[image26]: ./examples/conv_conv1_maxpool.JPG "Conv 1 Max-Pool"
[image27]: ./examples/conv_conv2.JPG "Conv2"
[image28]: ./examples/conv_conv2_activation.JPG "Conv2 Activation"
[image29]: ./examples/conv_conv2_maxpool.JPG "Conv 2 Max-Pool"


## Data Set Summary & Exploration

### 1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

In this project we need to build a model to classify traffic signs from the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).  As a start for the project we are given the dataset already split into training, validation and test datasets. 

First, I explored the different datasets using python and numpy methods such as "len()" or "np.ndarray.shape". The code for this step is contained in the second code cell of the IPython notebook. A summary of the datasets is:

Number of training examples = 34799

Number of validation examples = 4410

Number of testing examples = 12630

Image data shape = (32, 32, 3)

Number of classes = 43


Here we see that the validation set represents roughly 13% of the training set, and the test set is 36%. Based on the course recommendations I would have made the validation set slightly bigger, but I consider that the current split is good enough.

### 2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third and fourth code cells of the IPython notebook.  

First, we can see an example of each traffic sign class. We can observe that there is a big difference in the exposure and the detail of the images, being some of them quite blurry and dark.

![alt text][image1]

Second, a bar chart showing shows us how the data is not equally distributed between the different classes, this could be a problem for our model, as it could bias to the most numerous classes.

![alt text][image2]

## Design and Test a Model Architecture

### 1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the fifth to tenth code cells of the IPython notebook.

As a first step, I decided to convert the images to grayscale because, although some information is inevitably lost, it makes the images more simple and should help the model to focus in the most important parts of information.

Here is an example of each traffic sign image after grayscaling.

![alt text][image3]

Then, I normalized the image data, making all the pixel values to fall between [-1, 1]. When visualizing the resulting image, we can't appreciate any difference, but this will centre the problem and help the model to converge.

Here is an example of each traffic sign image after normalization.

![alt text][image4]

This preprocessing was applied to the three datasets. During the pre-processing, I only used data from each single image, in case of using statistical parameters from across the whole dataset, only information from the training dataset should be used. For example, using the mean of the test dataset, would imply using information which won't be available in a production environment.
 
### 2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

As explained above, the provided dataset was already split into training, validation and test, so I focused my efforts into improving the quality of the training dataset.

The eleventh code cell contains the code where I duplicate some of the images, to make the distribution across the different classes more even. I used the following logic: 

Example: Label 15
Most repeated label: 4, 2010 times
Label 15 is present 190 times
We will duplicate each image np.round_(2010x2/190) = 21 
(2010x2 is used to minimize the rounding error)

As a result we will have 3990 Label 15 images and 4020 Label 4.
Here we can see an histogram with the new distribution.

![alt text][image5]

The thirteenth code cell of the IPython notebook contains the code for augmenting the data set. I decided to use ImageDataGenerator from the Keras preprocessing library, I added horizontal and vertical shifting and zooming. I tried other transformations such as rotations and random noise, but they didn't seem to bring any improvement to the model. 

The ImageDataGenerator creates the new images from the original ones, in real time, during the training process. This reduces the required RAM memory to hold a training dataset, which otherwise would have been much bigger. My final training set consist in 675332 images while using the RAM needed for 168833.

Below you can see an example of some images after going through the ImageDataGenerator.

![alt text][image6]


### 3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the fifteenth cell of the ipython notebook. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   					| 
| Flatten       		| outputs 1024   								|
| Fully Connected       | outputs 20   									|
| Fully Connected       | outputs 6   									|
| Spatial transformer   | outputs 32x32x1 Grayscale image  				|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x108 	|
| Tanh					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x108 				|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x108 	|
| Tanh					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x108 					|
| Flatten*       		| outputs 23868   								|
| Fully connected		| outputs 100        							|
| Tanh					|												|
| Dropout				|												|
| Fully connected		| outputs 100        							|
| Tanh					|												|
| Dropout				|												|
| Fully connected		| outputs 43        							|
| Softmax				| 	        									|
|						|												|

*This layer is fed with the output from both convolutions
 
The model is based in the architecture used by Pierre Sermanet and Yann LeCun with a spatial transformer on top. The spatial transformer consist in a localization network and a grid generator, it focuses in the relevant data of the image and transforms it. I will analyse it's effect later on.

The particular spatial transformer class I used, was created by [David Dao](https://github.com/daviddao/spatial-transformer-tensorflow)

### 4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the sixteenth to nineteenth cells of the ipython notebook. 

To train the model, I used the Adam Optimizer, which is a gradient-based algorithm aimed towards machine learning problems with large datasets, requiring little memory, which is ideal for my current setup.

Being my final training dataset quite big, for computational efficiency reasons, I selected the biggest batch size I could afford without running into GPU memory errors, which is 512. 

For the number of epochs I selected 100, although this number is unlikely to be reached. I used "early stop", which means that after 5 epochs without improving the validation accuracy, the training will stop and the best model would have been saved.

Although an important property of the Adam Optimizer is that it uses adaptive step sizes for the parameter update, I observed that it stills benefits from learning rate decay. I manually selected the learning rate decay, using the following values:  [0.001,0.0001,0.00001,0.000001]

Finally in order to prevent overfitting, I used dropout, which randomly disconnects nodes in the fully connected layers. I set the dropout probability to 0.5. I also tried L2 regularization, but it didn't seem to bring any benefit for the architecture I finally chose.



### 5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the eighteenth cell of the Ipython notebook.

My final model results were:

 - Training set accuracy of 99.9876%
 - Validation set accuracy of 99.229% 
 - Test set accuracy of 98.234%

![alt text][image7]

First I started using LeNet's architecture, this was an easy starting point, as it was setup during one of the course labs. With this arquitecture I achieved 90% validation accuracy as "out of the box". Both validation and training accuracies where similar, suggesting that the model was underfitting. Because of this reason I decided to start to look into more complex architectures.

After reading some literature regarding the GTSRB problem, I tried some known architectures used in previous traffic sign classification works. Some of the architectures I tested were the ones created by D. Cireşan, Sermanet & LeCun and J.Jin. 

With the Sermanet & LeCun architecture I easily achieved around 97% validation accuracy. Taking this as a baseline I started introducing some changes.

First in order to decrease overfitting, I introduced dropout in the two fully connected layers. I also introduced the spatial transformer on top of the architecture, which helped to increase robustness against the different positions and sizes of the traffic signs. With this changes I achieved 98% validation accuracy.

Later I focused on improving the training dataset, adding image augmentation and having a more equal spread of the number of samples for each traffic sign. This improved the validation accuracy to 98,7%

Finally I introduced early stop and learning rate decay, which helped me to improve up to my final results. Learning rate decay, helps the model not to get stuck in a local minimum, by starting with an aggresive value, this will decay over time to give us more precision. Early stop will stop training once a number of epochs have gone through without any improvement.  
 

## Test a Model on New Images

### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are twelve German traffic signs that I found on the web:

![alt text][image8] ![alt text][image9] ![alt text][image10] ![alt text][image11]
![alt text][image12] ![alt text][image13]![alt text][image14] ![alt text][image15]
![alt text][image16] ![alt text][image17]![alt text][image18] ![alt text][image19]

All the images I found on the web or Google's Street View had good detail and exposure, so there are no special reason which should make them difficult to classify. As a further development for the future, I would apply different types of transformations to mimic real life examples, such as movement blur, to see how robust the model is against them. 

### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on the new signs is located in the 21th to 26th cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Ahead only      		| Ahead only   									| 
| No entry     			| No entry 										|
| Go straight or right	| Go straight or right							|
| 50 km/h	      		| 50 km/h					 					|
| Yield					| Yield      									|
| Vehicles over 3.5 metric tons prohibited			| Vehicles over 3.5 metric tons prohibited      							|
| Priority road			| Priority road     							|
| Stop			| Stop      							|
| 20 km/h			| 20 km/h      							|
| Slippery Road			| Slippery Road      							|
| Bicycles crossing			| Bicycles crossing      							|
| Roundabout mandatory			| Roundabout mandatory      							|


The model was able to correctly guess all of the 12 traffic signs, which gives an accuracy of 100%. This compares favourably to the accuracy on the test set of 98.266%.

While finding new examples, I wanted to know if the model has problems with any specific traffic sign, then I could find examples of it. To study this, I calculated the precision and recall for each sign. A high precision for a specific sign, means that the proportion of false positives against true positives is low, and a high recall means that proportion of false negatives is low. 

Below are the results for each dataset:

![alt text][image20] 

As we can see, looking at the validation and test dataset results, the model has some problems detecting some of the signals. For example, sign number 23 has a precision of 0.92 on the validation set, which is quite low compared to the average 0.993. It would be interesting to explore these signals that the model fails to predict, to see if this is because of the specific signal or other reasons. For example, it could be that most of the images for that type of signal are very dark, and the model struggles with dark images.

### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 27th and 28th cell of the Ipython notebook.

For all the images, the model is very sure of it's prediction (minimum probability of 0.995), this is due to the high quality of the images. The top five soft max probabilities for the images were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Ahead only   									| 
| .20     				| Turn left ahead 										|
| .05					| Go straight or left											|
| .04	      			| Turn right ahead					 				|
| .01				    | Go straight or right      							|

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| No entry   									| 
| .20     				| Stop 										|
| .05					| Turn left ahead											|
| .04	      			| Turn right ahead					 				|
| .01				    | End of no passing by vehicles over 3.5 metric tons      							|


| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Go straight or right   									| 
| .20     				| Ahead only 										|
| .05					| Turn left ahead											|
| .04	      			| Children crossing					 				|
| .01				    | Dangerous curve to the right      							|


| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Speed limit (50km/h)  									| 
| .20     				| Speed limit (30km/h) 										|
| .05					| Speed limit (80km/h)											|
| .04	      			| Speed limit (60km/h)					 				|
| .01				    | Speed limit (120km/h)      							|


| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Yield   									| 
| .20     				| Priority road 										|
| .05					| Keep right										|
| .04	      			| No vehicles					 				|
| .01				    | Ahead only      							|

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Vehicles over 3.5 metric tons prohibited   									| 
| .20     				| No passing 										|
| .05					| Speed limit (100km/h)											|
| .04	      			| No passing for vehicles over 3.5 metric tons					 				|
| .01				    | No entry      							|

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Priority road   								| 
| .20     				| Roundabout mandatory 							|
| .05					| End of all speed and passing limits			|
| .04	      			| No passing					 				|
| .01				    | Speed limit (20km/h)      					|

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop   									| 
| .20     				| No entry										|
| .05					| Turn right ahead								|
| .04	      			| Yield					 				|
| .01				    | Speed limit (50km/h)    						|


| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Speed limit (20km/h)   									| 
| .20     				| Speed limit (30km/h) 										|
| .05					| Speed limit (70km/h)											|
| .04	      			| Speed limit (120km/h)					 				|
| .01				    | Vehicles over 3.5 metric tons prohibited      							|

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Slippery road   									| 
| .20     				| Dangerous curve to the left 										|
| .05					| Beware of ice/snow											|
| .04	      			| Bicycles crossing					 				|
| .01				    | Wild animals crossing      							|

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Bicycles crossing  									| 
| .20     				| Slippery road 										|
| .05					| Bumpy road											|
| .04	      			| Wild animals crossing				 				|
| .01				    | Road narrows on the right      							|

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Roundabout mandatory   									| 
| .20     				| Go straight or left 										|
| .05					| Vehicles over 3.5 metric tons prohibited											|
| .04	      			| Slippery road					 				|
| .01				    | Speed limit (20km/h)      							|

The second and third rank predictions are similar signs to the first one, but still, the model it's very sure of it's predictions.

Below we can see the same information in the form of bar charts.

![alt text][image21] 

### 4. Visualize the Neural Network's State with Test Images

The code to visualize the Neural Network's State 29th and 30th cell of the Ipython notebook.

Below we can see the output of all the model's layers up to the fully connected layers. As input I have used one of the new images from the web, in this case a roundabout sign.

Original image:

![alt text][image22] 

Spatial transformer: We see that the spatial transformer fails to localize the bottom right corner of the sign, but at least it erases some of the irrelevant part of the image.

![alt text][image23] 

Conv1:

![alt text][image24]
 
Conv1 Activation: We can see how the convolution focuses on the important features of the sign such as the contour and the arrows.

![alt text][image25] 

Conv1 Max-Pool:

![alt text][image26] 

Conv2: Here it already very difficult to recognize any detail of the signal

![alt text][image27]
 
Conv2 Activation: 

![alt text][image28] 

Conv2 Max-Pool: In the used model, this and the previous Max-Pool are the inputs for the first fully connected layer.

![alt text][image29] 

