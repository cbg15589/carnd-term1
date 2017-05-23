**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.png
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/bboxes_and_heat.png
[image5]: ./examples/yolov2.jpg
[image6]: ./examples/vehicle&lane.jpg


## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the second code cell of the IPython notebook `Vehicle_Detection.ipynb`. 

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

Later I explored the characteristics of the provided dataset, which contains 8792 `vehicle` images and 8968 `non-vehicle` images. All of them being 64x64 RGB images.

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

After trying various combinations of parameters I settled for `YCrCb` as colorspace, other options detected the vehicles with more confidence, but also produced more false positives. 

Using `pixels_per_cell=(8, 8)` as in the lessons performed well so I stuck to it, finally I tried bigger numbers for both `orientations` and `cells_per_block`, but they brought hardly any benefit, so I used the previously shown values.

Additionally, I added the color  features to the feature vector. First I used Spatial Binning with `spatial_size=(16,16)`, this increased the performance of the classifier, increasing the resolution didn't produce any further improvement. Second, I added the histogram information for each color channel with 32 bins.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using the labelled data explored before, this data comes from the GTI and KITTI datasets. The GTI dataset contains time-series data, so random shuffling the data will make the training and test datasets to have some very similar images. We will need to take this into account as we will get a higher score in the test than expected and also we will have the danger of overfitting. Because of these reason, I based my parameter choice more on the performance of the video test images that in the test accuracy.

The code for the training is in the eight cell of the  IPython notebook `Vehicle_Detection.ipynb`. After random shuffling the data, I trained on 80% of the data and test on the rest 20%, achieving 99,04% accuracy. As said before, we need to be cautious with this number.

As classifier, I used the `LinearSVC` from  `sklearn.svm`. Before feeding the feature vectors into the classifier we normalize them for each individual image. Not doing this could lead to bias in favour of some of the features. This is done both during training and later use of the classifier.


### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The function used for the implementation of the sliding windows is in `find_cars` in the seventh cell of the IPython notebook `Vehicle_Detection.ipynb`. First I used `ystart = 400`, `ystop = 656`,  `scale = 1.5` and `cells_per_step = 2`. This effectively turns into 96x96 windows with an overlap of 75%.

Trying to reduce the amount of windows and hence reducing the processing time, I used 4 diferent types of windows. This are tuned to the expected size of the car depending on it's location of the image. As overlapp I stuck to 75%, more overlap brought hardly any benefit and less would decrease the accuracy of the detections. The final parameters are `ystart = [380,380,395,405]`, `ystop = [650,600,540,490]` , `scale = [4,2.85,2,1.09]` and `cells_per_step = 2`. An example of the used windows can be found below:

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

To optimize the performance of the pipeline, I reduced the number of windows as explained above, this increased the performance by 50%. Using only one channel of the `YCrCb` colorspace increased the performance by 30%, but induced to many false positives, so I kept using the three of them.

The overall performance is close to 1.5 fps on a first generation core i7. This is an old computer, but still, it's very far from real time performance.

Examples of the pipeline working can be found on the Video Implementation part of the writeup.

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_out_final.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

The code to filter false positives is in the seventh and tenth cell of the  IPython notebook `Vehicle_Detection.ipynb`, where the functions `find_cars` and `process_image` are respectively.

I recorded the positions of positive detections in each frame of the video. From the positive detections I created a heatmap, `find_cars` outputs the heatmap.  Then I thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video and the bounding boxes then overlaid on the last frame of video:

![alt text][image4]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

After tuning the parameters as explained previously, I managed to get a pipeline which could identify the vehicles during the whole video, although the bounding boxes would not accurately estimate the size of the car in some parts of the video. Also, even after reducing the number of searched windows, the performance of the pipeline was very far from real time. The performance could be increased by doing some tasks as the window search in parallel and maybe running some of the code in a GPU, but this is beyond the scope of the project.

This made me think of the need of finding a different and easier way of tackling the problem. After some research on object detection systems, especially on deep neural networks, I found [YOLOV2](https://www.pjreddie.com/media/files/papers/YOLO9000.pdf), which is a state-of-the-art deep neural network for real-time object recognition. YOLOV2 is one of the most accurate and faster systems at the moment. For more information about YOLOV2 visit this [webpage](https://www.pjreddie.com/darknet/yolo/)

 [Allanzelener](https://www.pjreddie.com/media/files/papers/YOLO9000.pdf) has created some scripts to port YOLOV2 to Keras, which makes it very easy to implement. After fixing some issues with the directories parsing due to the use of Windows, the scripts give us the required files, including the model.h5.

Later and also modifying slightly the code from Allanzelener I ran the model on the project video, achieving 10fps on the same computer with a GTX 1060. The accuracy with the pre-trained weights is also much better than with the HOG features and it could be further improved training on vehicle specific datasets as the one provided by Udacity. I will explore this possibility in further projects.

A smaller model called Tiny YOLO achieves 15fps but with worse accuracy.

Below you can find a frame example and a link to a video using YOLOV2

![alt text][image5]

Here's a [link to my video with YOLOV2](./project_video_out_yoloV2.mp4)

To finalize I combined this with the Advanced Lane Finding project, the result can be found below.

![alt text][image6]

Here's a [link to my video with YOLOV2 and Advanced Lane Finding](./project_video_out_yoloV2_lane_finding.mp4)
