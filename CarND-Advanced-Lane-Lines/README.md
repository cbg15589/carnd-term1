##**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

---

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the third code cell of the IPython notebook AdvanceLines.ipynb" .  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

For convenience, I stored the camera calibration in pickle format, this way I just need to run the calibration process one time.

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

The functions to apply color and gradient thresholds are located in the sixth code cell.

For gradients  we have functions that apply thresholding to the absolute value in x or y direction,  to the magnitude and finally to the direction. After experimenting with gradient thresholding, I was not able to get rid of some noise created by shadows, specially on the challenge video.

Because of this I dicided to only apply color thresholding, the HSV colorspace does a good job identifying white lines, while for yellow lines I decided to use the LAB colorspace.

To apply the thresholdings I created a pipeline which is located on the seventh code cell. Before applying the white and yellow thresholds, I used the function cv2.createCLAHE to perform adaptative histogram equalization on the V channel of the HSV colorspace, this will improve contrast and help to identify the lines. 

Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform is located in the seventh code cell.  The `warp_image()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points, these are base on two images where the lane is straight:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 258, 682      | 450, 720        | 
| 575, 464      | 450, 0      |
| 707, 464     | 830, 0      |
| 1049, 682      | 830, 720        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then, before fittting my lane lines with a 2nd order polynoms, I tried to identify the relevant pixels for both lines. To do that, I needed to apply ones mask for each line, each mask is made from nine small windows from bottom to top. To calculate the location of the first window, taking the bottom quarter of the image and using np.convolve, I get the location with most number of pixels. Based on this first window the next ones are calculated the same way, but minimizing the search to a margin away from the location of the first window.

The functions to appy the masking are on the ninth code cell. The result from those is the following:
![alt text][image5]

Finally, the function to fit the lines is in the eleventh code cell, there we use np.polyfit to calculate the polynom coefficients. The result can be seen in the following image:

![alt text][image6]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The code for calculating the curve radius  and the offset from the center is in the same function as the one fitting the lines.

To calculate the offset, we calculate the postition of the lines at the bottom of the image, then the center of the line, the distance of this from the middle of the image is the offset we are looking for. This distance in pixels, needs to be transformed into meters. To do that, based on the known dimesions of the lane, we apply a transformation coefficient.

To calculate the curvature we transform the pixel positions to the real space and then we fit two new 2nd order polynoms. Then we use the following formula to calculate the curvature at the bottom of the image, so at the vehicle position.

![alt text][image7]


#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

To end the image processing, we plot the results back to the original image. Below you can find an example.

![alt text][image8]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Once the pipeline for single images was finished, I applied it to a complete video. In the video some problems were easy to spot. The output lines, although correct during most of the video, they were very noisy. Also, in some parts of the video, specially with the shadows and the strong changes in the lighting of the challenge video, the lines were not correctly detected.

To solve this issues, I started using the information from previous frames and smoothing the output by taking the average of the last ten frames. I more detailed explanation of the algorithm used, can be seen below:

![alt text][image9]

This made the pipeline to perform very well in the project video and quite decently in the challenge video. But that is not the case on the harder challenge video, where the harder changes in the lighting and the sharper corners make the pipeline to perform very poorly.

The different type of roads and environmental conditions make very dificultt to tune the pipeline and the different thresholdings. The pipeline could be improved by adding dynamic thresholdings that adapt to the different conditions, but still, I think it would be very difficult and time consuming to get a robust system.

As conclusion, the time and effort I spent tuning the different pipeline parameter, made me being even more amazed of the capabilities of machine learning, where on the last project, very little tuning led to a fully driving car.


