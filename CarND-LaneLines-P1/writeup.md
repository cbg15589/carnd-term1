
**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

# Reflection

## 1. Pipeline description.

![Original Image][image1]

Taking the previous image as an example, my pipeline consisted in the following steps:

1. Convert the images to greyscale and apply an additional Gaussian smoothing to reduce noise and undesitable gradients. 

![Grayscale Image][image2]

2. Apply the Canny Edge detection algorithm, which will help us to identify the lane lines. After some experimenting, I selected 80 and 27 for the maximum and minimum thresholds, following the recommended ratios. These values seemed  quite low compared to the example in the course example, but I found they help on the light tarmac parts of the challenge video.

![Canny Edge Image][image3]

3. Mask the image, so that only the relevant areas are used for the lines identification. The provided helper funtion "region_of_interest" already take more than one set of vertices without problem, hence the use of two different areas where the lines are expected to be found. This helped in the challenge video, as many cracks of the pavement, in the middle of the lane, were being identified as edges.

![Masked Canny Edge Image][image4]

4. Using the Hough Transform will provide us with a set of lines based on the detected edges. To select the settings, when facing the challenge video, after improving the previous steps, I selected the most problematic frames and added them to the "test_images" folder. Then, I conducted a small Design of Experiments, and found out that the best compromise across all the images were: rho = 1, theta = 3*np.pi/180, threshold = 15, min_line_length = 15, max_line_gap = 5.

![Hough Lines Image][image5]

5. Process the lines to draw a single line on both sides. For each area which we used on the third step, we can calculate the maximum and minimum possible slope of a line across the area. We use this limits to sort the lines between right, left, and not relevant. Then for each group we average the slope and the constant parameter. Finally with the obtained line ecuations, we draw both the right and left lines.
 
![Masked Areas Image][image6]

![Lines Image][image7]

6. To finish we merge the lines image with the original image.
 
![Final Image][image8]


###2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when ... 

Another shortcoming could be ...


###3. Suggest possible improvements to your pipeline

A possible improvement would be to ...

Another potential improvement could be to ...