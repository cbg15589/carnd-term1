import numpy as np
import pickle
import cv2

#Line Detection Functions
def load_calibration(path):
    #Load image calibration
    calibration_file = path

    with open(calibration_file, mode='rb') as f:
        calibration = pickle.load(f)
    
    mtx, dist = calibration['mtx'], calibration['dist']
    
    return mtx, dist

def undistort(img,mtx,dist):
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    return dst

def hsv_thres_w(hsv):
    # Apply the following steps to img
    # 1) Init output image
    hsv_binary_w = np.zeros_like(hsv[:,:,0])
    # 2) Define thresholds to identify white color on HSV colorspace
    lower = np.array([0, 0, 210], dtype = "uint8")
    upper = np.array([255, 40, 255], dtype = "uint8")
    # 3) Apply mask
    mask = cv2.inRange(hsv, lower, upper)
    # 4) Create binary output with the found points
    if (cv2.findNonZero(mask) is not None):
  
        hsv_binary_w[cv2.findNonZero(mask)[:,0,1],cv2.findNonZero(mask)[:,0,0]] = 1
    
    return hsv_binary_w

def lab_thres_y(lab):
    # Apply the following steps to img
    # 1) Init output image
    lab_binary_y = np.zeros_like(lab[:,:,0])
    # 2) Define thresholds to identify yellow color on LAB colorspace
    lower = np.array([0, 0, 160], dtype = "uint8")
    upper = np.array([255, 255, 255], dtype = "uint8")
    # 3) Apply mask
    mask = cv2.inRange(lab, lower, upper)
    # 4) Create binary output with the found points
    if (cv2.findNonZero(mask) is not None):

            lab_binary_y[cv2.findNonZero(mask)[:,0,1],cv2.findNonZero(mask)[:,0,0]] = 1

    return lab_binary_y

# Color and Gradient thresholding function
def color_gradient_threshold(img, s_thresh=(175, 255), sx_thresh=(40, 100)):
    img = np.copy(img)
    # Convert to HSV color space and separate the V channel
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.uint8)
    v_channel = hsv[:,:,2]
    # Apply Local Histogram Equalization to the V channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    v_channel_eq = clahe.apply(v_channel)
    # Create new equalized image
    hsv_eq = np.zeros_like(hsv)
    hsv_eq[:,:,0] = hsv[:,:,0]
    hsv_eq[:,:,1] = hsv[:,:,1]
    hsv_eq[:,:,2] = v_channel_eq
    # Convert equalized HSV image to LAB colorspace
    img_eq = cv2.cvtColor(hsv_eq, cv2.COLOR_HSV2RGB).astype(np.uint8)
    lab = cv2.cvtColor(img_eq, cv2.COLOR_RGB2Lab).astype(np.uint8)
    
    # Threshold HSV image for "white pixels"
    hsv_binary_w = hsv_thres_w(hsv_eq)
    
    # Threshold LAB image for "yellow pixels"
    lab_binary_y = lab_thres_y(lab)

    # Stack each channel
    color_binary = np.dstack(( hsv_binary_w, lab_binary_y, np.zeros_like(v_channel))).astype(float)
    
    # Combine the two binary thresholds
    combined_binary = np.uint8(np.zeros_like(v_channel))
    combined_binary[(hsv_binary_w == 1) | (lab_binary_y == 1)] = 255

    return color_binary, combined_binary


# Define warping source and destination 
src = np.float32([[258,682],[575,464],[707,464],[1049,682]])
dst = np.float32([[450,720],[450,0],[830,0],[830,720]]) 

# Image warping function
def warp_image(img,src,dst):
    
    # Use cv2.getPerspectiveTransform() to get M, the transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    # Use cv2.warpPerspective() to warp your image to a top-down view
    img_size = (img.shape[1], img.shape[0])
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    
    return warped

# Draw window mask function
def window_mask(width, height, img_ref, center,level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
    return output


# Function to find windows centroids
def find_window_centroids(warped, window_width, window_height, margin):

    window_centroids = [] # Store the (left,right) window centroid positions per level
    window = np.ones(window_width) # Create our window template that we will use for convolutions

    # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
    # and then np.convolve the vertical image slice with the window template 

    # Sum quarter bottom of image to get slice, could use a different ratio
    l_sum = np.sum(warped[int(3*warped.shape[0]/4):,:int(warped.shape[1]/2)], axis=0)
    l_center = np.argmax(np.convolve(window,l_sum))-window_width/2
    r_sum = np.sum(warped[int(3*warped.shape[0]/4):,int(warped.shape[1]/2):], axis=0)
    r_center = np.argmax(np.convolve(window,r_sum))-window_width/2+int(warped.shape[1]/2)

    # Add what we found for the first layer
    window_centroids.append((l_center,r_center))

    # Go through each layer looking for max pixel locations
    for level in range(1,(int)(warped.shape[0]/window_height)):
            # convolve the window into the vertical slice of the image
        image_layer = np.sum(warped[int(warped.shape[0]-(level+1)*window_height):int(warped.shape[0]-level*window_height),:], axis=0)
        conv_signal = np.convolve(window, image_layer)
        # Find the best left centroid by using past left center as a reference
        # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
        offset = window_width/2
        l_min_index = int(max(l_center+offset-margin,0))
        l_max_index = int(min(l_center+offset+margin,warped.shape[1]))
        
        if (np.all((conv_signal[l_min_index:l_max_index])==0)) == False:
            l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset

        # Find the best right centroid by using past right center as a reference
        r_min_index = int(max(r_center+offset-margin,0))
        r_max_index = int(min(r_center+offset+margin,warped.shape[1]))
        if (np.all((conv_signal[r_min_index:r_max_index])==0)) == False:
            r_center = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset
        # Add what we found for that layer
        window_centroids.append((l_center,r_center))

    return window_centroids

# Function to mask the thresholded image based on fou dwindows centroids or previous identified lines
def mask_lines(warped, window_centroids = None, previous_fit = None, window_width = 50,window_height = 80, margin = 50):
        
    # Points used to draw all the left and right windows
    l_points = np.zeros_like(warped)
    r_points = np.zeros_like(warped)
    
    # If we have any window centers
    if window_centroids != None:

        # Go through each level and draw the windows 	
        for level in range(0,len(window_centroids)):
            # Window_mask is a function to draw window areas
            l_mask = window_mask(window_width,window_height,warped,window_centroids[level][0],level)
            r_mask = window_mask(window_width,window_height,warped,window_centroids[level][1],level)
            # Add graphic points from window mask here to total pixels found 
            l_points[(l_points == 255) | ((l_mask == 1) ) ] = 255
            r_points[(r_points == 255) | ((r_mask == 1) ) ] = 255

        # Draw the results
        # Add both left and right window pixels together
        template = np.array(r_points+l_points,np.uint8)
        # Add left window pixels together
        template_l = np.array(l_points,np.uint8)
        # Add right window pixels together
        template_r = np.array(r_points,np.uint8)
        # Create a zero color channel
        zero_channel = np.zeros_like(template) 
        # Make left and right window pixels green and blue respectively
        template_3channels = np.array(cv2.merge((zero_channel,template_l,template_r)),np.uint8)
        # Making the original road pixels 3 color channels
        warpage = np.array(cv2.merge((warped,warped,warped)),np.uint8)
        # Overlay the orignal road image with window results
        windows_warped = cv2.addWeighted(warpage, 1, template_3channels, 0.5, 0.0) 
        masked_warped = cv2.bitwise_and(template, warped)
        # Retrieve found pixels for each line
        pixels_left = cv2.findNonZero(cv2.bitwise_and(warped, template_l))
        pixels_right = cv2.findNonZero(cv2.bitwise_and(warped, template_r))
        
    # If we have information from previous fitted lines
    elif previous_fit != None:
        
        # Separate previous fit into both lines
        left_fit = previous_fit[0]
        right_fit = previous_fit[1]
        
        # Generate x and y values for plotting
        ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-int(window_width), ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+int(window_width), ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-int(window_width), ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+int(window_width), ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))
        
        # Draw the results
        # Draw search window for each of the lines
        window_img_l = np.zeros_like(warped)
        window_img_r = np.zeros_like(warped)
        l_mask = cv2.fillPoly(window_img_l, np.int_([left_line_pts]), (255,255,255))
        r_mask = cv2.fillPoly(window_img_r, np.int_([right_line_pts]), (255,255,255))
        template = np.array(l_mask+r_mask,np.uint8)
        template_l = np.array(l_mask,np.uint8) 
        template_r = np.array(r_mask,np.uint8)
        # Create a zero color channel
        zero_channel = np.zeros_like(template) # create a zero color channel
        # Make left and right window pixels green and blue respectively
        template_3channels = np.array(cv2.merge((zero_channel,template_l,template_r)),np.uint8) 
        # Making the original road pixels 3 color channels
        warpage = np.array(cv2.merge((warped,warped,warped)),np.uint8)
        # Overlay the orignal road image with window results
        windows_warped = cv2.addWeighted(warpage, 1, template_3channels, 0.5, 0.0) 
        masked_warped = cv2.bitwise_and(warped, template)
        pixels_left = cv2.findNonZero(cv2.bitwise_and(warped, template_l))
        pixels_right = cv2.findNonZero(cv2.bitwise_and(warped, template_r))
        
    # If no window centers found and not previous fit information is present, just display orginal road image
    else:
        windows_warped = np.array(cv2.merge((warped,warped,warped)),np.uint8)
        masked_warped = warped
        pixels_left = None
        pixels_right = None

    
    return masked_warped, windows_warped, pixels_left, pixels_right

# Function to fit lines based on pixels found
def fit_lines(masked_warped,pixels_left, pixels_right):
    
    # If no pixels are given return default values
    if pixels_left == None or pixels_right == None:
            left_fit = [np.array([False])]
            right_fit = [np.array([False])]
            left_fit_cr = None
            right_fit_cr = None
            left_curverad = None
            right_curverad = None
            pixels_left = None
            pixels_right = None
            offset = None
            left_line_x = None
            right_line_x = None
    else:
        # Separate pixels coordinates into a usable format
        lefty = pixels_left[:,0,1]
        righty = pixels_right[:,0,1]
        
        leftx = pixels_left[:,0,0]
        rightx = pixels_right[:,0,0]
        # Use np.polyfit to fit lines to the given pixels
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 21.95/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/380 # meters per pixel in x dimension
        left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
        right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)

        # Calculate the radio of curvature
        y_eval = masked_warped.shape[0]
        left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / (2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / (2*right_fit_cr[0])
    
        #Calculate x coordinate for start and end of both lines, also the offset from lane center  
        left_line_x = [left_fit[0]*y_eval**2 + left_fit[1]*y_eval + left_fit[2], left_fit[2]]
        right_line_x = [right_fit[0]*y_eval**2 + right_fit[1]*y_eval + right_fit[2], right_fit[2]]
        lane_center = (left_line_x[0] + right_line_x[0])/2
        offset = (lane_center - masked_warped.shape[1]/2)*xm_per_pix

        
    return left_fit, right_fit , left_fit_cr,right_fit_cr, left_curverad, right_curverad, offset, left_line_x, right_line_x

# Function to create and image representing the fitted lines
def fitted_lines_image(masked_warped,left_fit, right_fit, pixels_left, pixels_right):
    
    # Initialize output image as a 3-color image
    lines_img = np.zeros_like(masked_warped)
    lines_img = np.array(cv2.merge((lines_img,lines_img,lines_img)),np.uint8)
    # Add left and right pixels to the image as red and blue respectively
    for pixel in pixels_left:
        lines_img[pixel[0,1],pixel[0,0],0] = 255 
    for pixel in pixels_right:
        lines_img[pixel[0,1],pixel[0,0],2] = 255
    
    # Plot both lines
    ploty = np.linspace(0, int(masked_warped.shape[0])-1, num=int(masked_warped.shape[0]))# to cover same y-range as image
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    left_fit_pixels = np.stack((left_fitx,ploty),axis=1)
    right_fit_pixels = np.stack((right_fitx,ploty),axis=1)  
    
    # Add line pixels to the image as green pixels, checking that the calculated values are inside the image
    for pixel in left_fit_pixels:
        if (pixel[0] <= 1279) and (pixel[0] >= 0):  
            lines_img[pixel[1],pixel[0]] = (0,255,0) 
    for pixel in right_fit_pixels:
        if (pixel[0] <= 1279) and (pixel[0] >= 0):
            lines_img[pixel[1],pixel[0]] = (0,255,0)
        
    return lines_img

# Function to check if the fitted lines are plausible
def sanity_check(left_fit_cr,right_fit_cr, left_curverad, right_curverad, offset, left_line_x, right_line_x):
    result = True
    # Calculate width close to the car
    width_0 = (right_line_x[0] - left_line_x[0])*3.7/380
    # Calculate width far from the car
    width_1 = (right_line_x[1] - left_line_x[1])*3.7/380
    # Calculate curvature and curvature ratio between the both lines
    curvature_ratio_abs = abs(left_curverad/right_curverad)
    curvature_ratio = left_curverad/right_curverad
    curvature = (left_curverad+right_curverad)/2
    # Check for width
    if width_0 > 4.4 or width_0 < 2.8:
        result = False
        print("Sanity Check Failed due to width at the start")
        print(width_0)
    if width_1 >= width_0*1.08 or width_1 < 1.8:
        result = False
        print("Sanity Check Failed due to width at the end")
        print(width_1)
    # Check for curvature ratio
    if curvature_ratio_abs > 6 or curvature_ratio_abs < 0.166:
        print("Sanity Check Failed due to curvature")
        print(curvature_ratio)
        result = False
    return result

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        self.detected_counter = 0
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None
        # fit values of the last n fits of the line
        self.recent_fits = None
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = [np.array([False])]  
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        # radius of curvature of the line in some units
        self.radius_of_curvature = None 
        # distance in meters of vehicle center from the line
        self.line_base_pos = None 
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        # difference in fit coefficients between last and new fits (percent)
        self.diffsper = np.array([0,0,0], dtype='float') 
        # x values for detected line pixels
        self.allx = None  
        # y values for detected line pixels
        self.ally = None
        
# Image processing pipeline
def detect_lane(img):
    # Load camera calibration
    mtx, dist = load_calibration('C:\carnd-term1\CarND-Advanced-Lane-Lines\camera_cal\wide_dist_pickle.p')
    # Undistort input image
    undist = undistort(img, mtx, dist)
    # Apply thresholding to the image
    result, result_combined = color_gradient_threshold(undist)
    # Get perspective transform
    src = np.float32([[258,682],[575,464],[707,464],[1049,682]])
    dst = np.float32([[450,720],[450,0],[830,0],[830,720]]) 
    M = cv2.getPerspectiveTransform(src, dst)
    # Warp image
    img_size = (img.shape[1], img.shape[0])
    warped = cv2.warpPerspective(result_combined, M, img_size, flags=cv2.INTER_LINEAR)
    orig_warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    # Search window settings
    window_width = 50 
    window_height = 80 # Break image into 9 vertical layers since image height is 720
    margin = 35 # How much to slide left and right for searching
    # Get previous fit
    previous_fit = np.array([left_line.current_fit, right_line.current_fit])
    # Get previous best fit
    best_fit = np.array([left_line.best_fit, right_line.best_fit])
    # Initialize image
    lines_img = np.zeros_like(img)
    
    # In there is no previous fit or the non-detected counter achieved it's limit,
    # calculate windows centroids
    if (previous_fit == False).any() or left_line.detected_counter >= 10 or right_line.detected_counter >= 10:
        print('Calculating Centroids')
        window_centroids = find_window_centroids(warped, window_width, window_height, margin = margin)
    else:
        window_centroids = None        
    
    # Mask lines based on previous fit or calculated windows centroids
    masked_warped, windows_warped, pixels_left, pixels_right =  mask_lines(warped, window_centroids,best_fit, window_width,window_height, margin)
    
    # If no pixels were found for any of the lines, reset both and abort further calculations
    if pixels_left == None:
        left_line.detected = False
        left_line.detected_counter += 1
        left_curverad = None
        right_curverad = None
        print('No Left Pixels')
    elif pixels_right == None:
        right_line.detected = False
        right_line.detected_counter += 1
        right_curverad = None
        left_curverad = None
        print('No Right Pixels')
    else:
        # Separate found pixels coordinates
        left_line.allx = pixels_left[:,0,0]
        left_line.ally = pixels_left[:,0,1]
        right_line.allx = pixels_right[:,0,0]
        right_line.ally = pixels_right[:,0,1]
        # Fit lines
        left_fit, right_fit , left_fit_cr,right_fit_cr, left_curverad, right_curverad, offset, left_line_x, right_line_x = fit_lines(masked_warped,pixels_left, pixels_right)
        # Create image with fitted lines
        lines_img = fitted_lines_image(masked_warped,left_fit, right_fit, pixels_left, pixels_right)
        # Check for line plausability
        sanity_check_result = sanity_check(left_fit_cr,right_fit_cr, left_curverad, right_curverad, offset, left_line_x, right_line_x)
        # If check fails add 1 to the non-detected counter and abort further calculations
        if sanity_check_result == False:
            left_line.detected = False
            left_line.detected_counter += 1
            right_line.detected = False
            right_line.detected_counter += 1
        else:
            # Line was detected
            left_line.detected = True 
            right_line.detected = True
            
            # If previous fit was given
            if window_centroids == None:
                # Append calulated x values
                left_line.recent_xfitted.append(left_line_x)
                left_line.recent_xfitted = left_line.recent_xfitted[-10:]
                right_line.recent_xfitted.append(right_line_x)
                right_line.recent_xfitted = right_line.recent_xfitted[-10:]
                
                # Calculate difference between best and current fit
                left_line.diffs = left_line.best_fit - left_fit
                left_line.diffsper = (left_line.diffs / left_line.best_fit) * 100
                right_line.diffs = right_line.best_fit - right_fit
                right_line.diffsper = (right_line.diffs / right_line.best_fit) * 100
                
                # Append calculated fit
                left_line.recent_fits = np.column_stack((left_line.recent_fits,left_fit))
                left_line.recent_fits = left_line.recent_fits[:,-10:]
                left_line.best_fit = [np.mean(left_line.recent_fits[0,:]),np.mean(left_line.recent_fits[1,:]),np.mean(left_line.recent_fits[2,:])]
                # As line was succesfully calculated, reset non-detected counter
                left_line.detected_counter = 0
                # Append calculated fit
                right_line.recent_fits = np.column_stack((right_line.recent_fits,right_fit))
                right_line.recent_fits = right_line.recent_fits[:,-10:]
                right_line.best_fit = [np.mean(right_line.recent_fits[0,:]),np.mean(right_line.recent_fits[1,:]),np.mean(right_line.recent_fits[2,:])] 
                # As line was succesfully calculated, reset non-detected counter
                right_line.detected_counter = 0
                
                # Average radius of curvature over the last 10 values
                left_line.radius_of_curvature = (left_line.radius_of_curvature*9 + left_curverad)/10 
                left_line.line_base_pos = (left_line.line_base_pos*9 + offset)/10 
                right_line.radius_of_curvature = (right_line.radius_of_curvature*9 + right_curverad)/10 
                right_line.line_base_pos = (right_line.line_base_pos*9 + offset)/10
                # Set current fit
                left_line.current_fit = left_fit
                right_line.current_fit = right_fit
            # If window centrois were calculated
            else:
                # Append calulated x values
                left_line.recent_xfitted = []
                left_line.recent_xfitted.append(left_line_x)
                right_line.recent_xfitted = []
                right_line.recent_xfitted.append(right_line_x)
                
                # Append calculated fit
                left_line.recent_fits = left_fit
                left_line.best_fit = left_fit
                # As line was succesfully calculated, reset non-detected counter
                left_line.detected_counter = 0
                # Append calculated fit
                right_line.recent_fits = right_fit
                right_line.best_fit = right_fit
                # As line was succesfully calculated, reset non-detected counter
                right_line.detected_counter = 0
                
                # Calculate difference between best and current fit
                left_line.diffs = left_line.best_fit - left_fit
                left_line.diffsper = (left_line.diffs / left_line.best_fit) * 100
                right_line.diffs = right_line.best_fit - right_fit
                right_line.diffsper = (right_line.diffs / right_line.best_fit) * 100
                
                #Set radius of curvature
                left_line.radius_of_curvature = left_curverad
                left_line.line_base_pos = offset
                right_line.radius_of_curvature = right_curverad 
                right_line.line_base_pos = offset
                
                # Set current fit
                left_line.current_fit = left_fit
                right_line.current_fit = right_fit
    
    # Compile both best fits
    best_fit = np.array([left_line.best_fit, right_line.best_fit])
    # If best fit exists
    if (best_fit != False).all():
        
        #Plot best fir lines
        ploty = np.linspace(0, int(masked_warped.shape[0])-1, num=int(masked_warped.shape[0]))# to cover same y-range as image
        left_fitx = left_line.best_fit[0]*ploty**2 + left_line.best_fit[1]*ploty + left_line.best_fit[2]
        right_fitx = right_line.best_fit[0]*ploty**2 + right_line.best_fit[1]*ploty + right_line.best_fit[2]

        # Create an image to draw the lines on
        warp_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, M, (img.shape[1], img.shape[0]), flags=cv2.WARP_INVERSE_MAP) 
        # Combine the result with the original image
        output_warped = cv2.addWeighted(orig_warped, 1, color_warp, 0.3, 0)
        output = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
        curverad = (left_line.radius_of_curvature + right_line.radius_of_curvature)/2
    else:
        # Return original image
        output = undist
        output_warped = orig_warped
        curverad = 0
    # Print data into the image
    output_image = cv2.putText(output,'Line detected: L ' + str(left_line.detected) + ' R ' + str(right_line.detected),(50,130), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)
    output_image = cv2.putText(output,'Curve Radio:' + str(curverad) + ' L ' + str(left_line.radius_of_curvature) + ' R ' + str(right_line.radius_of_curvature),(50,190), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)
    output_image = cv2.putText(output,'Offset From Center:' + str(right_line.line_base_pos),(50,250), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)
    
    return output_image

left_line = Line()
right_line = Line()