## Writeup
---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/carnotcarexample.png
[image2]: ./output_images/hogfeatures.png
[image3]: ./output_images/examplewindows.png
[image4]: ./output_images/boxandoriginal.png
[image5]: ./output_images/framevideo.png
[image6]: ./output_images/labelmap.png
[video1]: ./project_video_output.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the second code cell of the IPython notebook. For this I defined a function called `get_hog_feat()`, this function has also a parameter called `vis` which if true allows the function to visualize the hog features of the imputed image. Inside this function the `hog` function from the `skimage.feature` library is run the obtain the features of an image.

I started by reading in all the `vehicle` and `non-vehicle` images in the first code snippet of the IPython notebook.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `Grayscale` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and different test images and tried to see which combinations differ more for vehicles and non-vehicle examples. With this I tried to find the best HOG features that could tell apart the vehicles from non-vehicle objects. I realized that the `YCrCb` color space is a good candidate and went on with this color space. The other parameters I chose again with a lot of experimenting but not only with the images of cars and no cars but also with experimenting with the classifier and seeing how well it does with the gathered parameters. 

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using HOG features, color features as well as color histogram features. All of these features are calculated for an image and put together into a vector. For this I use the `extract_features()` function which is defined in the 3rd code snippet in the `Project.ipynb` code. This function receives 3 boolean variables `spatial_feat`, `hist_feat`, and `hog_feat` that decide which one of these features should be collected from the image. I decided to use all three features. The parameters for the HOG features are `orientations=8`, `pixels_per_cell=(4, 4)` and `cells_per_block=(16, 16)` and also for all the channels of the image in the `YCrCB` color space. The colro features are from the resized image as an 16x16 pixel image. For the histogram features I used 9 bins for the image. 
After gathering all this features and putting them together, I normalized the vector in the 4th code snippet of the `Project.ipynb` file before the training of the SVM classifier. The following code shows how the car and not car features are combined and normalized in python:

```python
X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

```

After the classifier has been trained on the training set, it has been tested on the test size and an accuracy of 98.4% is reached.


### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search. How did you decide what scales to search and how much to overlap windows?

The first version of the window search was implemented in the 5th code snippet of the `Project.ipynb` file. It is simple version of the `find_cars()` function that is explained in the next section and is called `search_window()`.
This function works good but the problem was the high number of false positives so I had to change this function later. 
For this part I decided to search from the y position 500 onward and used an overlap of 75% and got the following image:

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working. What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result. I decided to increase the number of search windows and optimized the `search_window()` function and change it to `find_cars()` function which can be found in the 6th code snippet of the `Project.ipynb` file. Here another optimization is that we take only the HOG features of the image once and then use sub-images to take the features of the image again.  Here is an example image of a picture and the many boxes found in it:

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_output.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, and the result of `scipy.ndimage.measurements.label()`  on the first frame of video:

### Here are five frames after applying the heat threshold and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

For implementing this project I started with using the grayscale image space and experimented with a lot of different parameters for the HOG, color and histogram features. At first I got good results but a lot of false positives, and one problem was that the car detection wasn't also so good that I could clear this with setting a low threshold for the heat map. So what I tried was to search each frame very intensively to find the cars a lot of times with my windows and set the heat threshold very high to eliminate the false positives. One week point that I can think of for my pipeline is that I have fixed values for ystar point which tells the algorithm from which position in y axis it should start to look for cars. If the camera resolution changes this algorithm might not work well. A good way to make my algorithm more robust would be to implement kind of a history keeping for the heat maps to average it over different frames to smooth the boxes around the found cars.  

