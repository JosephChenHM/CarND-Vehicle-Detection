# **Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

---

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the third code cell of the IPython notebook.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![](https://i.imgur.com/bBYIpi0.png)


I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![](https://i.imgur.com/BVZHUPY.png)


#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and color spaces. Here is my final setting of parameters.



| Name of Parameter | Value |
| -------- | -------- |
| color_space     | YCrCb     |
| spatial_size     | (32,32)     |
| orient     | 9     |
| pix_per_cell     | 8     |
| cell_per_block     | 2     |
| hist_bins     | 32     |

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using HOG and color features. To train a robust classifier, firstly, I need to extract all the features from all training car and non-cat image. Then I split those image dataset into 80-20 train-test dataset. Finally, those features were fed to LinearSVC classifier of `sklearn`. My model training pipeline is written in `Model Training.ipynb` To get more robust image features, I transformed image color space from RGB to YCrCb to better classify performance.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search pre-defined area of an image with the different position of Y-axis and three different scales.


|          | Scale 1  | Scale 2  | Scale 3  |
| -------- | -------- | -------- | -------- |
| ystart   | 380      | 480      | 1        |
| ystop    | 380      | 480      | 1        |
| scale    | 380      | 480      | 1        |

![](https://i.imgur.com/4bey3K1.png)


#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on three different scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![](https://i.imgur.com/ayRkCjh.png)


---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections, I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of the video:

### Here image , their corresponding heatmap and resulting bounding boxes on the frame:

![](https://i.imgur.com/oHwv9pa.jpg)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

In order to have robust performance, we have to extract more features for train better model to find the car. However, it will take too much computation, and it won't be useful in real-time and in reality. Also, our developed method maybe fails to detect the car in varied lighting world conditions.
 


