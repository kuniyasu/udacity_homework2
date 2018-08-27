# **Traffic Sign Recognition** 

### Writeup

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is ?

Number of training examples = 34799. It can calculate to length of training image _np.array()_.

* The size of the validation set is ?

Number of validation examples = 4410.  It can calculate to length of validation image _np.array()_.

* The size of test set is ?

Number of testing examples = 12630.  It can calculate to length of test image _np.array()_.

* The shape of a traffic sign image is ?

Image data shape = (32, 32, 3). ( _X_train[0].shape_ )

* The number of unique classes/labels in the data set is ?

Number of classes = 43.  It can calculate to pandas data structure index size.



#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. 
![alt text](https://raw.githubusercontent.com/kuniyasu/udacity_homework2/master/image/image_load_test.png)

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because ...

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data because ...

I decided to generate additional data because ... 

To add more data to the the data set, I used the following techniques because ... 

Here is an example of an original image and an augmented image:

![alt text][image3]

The difference between the original data set and the augmented data set is the following ... 


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		      |     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		      | 32x32x3 RGB image   							| 
| Convolution 5x5     	 | 1x1 stride, same padding, outputs 28x28x8 	|
| RELU					             |												|
| Max pooling	      	   | 2x2 stride,  outputs 14x14x8 				|
| Convolution 5x5	      | 1x1 stride, same padding, outputs 10x10x16     |
| RELU					             |												|
| Max pooling	      	   | 2x2 stride,  outputs 5x5x16 				|
| Flatten    	      	   | outputs 400 				|
| Fully connected		     | outputs 200        									|
| Fully connected		     | outputs 100        									|
| Fully connected		     | outputs 43        									|
| Softmax				           |         									|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

- EPOCHS = 50
- BATCH_SIZE = 32
- rate = 0.001


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
- Training Accuracy = 0.998
- Validation Accuracy = 0.952
- Test Accuracy = 0.932

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text](https://raw.githubusercontent.com/kuniyasu/udacity_homework2/master/image/test_image1.png)
![alt text](https://raw.githubusercontent.com/kuniyasu/udacity_homework2/master/image/test_image2.png)
![alt text](https://raw.githubusercontent.com/kuniyasu/udacity_homework2/master/image/test_image3.png)
![alt text](https://raw.githubusercontent.com/kuniyasu/udacity_homework2/master/image/test_image4.png)
![alt text](https://raw.githubusercontent.com/kuniyasu/udacity_homework2/master/image/test_image5.png)

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| 	Prediction |	Truth |
|:---:|:---:|
| 	2 	|2   |
| 	18 |	18 |
| 	1 	|1   |
| 	13 |	13 |
| 	27 |	27 |


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

- For the first image, the model is relatively sure that this is a steed limit sign (softmax is 9.999996e-01)

![graph](https://raw.githubusercontent.com/kuniyasu/udacity_homework2/master/image/graph_1.png)


| 	Label ID  |	Label Name 	| Probability Softmax value| 
|:----------:|:-----:|:-----:|
| 	2 	       | Speed limit (50km/h)|  	9.999996e-01| 
| 	5 	       | Speed limit (80km/h)|  	4.098541e-07| 
| 	1 	       | Speed limit (30km/h)|  	2.018905e-17| 
| 	6 	| End of speed limit (80km/h)|  	1.909201e-18| 
| 	3 	| Speed limit (60km/h)|  	3.594470e-21| 


- For the second image ... 

![graph](https://raw.githubusercontent.com/kuniyasu/udacity_homework2/master/image/graph_2.png)

| 	Label ID  |	Label Name 	| Probability Softmax value| 
|:----------:|:-----:|:-----:|
|18 	|General caution 	|1.000000e+00|
|26 	|Traffic signals 	|2.492923e-15|
|27 	|Pedestrians 	|2.734097e-17|
|4 	 |Speed limit (70km/h) 	|5.929795e-18|
|33 	|Turn right ahead 	|5.518930e-21|

- For the third image ... 

![graph](https://raw.githubusercontent.com/kuniyasu/udacity_homework2/master/image/graph_3.png)

| 	Label ID  |	Label Name 	| Probability Softmax value| 
|:----------:|:-----:|:-----:|
|1 	|Speed limit (30km/h) 	|1.000000e+00|
|5 	|Speed limit (80km/h) 	|1.691764e-12|
|7 	|Speed limit (100km/h) |9.439109e-15|
|2 	|Speed limit (50km/h) 	|1.936896e-16|
|6 	|End of speed limit (80km/h) 	|5.602811e-19|

- For the fourth image ... 

![graph](https://raw.githubusercontent.com/kuniyasu/udacity_homework2/master/image/graph_4.png)

| 	Label ID  |	Label Name 	| Probability Softmax value| 
|:----------:|:-----:|:-----:|
|13| 	Yield 	|1.000000e+00|
|12| 	Priority road 	|3.126491e-31|
|	1| 	Speed limit (30km/h) 	|3.862392e-32|
|	9| 	No passing 	|2.734379e-32|
|25| 	Road work 	|1.629412e-32|

- For the fifith image ... 

![graph](https://raw.githubusercontent.com/kuniyasu/udacity_homework2/master/image/graph_5.png)

| 	Label ID  |	Label Name 	| Probability Softmax value| 
|:----------:|:-----:|:-----:|
|27| 	Pedestrians 	|9.999998e-01|
|24| 	Road narrows on the right 	|1.944014e-07|
|18| 	General caution 	|3.473278e-13|
|11| 	Right-of-way at the next intersection 	|4.895877e-16|
|28| 	Children crossing 	|1.022226e-18|

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

A feature image is almost dark in layer2. It means that the feture map is not effective. If we will input all image and the feture will be dark on every image, it is possible that drop down is effect.

- Input Image

![input_image](https://raw.githubusercontent.com/kuniyasu/udacity_homework2/master/image/ext_image1.png)

- Layer 1

![feature_image1](https://raw.githubusercontent.com/kuniyasu/udacity_homework2/master/image/feature_stage1.png)

- Layer 2

![feature_image2](https://raw.githubusercontent.com/kuniyasu/udacity_homework2/master/image/feature_stage2.png)

