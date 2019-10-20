# **Traffic Sign Recognition** 

## Writeup

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

[image1]: ./writeup_images/distribution_classes.png "Distribution of dataset"
[image2]: ./writeup_images/original_image.png "Original Image"
[image2p]: ./writeup_images/preprocessed_image.png "Preprocessed Image"
[image3]: ./writeup_images/aug_image.png "Augmented Image"
[image4]: ./test_images/_67626131_speed-limit.jpg "Traffic Sign 1"
[image5]: ./test_images/eyeem-141845386.jpg "Traffic Sign 2"
[image6]: ./test_images/test1.jpg "Traffic Sign 3"
[image7]: ./test_images/test3.jpg "Traffic Sign 4"
[image8]: ./test_images/test4.jpg "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed. Here we can see that this dataset is highly unbalanced and this could lead to an unproper training of our network and thus having some road sign not correctly predicted

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale but that didn't seem to increase my network accurancy. I've run many training session and it seemed that preserving the color (with an additional preprocess phase) was increasing the accurancy (but sometimes even running twice the same network changed the accurancy by 1%). The preprocess I've tried to apply was first an augmentation of brightness but even there, no major improvement where found while instead, equalizing histogram by enhancing with low contrast seemed to bring some improvement in accurancy.

Here is an example of a traffic sign image before and after applying the histogram equalization.

![alt text][image2]
![alt text][image2p]

You can see that even applying that preprocess, the image kepps looking very natural

Due to the unimbalanced nature of dataset, I decided to generate additional data. I wrote a code that, given a threshold, generates many new data starting from the original one by applying randomly a rotation, translation, shear and brightness. I've choosen this way instead of relying on the *ImageDataGenerator* augmentation function provided by keras because here I wanted to augment only some particular classes and not the whole dataset

Here is an example an augmentation (original is the one above):

![alt text][image3]



#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3x32   	| 3x3 stride, same padding, outputs 32x32x32 	|
| RELU					|												|
| Convolution 3x3x32   	| 3x3 stride, same padding, outputs 30x30x32 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 15x15x32 				|
| Convolution 3x3x64   	| 3x3 stride, same padding, outputs 15x15x64 	|
| RELU					|												|
| Convolution 3x3x64   	| 3x3 stride, same padding, outputs 13x13x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 6x6x64   				|
| Dropout               | Droput with 20% probability                   |
| Convolution 3x3x128  	| 3x3 stride, same padding, outputs 6x6x128 	|
| RELU					|												|
| Convolution 3x3x128  	| 3x3 stride, same padding, outputs 4x4x128 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 2x2x128  				|
| Dropout               | Droput with 20% probability                   |
| Flatten               | Flattening to link with Dense layer           |
| Dense         	    | 512 neurons  									|
| RELU					|												|
| Dense         	    | 43 neurons  									|
| Softmax				|           									|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To build the above model, I used Keras framework since is much less verbose than tensorflow (I swear I know even programming with tensorflow). To train the model, I've tried many different hyperparameters, included the one related on the augmentation function (generating many augmented images didn't increase the final accurancy). I've tryed different optimizers like *Adam* or *Adagrad* and finally *SGD*. To avoid overfitting I've introduced then the droput and even the regularizers (that then I removed). Leveraging on the the already available callbacks in Keras I didn't take much care of the epochs since I used the early stopping while saving the best model based on *validation_loss* score. I even used the *ReduceLROnPlateau* callback to lower the learning rate when no improvements are found, but then I wrote a custom function for the SGD optimizer.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

In addition of what described above, I've implemented several networks, starting from LeNet to a custom Network that probably was to complex for the dataset we are using. I've tried different run, with many parameters and I tuned my networks based on the outcome (overfitting, not enough generalization) by adding or subtracting layers or fine tuning layer parameters (regularizers, activations, MaxPool vs AvgPool). 
I've tried different kernel size and different stride options and then different probabilities for the droputs.
I even thought to use transfer learning starting from a well known architecture but that felt like using too much power for such a simple task (I mean, based on the dataset provided)

My final model results were:
* training set accuracy of 99.7%
* validation set accuracy of 97.26%
* test set accuracy of 94%


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first and the second images might be difficult to classify because the provided training dataset is depicting well cropped images while these first 2 include a background that might fool the network. if you look at the notebook, you can find that cropping those images I've found on internet led to a better prediction accurancy

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction with the original images found on internet:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Wild animal crossing	| No entry  									| 
| Children crossing 	| Children crossing								|
| Yield					| Yield											|
| 120 km/h	      		| 20 km/h   					 				|
| Stop      			| End of no passing by vehicles over 3.5 tons	|


If considering the prediction on the original images, the model was able to correctly guess 2 of the 5 traffic signs, which gives an accuracy of 80%. This is much less than the test 

Below you can find the prediction after cropping the images

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Wild animal crossing	| Wild animal crossing							| 
| Children crossing 	| Children crossing								|
| Yield					| Yield											|
| 120 km/h	      		| 20 km/h   					 				|
| Stop      			| Stop                                      	|


If considering the prediction on the cropped images, the model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This is much less than the test but better than the above score

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located at the end of the Ipython notebook.

For all the images but the speed limit one, the model is relatively sure that the sign is what it really is (there are relatively high probabilities for the *children crossing* one). For the speed limits, even here the model is pretty sure it is 20km/h one but the image tells us it wrong. Probably is an effect of the unbalanced dataset


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

