## Simple and Practical Traffic Sign Classifier (Recognition)
## Writeup

Overview
---
In this project, I will use the deep neural networks and convolutional neural networks to classify traffic signs. I will train and validate a model so it can classify traffic sign images using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). After the model is trained, I will then try out my model on new test images of German traffic signs that I found on the web.

I have included 
* an Ipython notebook `Traffic_Sign_Classifier.ipynb` that contains the all the code you need to run this project. 
* a html file `Traffic_Sign_Classifier.html` which was exported from the Ipython notebook file.
* this README.md as a writeup report 

I would be grateful if you cite this work if you like it! 


The Project
---
The goals / steps of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

In this project, I used and improved the parts of this GitHub repo: https://github.com/MarkBroerkens/CarND-Traffic-Sign-Classifier-Project. Compared to MarkBroerkens, my code uses differnet test data and provides the visualization results for the Neural Network's state `Optional Part`. Finally, please fell free and continue to develop this code more and more. 

### Dependencies
This work requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab environment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

### Dataset
1. Download the data set. Udacity provided a link to the data set [data set](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/traffic-signs-data.zip). This is a pickled dataset in which have been already resized the images to 32x32. It contains a training, validation and test set.

### Data Set Summary & Exploration

#### 1. Basic summary

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Visualization

Here is an exploratory visualization of the data set. 

![calibration1](examples/dataset_visualization.png) 

This bar chart showing how many classes we have in the data set and each class includes how many samples.

![calibration1](examples/all_traffic_signs.png)

### Design and Test a Model Architecture

#### 1. Preprocessing

As a first step, I decided to convert the images to grayscale because several images in the training were pretty dark and contained only little color und the grayscaling reduces the amount of features and thus reduces execution time. Additionally, several research papers have shown good results with grayscaling of the images. [Yann LeCun - Traffic Sign Recognition with Multi-Scale Convolutional Networks](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf)

Here is an example of a traffic sign image before and after grayscaling.

![alt text][grayscale]

Then, I normalized the image using the formular `(pixel - 128)/ 128` which converts the int values of each pixel [0,255] to float values with range [-1,1]

#### 2. Model Architecture

The model architecture is based on the LeNet model architecture. I added dropout layers before each fully connected layer in order to prevent overfitting. My final model consisted of the following layers:

| Layer                  |     Description                                |
|------------------------|------------------------------------------------|
| Input                  | 32x32x1 gray scale image                       |
| Convolution 5x5        | 1x1 stride, valid padding, outputs 28x28x6     |
| RELU                   |                                                |
| Max pooling            | 2x2 stride,  outputs 14x14x6                   |
| Convolution 5x5        | 1x1 stride, valid padding, outputs 10x10x16    |
| RELU                   |                                                |
| Max pooling            | 2x2 stride,  outputs 5x5x16                    |
| Flatten                | outputs 400                                    |
| **Dropout**            |                                                |
| Fully connected        | outputs 120                                    |
| RELU                   |                                                |
| **Dropout**            |                                                |
| Fully connected        | outputs 84                                     |
| RELU                   |                                                |
| **Dropout**            |                                                |
| Fully connected        | outputs 43                                     |
| Softmax                |                                                |

![alt text][model_architecture]


#### 3. Model Training
To train the model, I used an Adam optimizer and the following hyperparameters:
* batch size: 128
* number of epochs: 150
* learning rate: 0.0006
* Variables were initialized using the truncated normal distribution with mu = 0.0 and sigma = 0.1
* keep probalbility of the dropout layer: 0.5


My final model results were:
* training set accuracy of 99.6%
* validation set accuracy of 97.5%
* test set accuracy of 95.1%

#### 4. Solution Approach
I used an iterative approach for the optimization of validation accuracy:
1. As an initial model architecture the original LeNet model from the course was chosen. In order to tailor the architecture for the traffic sign classifier usecase I adapted the input so that it accepts the colow images from the training set with shape (32,32,3) and I modified the number of outputs so that it fits to the 43 unique labels in the training set. The training accuracy was **83.5%** and my test traffic sign "pedestrians" was not correctly classified. 
  (used hyper parameters: EPOCHS=10, BATCH_SIZE=128, learning_rate = 0,001, mu = 0, sigma = 0.1) 

1. After adding the grayscaling preprocessing the validation accuracy increased to **91%** 
   (hyperparameter unmodified)

1. The additional normalization of the training and validation data resulted in a minor increase of validation accuracy: **91.8%** (hyperparameter unmodified)

1. reduced learning rate and increased number of epochs. validation accuracy = **94%** 
   (EPOCHS = 30, BATCH_SIZE = 128, rate = 0,0007, mu = 0, sigma = 0.1)

1. overfitting. added dropout layer after relu of final fully connected layer: validation accuracy = **94,7%** 
   (EPOCHS = 30, BATCH_SIZE = 128, rate = 0,0007, mu = 0, sigma = 0.1)

1. still overfitting. added dropout after relu of first fully connected layer. Overfitting reduced but still not good

1. added dropout before validation accuracy = 0.953 validation accuracy = **95,3%** 
   (EPOCHS = 50, BATCH_SIZE = 128, rate = 0,0007, mu = 0, sigma = 0.1)

1. further reduction of learning rate and increase of epochs. validation accuracy = **97,5%** 
   (EPOCHS = 150, BATCH_SIZE = 128, rate = 0,0006, mu = 0, sigma = 0.1)

![alt text][Learning]

### Test a Model on New Images
#### 1. Acquiring New Images
Here are some German traffic signs that I found on the web:
![alt text][traffic_signs_orig]

The "right-of-way at the next intersection" sign might be difficult to classify because the triangular shape is similiar to several other signs in the training set (e.g. "Child crossing" or "Slippery Road"). 
Additionally, the "Stop" sign might be confused with the "No entry" sign because both signs have more ore less round shape and a pretty big red area.

#### 2. Performance on New Images
Here are the results of the prediction:

![alt text][traffic_signs_prediction]

The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 95.1%

The code for making predictions on my final model is located in the 21th cell of the [jupyter notebook](Traffic_Sign_Classifier.html).

#### 3. Model Certainty - Softmax Probabilities
In the following images the top five softmax probabilities of the predictions on the captured images are outputted. As shown in the bar chart the softmax predictions for the correct top 1 prediction is bigger than 98%. 
![alt text][prediction_probabilities_with_barcharts]

The detailed probabilities and examples of the top five softmax predictions are given in the next image.
![alt text][prediction_probabilities_with_examples]

### Possible Future Work
#### 1. Augmentation of Training Data
Augmenting the training set might help improve model performance. Common data augmentation techniques include rotation, translation, zoom, flips, inserting jitter, and/or color perturbation. I would use [OpenCV](https://opencv.org) for most of the image processing activities.

#### 2. Analyze the New Image Performance in more detail
All traffic sign images that I used for testing the predictions worked very well. It would be interesting how the model performs in case there are traffic sign that are less similiar to the traffic signs in the training set. Examples could be traffic signs drawn manually or traffic signs with a label that was not defined in the training set. 

#### 3. Visualization of Layers in the Neural Network
In Step 4 of the jupyter notebook some further guidance on how the layers of the neural network can be visualized is provided. It would be great to see what the network sees. 
Additionally it would be interesting to visualize the learning using [TensorBoard](https://www.tensorflow.org/programmers_guide/summaries_and_tensorboard)

#### 4. Further Experiments with TensorFlow
I would like to investigate how alternative model architectures such as Inception, VGG, AlexNet, ResNet perfom on the given training set. There is a tutorial for the [TensorFlow Slim](https://github.com/tensorflow/models/tree/master/research/slim) library which could be a good start.

### Additional Reading
#### Extra Important Material
* [Fast AI](http://www.fast.ai/)
* [A Guide To Deep Learning](http://yerevann.com/a-guide-to-deep-learning/)
* [Dealing with unbalanced data](https://medium.com/@vivek.yadav/dealing-with-unbalanced-data-generating-additional-data-by-jittering-the-original-image-7497fe2119c3#.obfuq3zde)
* [Improved Performance of Deep Learning On Traffic Sign Classification](https://medium.com/@vivek.yadav/improved-performance-of-deep-learning-neural-network-models-on-traffic-sign-classification-using-6355346da2dc#.tq0uk9oxy)

#### Batch size discussion
* [How Large Should the Batch Size be](http://stats.stackexchange.com/questions/140811/how-large-should-the-batch-size-be-for-stochastic-gradient-descent)

#### Adam optimizer discussion
* [Optimizing Gradient Descent](http://sebastianruder.com/optimizing-gradient-descent/index.html#adam)

#### Dropouts
* [Analysis of Dropout](https://pgaleone.eu/deep-learning/regularization/2017/01/10/anaysis-of-dropout)




## How to write a README
A well written README file can enhance your project and portfolio.  Develop your abilities to create professional README files by completing [this free course](https://www.udacity.com/course/writing-readmes--ud777).

Referencing The Project
---
If you like my code and you want to use it in your project, please refer it like this:

`Amani, Sajjad. "Simple and Practical Traffic Sign Classifier on the Road." GitHub, 25 September 2019, https://github.com/Sj-Amani/TrafficSign_Classifier`



