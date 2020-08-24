# face Recognition Based Attendence

## Intro ..
how to code face recognition with OpenCV, after all this is the only reason why you are reading this article, right? OK then. You might say that our mind can do these things easily but to actually code them into a computer is difficult? Don't worry, it is not. Thanks to OpenCV, coding face recognition is as easier as it feels. The coding steps for face recognition are very similar in real life also.

- Training Data Gathering: Gather face data (face images in this case) of the persons you want to recognize
- Training of Recognizer: Feed that face data (and respective names of each face) to the face recognizer so that it can learn.
- Recognition: Feed new faces of the persons and see if the face recognizer you just trained recognizes them.
OpenCV comes equipped with built in face recognizer, all you have to do is feed it the face data. It's that simple and this how it will look once we are done coding it.

## OpenCV Face Recognizers

OpenCV has three built in face recognizers and thanks to OpenCV's clean coding, you can use any of them by just changing a single line of code. Below are the names of those face recognizers and their OpenCV calls. 

1. EigenFaces Face Recognizer Recognizer - `cv2.face.createEigenFaceRecognizer()`
2. FisherFaces Face Recognizer Recognizer - `cv2.face.createFisherFaceRecognizer()`
3. Local Binary Patterns Histograms (LBPH) Face Recognizer - `cv2.face.createLBPHFaceRecognizer()`



We have got three face recognizers but do you know which one to use and when? Or which one is better? I guess not. So why not go through a brief summary of each, what you say? I am assuming you said yes :) So let's dive into the theory of each. 

### EigenFaces Face Recognizer

This algorithm considers the fact that not all parts of a face are equally important and equally useful. When you look at some one you recognize him/her by his distinct features like eyes, nose, cheeks, forehead and how they vary with respect to each other. So you are actually focusing on the areas of maximum change (mathematically speaking, this change is variance) of the face. For example, from eyes to nose there is a significant change and same is the case from nose to mouth. When you look at multiple faces you compare them by looking at these parts of the faces because these parts are the most useful and important components of a face. Important because they catch the maximum change among faces, change the helps you differentiate one face from the other. This is exactly how EigenFaces face recognizer works.  

EigenFaces face recognizer looks at all the training images of all the persons as a whole and try to extract the components which are important and useful (the components that catch the maximum variance/change) and discards the rest of the components. This way it not only extracts the important components from the training data but also saves memory by discarding the less important components. These important components it extracts are called **principal components**. Below is an image showing the principal components extracted from a list of faces.

**Principal Components**
![eigenfaces_opencv](visualization/eigenfaces_opencv.png)
**[source](http://docs.opencv.org/2.4/modules/contrib/doc/facerec/facerec_tutorial.html)**

You can see that principal components actually represent faces and these faces are called **eigen faces** and hence the name of the algorithm. 

So this is how EigenFaces face recognizer trains itself (by extracting principal components). Remember, it also keeps a record of which principal component belongs to which person. One thing to note in above image is that **Eigenfaces algorithm also considers illumination as an important component**. 

Later during recognition, when you feed a new image to the algorithm, it repeats the same process on that image as well. It extracts the principal component from that new image and compares that component with the list of components it stored during training and finds the component with the best match and returns the person label associated with that best match component. 

#### CONCLUSIONS
1. Eigenfaces is one of the simplest and oldest face recognition algorithms.
2. It is relatively fast compared to other techniques for classifying faces
3. The feature extractor must be retrained if large number of new faces are added to the system
4. It is not accurate enough by itself and needs boosting methods for improvement.

Easy peasy, right? Next one is easier than this one. 

### FisherFaces Face Recognizer 

This algorithm is an improved version of EigenFaces face recognizer. Eigenfaces face recognizer looks at all the training faces of all the persons at once and finds principal components from all of them combined. By capturing principal components from all the of them combined you are not focusing on the features that discriminate one person from the other but the features that represent all the persons in the training data as a whole.

This approach has drawbacks, for example, **images with sharp changes (like light changes which is not a useful feature at all) may dominate the rest of the images** and you may end up with features that are from external source like light and are not useful for discrimination at all. In the end, your principal components will represent light changes and not the actual face features. 

Fisherfaces algorithm, instead of extracting useful features that represent all the faces of all the persons, it extracts useful features that discriminate one person from the others. This way features of one person do not dominate over the others and you have the features that discriminate one person from the others. 

Below is an image of features extracted using Fisherfaces algorithm.

**Fisher Faces**
![eigenfaces_opencv](visualization/fisherfaces_opencv.png)
**[source](http://docs.opencv.org/2.4/modules/contrib/doc/facerec/facerec_tutorial.html)**

You can see that features extracted actually represent faces and these faces are called **fisher faces** and hence the name of the algorithm. 

One thing to note here is that **even in Fisherfaces algorithm if multiple persons have images with sharp changes due to external sources like light they will dominate over other features and affect recognition accuracy**. 
#### CONCLUSION
Fischerfaces yields much better recognition performance than eigen faces.  However, it loses the ability to reconstruct faces because the Eigenspace is lost.  Also, Fischer faces greatly reduces the dimensionality of the images making small template sizes.

Getting bored with this theory? Don't worry, only one face recognizer is left and then we will dive deep into the coding part. 

### Local Binary Patterns Histograms (LBPH) Face Recognizer 

I wrote a detailed explaination on Local Binary Patterns Histograms in my previous article on [face detection](https://www.superdatascience.com/opencv-face-detection/) using local binary patterns histograms. So here I will just give a brief overview of how it works.

We know that Eigenfaces and Fisherfaces are both affected by light and in real life we can't guarantee perfect light conditions. LBPH face recognizer is an improvement to overcome this drawback.

Idea is to not look at the image as a whole instead find the local features of an image. LBPH alogrithm try to find the local structure of an image and it does that by comparing each pixel with its neighboring pixels. 

Take a 3x3 window and move it one image, at each move (each local part of an image), compare the pixel at the center with its neighbor pixels. The neighbors with intensity value less than or equal to center pixel are denoted by 1 and others by 0. Then you read these 0/1 values under 3x3 window in a clockwise order and you will have a binary pattern like 11100011 and this pattern is local to some area of the image. You do this on whole image and you will have a list of local binary patterns. 

**LBP Labeling**
![LBP labeling](visualization/lbp-labeling.png)

Now you get why this algorithm has Local Binary Patterns in its name? Because you get a list of local binary patterns. Now you may be wondering, what about the histogram part of the LBPH? Well after you get a list of local binary patterns, you convert each binary pattern into a decimal number (as shown in above image) and then you make a [histogram](https://www.mathsisfun.com/data/histograms.html) of all of those values. A sample histogram looks like this. 

**Sample Histogram**
![LBP labeling](visualization/histogram.png)


I guess this answers the question about histogram part. So in the end you will have **one histogram for each face** image in the training data set. That means if there were 100 images in training data set then LBPH will extract 100 histograms after training and store them for later recognition. Remember, **algorithm also keeps track of which histogram belongs to which person**.

Later during recognition, when you will feed a new image to the recognizer for recognition it will generate a histogram for that new image, compare that histogram with the histograms it already has, find the best match histogram and return the person label associated with that best match histogram. 
<br><br>
Below is a list of faces and their respective local binary patterns images. You can see that the LBP images are not affected by changes in light conditions.

**LBP Faces**
![LBP faces](visualization/lbph-faces.jpg)
**[source](http://docs.opencv.org/2.4/modules/contrib/doc/facerec/facerec_tutorial.html)**
#### Operation:
---iamge
* Convert facial image to grayscale.
* Select a window of 3×3 pixels.  It will be a 3×3 matrix containing the intensity of each pixel (0~255).
* Take the central value of the matrix and use it to threshold the neighboring pixels.
* For each neighbor of the central value (threshold), we set a new binary value. We set 1 for values equal or higher than the threshold and 0 for values lower than the threshold.
* Now, the matrix will contain only binary values (ignoring the central value). We need to concatenate each binary value from each position from the matrix line by line into a new binary value (e.g. 10001101). Note: some authors use other approaches to concatenate the binary values (e.g. clockwise direction), but the final result will be the same.
* Then, we convert this binary value to a decimal value and set it to the central value of the matrix, which is actually a pixel from the original image.
* At the end of this procedure (LBP procedure), we have a new image which represents better the characteristics of the original image.
* Extract histogram of the LBP patterns by dividing the image into a Grid.
* As we have an image in grayscale, each histogram (from each grid) will contain only 256 positions (0~255) representing the occurrences of each pixel intensity.
* Then, we need to concatenate each histogram to create a new and bigger histogram. Supposing we have 8×8 grids, we will have 8x8x256=16.384 positions in the final histogram. * The final histogram represents the characteristics of the image original image.

----image

#### CONCLUSIONS
1. LBPH is one of the easiest face recognition algorithms.
2. It can represent local features in the images.
3. It is possible to get great results mainly in a controlled environment.
4. It is robust against monotonic gray scale transformations.



The theory part is over and now comes the coding part! Ready to dive into coding? Let's get into it then. 

# Coding Face Recognition with OpenCV

The Face Recognition process in this tutorial is divided into three steps.

1. **Prepare training data:** In this step we will read training images for each person/subject along with their labels, detect faces from each image and assign each detected face an integer label of the person it belongs to.
2. **Train Face Recognizer:** In this step we will train OpenCV's LBPH face recognizer by feeding it the data we prepared in step 1.
3. **Testing:** In this step we will pass some test images to face recognizer and see if it predicts them correctly.

This repository contains code for facial recognition using openCV and python with a tkinter gui interface. If you want to test the code then run train.py file

## Technology used :
-openCV (Opensource Computer Vision)
-Python
-tkinter GUI interface

Here I am working on Face recognition based Attendance Management System by using OpenCV(Python). One can mark thier attendance by simply facing the camera. 
## Intuition:
Here we apply LHBP(linear histograms binary patterns ) algorithm for facial recognition since its very effective and light invariance algorithm. 
Using opencv, we can implement LHBP algorithm with ease

## How it works :

When we run train.py a window is opened and ask for Enter Id and Enter Name. After enter name and id then we have to click Take Images button. By clicking Take Images camera of running computer is opened and it start taking image sample of person.This Id and Name is stored in folder StudentDetails and file name is StudentDetails.csv. It takes 60 images as sample and store them in folder TrainingImage.After completion it notify that iamges saved.

After taking image sample we have to click Train Image button.Now it take few seconds to train machine for the images that are taken by clicking Take Image button and creates a Trainner.yml file and store in TrainingImageLabel folder.

Now all initial setups are done. By clicking Track Image button camera of running machine is opened again. If face is recognised by system then Id and Name of person is shown on Image. Press Q(or q) for quit this window.After quitting it attendance of person will be stored in Attendance folder as csv file with name, id, date and time and it is also available in window.
-------

wohooo! Is'nt it beautiful? Indeed, it is! 

## End Notes

Face Recognition is a fascinating idea to work on and OpenCV has made it extremely simple and easy for us to code it. It just takes a few lines of code to have a fully working face recognition application and we can switch between all three face recognizers with a single line of code change. It's that simple. 

Although EigenFaces, FisherFaces and LBPH face recognizers are good but there are even better ways to perform face recognition like using Histogram of Oriented Gradients (HOGs) and Neural Networks. So the more advanced face recognition algorithms are now a days implemented using a combination of OpenCV and Machine learning.
