# PanoramicImage

Final project for Computer Vision course.

Develop an algorithm to generate a panoramic image starting from a dataset of images acquired with a rotating camera from a single viewpoint.

Steps of the algorithm:
1. load a set of images and apply some pre-processing operation;
2. extract the features from the images;
3. compute the matches between features extracted in consecutive images;
4. calculate the homography transform between the panoramic image under construction and the new image;
5. blend adjacent images into a single image to generate a panorama mosaic.
