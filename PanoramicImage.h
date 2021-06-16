#ifndef PANORAMICIMAGES_H
#define PANORAMICIMAGES_H

#include <iostream>
#include <opencv2/opencv.hpp>

class PanoramicImage {
public:
	//constructor
	// input_img: array of images
	// output_img: the panoramic image as output
	PanoramicImage(std::vector<cv::Mat> input_img, cv::Mat output_img);

	//merge the input images
	void mergeImg();

	//obtain the resulting image
	cv::Mat getResult();

protected:	

	//input images
	std::vector<cv::Mat> input;

	//result
	cv::Mat output;
};
#endif //PANORAMICIMAGE_H