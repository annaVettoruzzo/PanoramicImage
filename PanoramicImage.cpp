//PanoramicImage.cpp class
#include <memory>
#include <iostream>
#include <opencv2/opencv.hpp>

#include "panoramicUtils.h"
#include "PanoramicImage.h"

using namespace std;
using namespace cv;

void equalizeImages(Mat *input0, Mat *input1, int tx);

const double RATIO = 4;

//constructor
PanoramicImage::PanoramicImage(vector<Mat> input_img, Mat output_img) {
	input = input_img;
	output = output_img;
}

void PanoramicImage::mergeImg() {
	//*************** ORB *************
	//apply ORB	(try to change nfeatures)
	Ptr<ORB> orb = ORB::create(500, 1.2f, 8, 31, 0, 2, ORB::HARRIS_SCORE, 31, 20);

	cout << "Size of the panoramic image under construction: " << input[0].size() << endl;
		
	//instead of using a mask in the vector detectAndCompute I create a temporary images with only the portion 
	//of image in which I want to detect the features
	Mat tempImg = input[0](Rect(input[0].cols - input[1].cols, 0, input[1].cols, input[0].rows));

	// detect and compute descriptors and keypoints
	vector<KeyPoint> KP1, KP2;
	Mat descriptor1, descriptor2;
	orb->detectAndCompute(tempImg, noArray(), KP1, descriptor1);	//noArray to work without a mask or Mat()
	orb->detectAndCompute(input[1], noArray(), KP2, descriptor2);
	
	//find the matches between each pair of images
	vector<DMatch> matches;
	BFMatcher matcher(NORM_HAMMING, false); //we can use true to have more robust features
	matcher.match(descriptor1, descriptor2, matches);

	//*************** REFINE MATCHES *************
	//refine the matches found selecting the matches with distance less than ratio * min_distance.
	//These are the better matches because they have small distance (very similar)
	double min_dist = 1000;
	for (int i = 0; i < matches.size(); i++) {
		double dist = matches[i].distance;
		if (dist < min_dist) min_dist = dist;
	}
	cout << "minimum distance: " << min_dist << endl;
	vector<DMatch> good_matches;
	for (int i = 0; i < matches.size(); i++) {
		if (matches[i].distance < RATIO*min_dist) good_matches.push_back(matches[i]);
	}
	cout << "Number of matches after refinement: " << good_matches.size() << endl;
	/*
	//ADVANCED: refine the matches found like proposed by Lowe:
	//good matches if the ratio between the first and the second is low than 0.8 (like in Lowe paper)
	const float ratioLowe = 0.8;
	vector<vector<DMatch>> matchesL;
	BFMatcher matcherL;
	matcherL.knnMatch(descriptor1, descriptor2, matchesL, 2);
	vector<DMatch> good_matches;
	for (int i = 0; i < matchesL.size(); i++) {
		if (matchesL[i][0].distance < ratioLowe*matchesL[i][1].distance) {
			good_matches.push_back(matchesL[i][0]);
		}
	}
	cout << "Number of matches after refinement (with Lowe approach): " << good_matches.size() << endl;
	*/

	//*************** HOMOGRAPHY **************
	//fitting models to several random samplings of the data and returning the model 
	//that has the best fit to a subset of the data. fitting models to several random samplings of the data 
	//and returning the model that has the best fit to a subset of the data.
	//find homography with RANSAC
	vector<Point2f> res1;
	vector<Point2f> res2;
	for (size_t i = 0; i < good_matches.size(); i++) {
		res1.push_back(KP1[good_matches[i].queryIdx].pt);
		res2.push_back(KP2[good_matches[i].trainIdx].pt); 
	}
	Mat H = findHomography(res2, res1, RANSAC);	// RANSAC needs a threshold (3 by default) to distinguish inliers from outliers
	//		| H_00, H_01, H_02 |
	//	H = | H_10, H_11, H_12|
	//		|H_20, H_21, H_22 |
	int tx = (int)H.at<double>(0, 2);	//return a reference of =H_02
	//int ty = (int)H.at<double>(1, 2);	//=H_12

	cout << "X offset with function : " << tx << endl;
	//cout << "Y offset with function : " << ty << endl;
	
	/*
	//ADVANCED: implement RANSAC by hand considering 80 iterations 
	int niter = 80;
	vector<int> count(niter,0);
	vector<float> deltaX(niter,0);
	vector<float> deltaY(niter,0);
	for (int k = 0; k < niter; k++) {
		//pick an element at random and get the shift
		int index = rand() % good_matches.size();
		float dx = KP1[good_matches[index].queryIdx].pt.x - KP2[good_matches[index].trainIdx].pt.x;
		float dy = KP1[good_matches[index].queryIdx].pt.y - KP2[good_matches[index].trainIdx].pt.y;
		deltaX[k] = dx;
		deltaY[k] = dy;
		//count the number of correspondences that are consistent with the selected one
		for (int j = 0; j < good_matches.size(); j++) {
			float dxn = KP1[good_matches[j].queryIdx].pt.x - KP2[good_matches[j].trainIdx].pt.x;
			float dyn = KP1[good_matches[j].queryIdx].pt.y - KP2[good_matches[j].trainIdx].pt.y;
			float diffx = abs(dxn - dx);
			float diffy = abs(dyn - dy);
			if ((diffx + diffy) < 5) {
				count[k] = count[k] + 1;
				deltaX[k] = deltaX[k] + dxn;
				deltaY[k] = deltaY[k] + dyn;
			}
		}
	}
	int max_index = max_element(count.begin(), count.end()) - count.begin(); //index of the correspondence with largest compatible set
	int tx_byHand = ceil(deltaX[max_index] / count[max_index]);
	//int ty = ceil(deltaY[max_index] / count[max_index]);
	
	cout << "X offset by hand : " << tx_byHand << endl;
	*/

	//*************** FINAL MERGING ***************
	//ADVANCED: make some processing on the new image before the blending
	//eliminate the first column of the image to be merged (it looks damaged)
	input[1] = input[1](Rect(1, 0, input[1].cols - 1, input[1].rows));
	
	//ADVANCED: equalize the image in the neighborhood of the attachment (choose the best params)
	Mat *tempImgPtr, *input1Ptr;
	tempImgPtr = &tempImg;
	input1Ptr = &input[1];
	equalizeImages(tempImgPtr, input1Ptr, tx);
	
	//initialize the new image
	cout << "Size" << Size(input[1].cols + tx, input[1].rows) << endl;
	Mat merge = Mat::zeros(Size(input[1].cols + tx, input[1].rows), CV_8U);
	
	//create regions of interest (ROIs) in the new image
	Mat roi1(merge, Rect(0, 0, tempImg.cols, tempImg.rows)); //create a rectangle in merge from (0,0) with width and height
	Mat roi2(merge, Rect(tx, 0, input[1].cols, input[1].rows));
	//copy the images in these regions
	tempImg.copyTo(roi1);
	input[1].copyTo(roi2);

	//merge the old panoramic with the new obtained one (do the same as before but considering input[0])
	//initialize the new image
	output = Mat::zeros(Size(input[0].cols + merge.cols - input[1].cols, input[1].rows), CV_8U);
	//create regions of interest (ROIs) in the new image
	Mat roi21(output, Rect(0, 0, input[0].cols, input[0].rows));
	Mat roi22(output, Rect(input[0].cols - input[1].cols, 0, merge.cols, input[1].rows));
	//copy the images in these regions
	input[0].copyTo(roi21);
	merge.copyTo(roi22);

}


Mat PanoramicImage::getResult() {
	return output;
}

void equalizeImages(Mat *input0, Mat *input1, int tx) {
	//b1 = brightness in the first image
	//b2 = brightness in the second image
	//diff = cumulative difference of the brightness in the two images
	int b1 = 0, b2 = 0, diff = 0;
	//I focus only in a small portion of the overlapping area because there may be many ghost effect 
	int size = ceil(tx / 10); //common region between the two images to analyze (3 for lab and 10 for kitchen)
	
	for (int i = 0; i < (*input0).rows; i++) {
		for (int j = 0; j < size; j++) {
			b1 += (*input0).at<uchar>(i, (*input0).cols - tx + j);
			b2 += (*input1).at<uchar>(i, j);
			diff += abs((*input0).at<uchar>(i, (*input0).cols - tx + j) - (*input1).at<uchar>(i, j));
		}
	}

	int avg = ceil(diff / ((*input0).rows * size * 8));	//2 is a good value (Lab) and 8 (kitchen) 
	//equalization of the second image 
	//increase the brightness if the difference is positive otherwise we decrease it
	for (int i = 0; i < (*input1).rows; i++) {
		for (int j = 0; j < (*input1).cols; j++) {
			if ((*input1).at<uchar>(i, j) + avg > 255) {
				(*input1).at<uchar>(i, j) = 255;
			}
			else {
				if (b1 > b2)	(*input1).at<uchar>(i, j) = (*input1).at<uchar>(i, j) + avg;
				else
				{
					(*input1).at<uchar>(i, j) = (*input1).at<uchar>(i, j) - avg;
				}
			}
		}
	}

}


