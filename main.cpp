#include <memory>
#include <iostream>
#include <opencv2/opencv.hpp>

#include "panoramicUtils.h"
#include "PanoramicImage.h"

using namespace std;
using namespace cv;

const string IMG_FORMAT = ".png";

int main(int argc, char* argv[]){

	//**********INPUT IMAGES***********
	String folderPath;
	cout << "Please enter the path of the folder with the images: " ;
	cin >> folderPath;
	double angle;
	cout << "Please enter the FOV used to acquire images: " ;
	cin >> angle;
	angle = angle / 2;

	vector<String> files;
	glob(folderPath, files);	//to find all the instances with the same pattern in a recursive way
	
	//*********PRE PROCESSING************
	//I've also added some lines of code in panoramicUtils.h to equalize the image
	vector<Mat> cylindricalImg;
	for (int i = 0; i < files.size(); i++) {
		Mat img = imread(files[i]);
		if (img.empty())	continue;	//we proceed only if successful

		//project images in a cylindric surface (it also converts images into grayscale images)
		cylindricalImg.push_back(PanoramicUtils::cylindricalProj(img, angle));

		/*
		//ADVANCED: equalization of the input image (before the tranformation into grayscale)
		Mat splImg[3];
		split(img, splImg);
		vector<Mat> eq_splImg;
		for (int i = 0; i < 3; i++) {
			Mat temp;
			equalizeHist(splImg[i], temp);
			eq_splImg.push_back(temp);
		}
		Mat img_equal;
		merge(eq_splImg, img_equal);
		cylindricalImg.push_back(PanoramicUtils::cylindricalProj(img_equal, angle));
		*/
	}

	//***********RESULT**************
	Mat panorama = cylindricalImg[0];	//vector to store the output panoramic img 
	for (int i = 1; i < cylindricalImg.size(); i++) {
		cout << "*******" << i << "*******" <<endl;
		//store the current panoramic image and the next image to add in a vector
		vector<Mat> outputTemp = { panorama, cylindricalImg[i] };
		//constructor
		PanoramicImage obj(outputTemp, panorama);
		obj.mergeImg();
		//get the result between the previous panorama and the current image
		panorama = obj.getResult();

		//imshow("Panoramic", panorama);
		//waitKey(0);
	}
	imwrite(folderPath + "/panoramic" + IMG_FORMAT, panorama);

}