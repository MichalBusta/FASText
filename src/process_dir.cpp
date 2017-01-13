/*
 * test_processing.cpp
 *
 *  Created on: Dec 15, 2015
 *      Author: Michal.Busta at gmail.com
 */

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "segm/segmentation.h"

#include <iostream>
#include <sstream>

#include <fstream>
#include <unordered_map>

#include "FTPyramid.hpp"

#include "IOUtils.h"
#include "TimeUtils.h"
#include "CharClassifier.h"
#include "Segmenter.h"

#include "FastTextLineDetector.h"

#define VERBOSE 1

using namespace cmp;

int main(int argc, char **argv)
{


	//cv::GaussianBlur(gray, gray, cv::Size(3, 3), 0);
	float scaleFactor = 1.6f;
	int nlevels = -1;
	int edgeThreshold = 12;
	int keypointTypes = 3;
	int kMin = 9;
	int kMax = 11;
	bool color = false;

	cv::Ptr<cmp::FTPyr> ftDetector = cv::Ptr<cmp::FTPyr> (new cmp::FTPyr(3000, scaleFactor, nlevels, edgeThreshold, keypointTypes, kMin, kMax, color, false, false));
	cv::Ptr<cmp::CharClassifier> charClassifier = cv::Ptr<cmp::CharClassifier> (new cmp::CvBoostCharClassifier("cvBoostChar.xml"));
	cv::Ptr<cmp::Segmenter> segmenter = cv::Ptr<cmp::Segmenter> (new cmp::PyramidSegmenter(ftDetector, charClassifier));

	FastTextLineDetector textLineDetector;

	long long segmentationTime = 0;
	long long lettersTotal = 0;
	long long keypointsTime = 0;
	long long keypointsTotal = 0;
	long long clsTime = 0;

	std::vector<std::string> files = cmp::IOUtils::GetFilesInDirectory( argv[1], "*.png", true );
	std::vector<std::string> files2 = cmp::IOUtils::GetFilesInDirectory( argv[1], "*.jpg", true );
	files.insert(files.end(), files2.begin(), files2.end());
	std::vector<cv::Point> queue;
	std::string outDir = "/tmp/processDir/";
	for(size_t x = 0; x < files.size(); x++)
	{
		std::cout << "Processing: " << files[x] << std::endl;
		cv::Mat img = cv::imread(files[x]);
		if(img.empty())
			continue;

		cv::Mat gray;
		cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

		cv::Mat strokes;
		std::string imgName = outDir;
		imgName += cmp::IOUtils::Basename(files[x]) + ".png";
		cv::imwrite(imgName, strokes);
		
		cv::Mat procImg = img;

		std::vector<cmp::FastKeyPoint> img1_keypoints;
		std::unordered_multimap<int, std::pair<int, int> > keypointsPixels;
		std::vector<cmp::LetterCandidate*> letters;
		if( color || true)
		{
			long long start = TimeUtils::MiliseconsNow();
			if(color){
				ftDetector->detect(img, img1_keypoints, keypointsPixels);
			}else{
				ftDetector->detect(gray, img1_keypoints, keypointsPixels);
			}
			std::cout << "Detected keypoints: " << img1_keypoints.size() << std::endl;
			keypointsTime +=  TimeUtils::MiliseconsNow() - start;
			keypointsTotal += img1_keypoints.size();

			start = TimeUtils::MiliseconsNow();
			//cv::Mat imgOut;
			//cv::imshow("gray", gray);

			segmenter->getLetterCandidates( gray, img1_keypoints, keypointsPixels, letters );
			std::cout << "Segmented: " << letters.size() << "/" <<  segmenter->getLetterCandidates().size() << std::endl;
			lettersTotal += letters.size();
			segmentationTime += TimeUtils::MiliseconsNow() - start;
			clsTime += segmenter->getClassificationTime();

			std::vector<FTextLine> textLines;
			textLineDetector.findTextLines(gray, segmenter->getLetterCandidates(), ftDetector->getScales(), textLines);
#ifdef VERBOSE
			cv::Mat lineImage = img.clone();
			for(size_t i = 0; i < textLines.size(); i++){
				FTextLine& line = textLines[i];
				cv::RotatedRect rr = line.getMinAreaRect(segmenter->getLetterCandidates());

				cv::Scalar c(255, 0, 0);
				cv::Point2f rect_points[4]; rr.points( rect_points );
				cv::line(lineImage, rect_points[0], rect_points[1], c, 1);
				cv::line(lineImage, rect_points[1], rect_points[2], c, 1);
				cv::line(lineImage, rect_points[2], rect_points[3], c, 1);
				cv::line(lineImage, rect_points[3], rect_points[0], c, 1);

			}
			cv::imshow("textLines", lineImage);
			cv::waitKey(0);
#endif

		}
	}
	std::cout << "Total keypoints time: " << keypointsTime << std::endl;
	std::cout << "Total segmentation time: " << segmentationTime << std::endl;
	std::cout << "Cls time: " << clsTime / (cv::getTickFrequency()) * 1000 << std::endl;

	std::cout << "Keypoints total: " << keypointsTotal << std::endl;
	std::cout << "Letters total: " << lettersTotal << std::endl;

}
