/*
 * CharClassifier.cpp
 *
 *  Created on: Dec 15, 2015
 *      Author: Michal.Busta at gmail.com
 *
 * Copyright (c) 2015, Michal Busta, Lukas Neumann, Jiri Matas.
 *
 *   This program is free software; you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation; either version 2 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 * Based on:
 *
 * FASText: Efficient Unconstrained Scene Text Detector,Busta M., Neumann L., Matas J.: ICCV 2015.
 * Machine learning for high-speed corner detection, E. Rosten and T. Drummond, ECCV 2006
 */

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/highgui/highgui.hpp>

#include "CharClassifier.h"

using namespace std;

namespace cmp
{

CharClassifier::CharClassifier() : classificationTime(0)
{
	// TODO Auto-generated constructor stub

}

CharClassifier::~CharClassifier()
{
	// TODO Auto-generated destructor stub
}

bool CharClassifier::classifyLetter(LetterCandidate& letter, cv::Mat debugImage)
{
	return true;
}

double CharClassifier::isWord(LetterCandidate& letter, cv::Mat debugImage)
{
	return 0;
}

bool CharClassifier::extractLineFeatures(LetterCandidate& letter)
{
	cv::Mat mask;
	if(letter.contours.size() == 0)
	{
		cv::copyMakeBorder(letter.mask, mask, 1, 1, 1, 1, cv::BORDER_CONSTANT);
		cv::findContours(mask, letter.contours, letter.hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
		if(letter.contours.size() == 0)
			return false;
	}

	double epsion = (mask.cols + mask.rows) / 2.0 * 0.022;

	letter.contoursAp.resize(letter.contours.size());
	float outerContourArea = 0;
	for(size_t i = 0; i < letter.contours.size(); i++)
	{

		cv::approxPolyDP(letter.contours[i], letter.contoursAp[i], epsion, true);
		if(letter.hierarchy[i][3] < 0) //contour
		{
			float area = cv::contourArea(letter.contours[i]);
			letter.featuresArea += area;
			if(area >= outerContourArea)
			{
				outerContourArea = area;
				if( letter.cHullPoints.size() == 0 )
					cv::convexHull(letter.contours[i], letter.cHullPoints);
				if( letter.cHullPoints.size() == 0 )
					letter.cHullPoints = letter.contours[i];
				letter.convexHullArea = cv::contourArea(letter.cHullPoints);
			}
		}else{ //hole
			float area = cv::contourArea(letter.contours[i]);
			letter.featuresArea -= area;
		}
	}
	letter.meanStrokeWidth = 2;

	return true;
}

void extractFeatureVect(cv::Mat& maskO, std::vector<float>& featureVector, LetterCandidate& letter)
{
	featureVector.reserve(6);

	if(letter.contours.size() == 0)
	{
		cv::Mat mask;
		cv::copyMakeBorder(maskO, mask, 1, 1, 1, 1, cv::BORDER_CONSTANT);
		findContours(mask, letter.contours, letter.hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE);
	}
	float holesArea = 0;
	float outerContourArea = 0;
	float perimeter = 0;

	double al = cv::arcLength(letter.contours[0], true);
	double epsion = al * 0.022;
	letter.contoursAp.resize(letter.contours.size());
	for(size_t i = 0; i < letter.contours.size(); i++)
	{

		cv::approxPolyDP(letter.contours[i], letter.contoursAp[i], epsion, true);
		if(letter.hierarchy[i][3] < 0) //contour
		{
			float area = cv::contourArea(letter.contours[i]);
			letter.featuresArea += area;
			if(area > outerContourArea)
			{
				outerContourArea = area;
				perimeter = cv::arcLength(letter.contours[i], false);
				if( letter.cHullPoints.size() == 0 )
					cv::convexHull(letter.contours[i], letter.cHullPoints);
				letter.convexHullArea = cv::contourArea(letter.cHullPoints);
			}
		}else{ //hole
			float area = cv::contourArea(letter.contours[i]);
			holesArea += area;
			letter.featuresArea -= area;
		}
	}

	featureVector.push_back(letter.getStrokeAreaRatioP() / letter.area);
	if( perimeter == 0)
		featureVector.push_back(0);
	else
		featureVector.push_back(letter.featuresArea / (float) (perimeter * perimeter));
	if(outerContourArea == 0)
		featureVector.push_back(0);
	else
		featureVector.push_back(letter.convexHullArea / (float) outerContourArea);
	featureVector.push_back((float) holesArea / (float) letter.area);
	//std::cout << outerContourArea << std::endl;
	if(letter.cHullPoints.size() == 0 || perimeter == 0)
	{
		featureVector.push_back(0);
		featureVector.push_back(MIN(maskO.rows, maskO.cols) / (float) MAX(maskO.rows, maskO.cols));
	}
	else
	{
		featureVector.push_back(cv::arcLength(letter.cHullPoints, true) / perimeter);
		cv::RotatedRect rotatedRect = cv::minAreaRect(letter.cHullPoints);
		cv::Point2f vertices[4];
		rotatedRect.points(vertices);

		float width = rotatedRect.size.width;
		float height = rotatedRect.size.height;
		featureVector.push_back(MIN(width, height) / MAX(width, height));
	}
}

void extractFeatureVectNoSsp(cv::Mat& maskO, std::vector<float>& featureVector)
{
	std::vector<std::vector<cv::Point> > contours;
	std::vector<cv::Vec4i> hierarchy;
	if(contours.size() == 0)
	{
		cv::Mat mask;
		cv::copyMakeBorder(maskO, mask, 1, 1, 1, 1, cv::BORDER_CONSTANT);
		findContours(mask, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE);
	}
	float holesArea = 0;
	float outerContourArea = 0;
	float perimeter = 0;

	double al = cv::arcLength(contours[0], true);
	double epsion = al * 0.022;
	std::vector<std::vector<cv::Point> > contoursAp;
	contoursAp.resize(contours.size());
	std::vector<cv::Point> cHullPoints;
	double featuresArea = 0;
	double convexHullArea = 0;
	for(size_t i = 0; i < contours.size(); i++)
	{

		cv::approxPolyDP(contours[i], contoursAp[i], epsion, true);
		if(hierarchy[i][3] < 0) //contour
		{
			float area = cv::contourArea(contours[i]);
			featuresArea += area;
			if(area >= outerContourArea)
			{
				outerContourArea = area;
				perimeter = cv::arcLength(contours[i], false);
				if( cHullPoints.size() == 0 )
					cv::convexHull(contours[i], cHullPoints);
				convexHullArea = cv::contourArea(cHullPoints);
			}
		}else{ //hole
			float area = cv::contourArea(contours[i]);
			holesArea += area;
			featuresArea -= area;
		}
	}

	if( perimeter == 0)
		featureVector.push_back(0);
	else
		featureVector.push_back(featuresArea / (float) (perimeter * perimeter));
	if(outerContourArea == 0)
		featureVector.push_back(0);
	else
		featureVector.push_back(convexHullArea / (float) outerContourArea);
	featureVector.push_back((float) holesArea / (float) cv::countNonZero(maskO));
	//std::cout << outerContourArea << std::endl;
	if(cHullPoints.size() == 0 || perimeter == 0)
	{
		featureVector.push_back(0);
		featureVector.push_back(MIN(maskO.rows, maskO.cols) / (float) MAX(maskO.rows, maskO.cols));
	}
	else
	{
		featureVector.push_back(cv::arcLength(cHullPoints, true) / perimeter);
		cv::RotatedRect rotatedRect = cv::minAreaRect(cHullPoints);
		cv::Point2f vertices[4];
		rotatedRect.points(vertices);

		float width = cv::norm(vertices[0] - vertices[1]);
		float height = cv::norm(vertices[1] - vertices[2]);
		featureVector.push_back(MIN(width, height) / MAX(width, height));
	}
}


static void extractCharFeatures(cv::Mat& maskO, cv::Mat& featureVector, LetterCandidate& letter)
{
	cv::Mat mask;
	cv::copyMakeBorder(maskO, mask, 1, 1, 1, 1, cv::BORDER_CONSTANT);
	featureVector = cv::Mat::zeros(1, 6, CV_32F);
	float *pFeatureVector = featureVector.ptr<float>(0);

	if(letter.contours.size() == 0)
	{
		findContours(mask, letter.contours, letter.hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE);
	}
	float holesArea = 0;
	float outerContourArea = 0;
	float perimeter = 0;
	double al = cv::arcLength(letter.contours[0], true);
	double epsion = al * 0.022;
	letter.contoursAp.resize(1);
	for(size_t i = 0; i < letter.contours.size(); i++)
	{
		if( i == 0)
			cv::approxPolyDP(letter.contours[i], letter.contoursAp[i], epsion, true);
		if(letter.hierarchy[i][3] < 0) //contour
		{
			float area = cv::contourArea(letter.contours[i]);
			area = MAX(area, al);
			letter.featuresArea += area;
			if(area >= outerContourArea)
			{
				outerContourArea = area;
				perimeter = cv::arcLength(letter.contours[i], false);
				if( letter.cHullPoints.size() == 0 )
					cv::convexHull(letter.contours[i], letter.cHullPoints);
				if(letter.cHullPoints.size() == 0)
				{
					letter.cHullPoints = letter.contours[i];
				}
				letter.convexHullArea = cv::contourArea(letter.cHullPoints);
			}
				}else{ //hole
					float area = cv::contourArea(letter.contours[i]);
					holesArea += area;
					letter.featuresArea -= area;
				}
	}

	double cHullLength = cv::arcLength(letter.cHullPoints, true);
	letter.convexHullArea = MAX(letter.convexHullArea, cHullLength);

	*(pFeatureVector) = letter.getStrokeAreaRatioP() / letter.area;
	pFeatureVector++;
	if( perimeter == 0)
		*(pFeatureVector) = 0;
	else
		*(pFeatureVector) = letter.featuresArea / (float) (perimeter * perimeter);
	pFeatureVector++;
	if(outerContourArea == 0)
		*(pFeatureVector) = 0;
	else
		*(pFeatureVector) = letter.convexHullArea / (float) outerContourArea;
	pFeatureVector++;
	*(pFeatureVector) = (float) holesArea / (float) letter.area;
	pFeatureVector++;
	//std::cout << outerContourArea << std::endl;

	if(letter.cHullPoints.size() == 0 || perimeter == 0)
	{
		*(pFeatureVector) = 0;
		pFeatureVector++;
		*(pFeatureVector) = MIN(mask.rows, mask.cols) / (float) MAX(mask.rows, mask.cols);
	}
	else
	{
		*(pFeatureVector) = cHullLength / perimeter;
		pFeatureVector++;
		cv::RotatedRect rotatedRect = cv::minAreaRect(letter.cHullPoints);
		cv::Point2f vertices[4];
		rotatedRect.points(vertices);

		float width = rotatedRect.size.width;
		float height = rotatedRect.size.height;
		*(pFeatureVector) = MIN(width, height) / MAX(width, height);
	}
}

bool CvBoostCharClassifier::classifyLetter(LetterCandidate& letter, cv::Mat debugImag)
{
	double probability;
	bool val = predictProbability(letter, probability, debugImag  );
	letter.quality = probability;
	return val || letter.quality > 0.2;
}

double CvBoostCharClassifier::isWord(LetterCandidate& letter, cv::Mat debugImage)
{
	if( letter.featureVector.empty() )
		extractCharFeatures(letter.mask, letter.featureVector, letter);

	cv::Mat featureVectorMulti;
	cv::Mat cols = cv::Mat::zeros(1, 1, CV_32F);
	cv::hconcat(letter.featureVector, cols, featureVectorMulti);

	featureVectorMulti.at<float>(0, 6) = letter.keypointIds.size();

	int64 startTime = cv::getTickCount();
#ifdef OPENCV_24
	float sum = classifier->predict(featureVectorMulti, cv::Mat(), cv::Range::all(), false, true);
	double probability = 1.0f / (1.0f + exp (-sum) );
#else

	float votes = classifier->predict( featureVectorMulti, cv::noArray(), cv::ml::DTrees::PREDICT_SUM | cv::ml::StatModel::RAW_OUTPUT);
	double probability = (double)1-(double)1/(1+exp(-2*votes));
#endif
	classificationTime += cv::getTickCount() - startTime;

	return probability;
}

bool CvBoostCharClassifier::predictProbability(LetterCandidate& letter, double& probability, cv::Mat debugImag  )
{
	if( letter.featureVector.empty() )
		extractCharFeatures(letter.mask, letter.featureVector, letter);
	int64 startTime = cv::getTickCount();
#ifdef OPENCV_24
	float sum = classifier->predict(letter.featureVector, cv::Mat(), cv::Range::all(), false, true);

	int cls_idx = sum >= 0;
	const int* cmap = classifier->get_data()->cat_map->data.i;
	const int* cofs = classifier->get_data()->cat_ofs->data.i;
	const int* vtype = classifier->get_data()->var_type->data.i;

	int val = (float) cmap[cofs[vtype[classifier->get_data()->var_count]] + cls_idx];
	probability = 1.0f / (1.0f + exp (-sum) );
#else
	float votes = classifier->predict( letter.featureVector, cv::noArray(), cv::ml::DTrees::PREDICT_SUM | cv::ml::StatModel::RAW_OUTPUT);
	probability = (double)1-(double)1/(1+exp(-2*votes));
	int val = probability > 0.5;
#endif
	classificationTime += cv::getTickCount() - startTime;

	return val;
}

void CvBoostCharClassifier::load(std::string& modelFile){
	std::cout << "Loading CharCls Model from: " << modelFile << std::endl;
#ifdef OPENCV_24
	classifier = new CvBoost();
	classifier->load(modelFile.c_str(), "classifier");
#else
	classifier = cv::ml::StatModel::load<cv::ml::Boost>( modelFile.c_str()/*, "classifier" */);
#endif
}

} /* namespace cmp */
