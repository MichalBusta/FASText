/*
 * CharClassifier.h
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
#ifndef CHARCLASSIFIER_H_
#define CHARCLASSIFIER_H_

#include <opencv2/ml/ml.hpp>

#include "segm/segmentation.h"

namespace cmp
{

/**
 * @class cmp::CharClassifier
 * 
 * @brief The character classifier interface
 *
 */
class CharClassifier
{
public:
	CharClassifier();
	virtual ~CharClassifier();

	virtual bool classifyLetter(LetterCandidate& letter, cv::Mat debugImage = cv::Mat());

	virtual double isWord(LetterCandidate& letter, cv::Mat debugImage = cv::Mat());

	virtual bool predictProbability(LetterCandidate& letter, double& probability, cv::Mat debugImage  = cv::Mat() ){
		probability = 0.5;
		return classifyLetter(letter, debugImage);
	}

	static bool extractLineFeatures(LetterCandidate& letter);

	int64 classificationTime;
};

void extractFeatureVect(cv::Mat& maskO, std::vector<float>& featureVector, LetterCandidate& letter);
void extractFeatureVectNoSsp(cv::Mat& maskO, std::vector<float>& featureVector);

/**
 *
 */
class CvBoostCharClassifier : public CharClassifier
{
public:

	CvBoostCharClassifier() : CharClassifier(){

	};

	CvBoostCharClassifier(const char* modelFile) : CharClassifier(){
		std::string sname = modelFile;
		load(sname);
	};

	virtual ~CvBoostCharClassifier(){

	};

	virtual bool classifyLetter(LetterCandidate& letter, cv::Mat debugImage = cv::Mat() );

	virtual double isWord(LetterCandidate& letter, cv::Mat debugImage = cv::Mat());

	virtual bool predictProbability(LetterCandidate& letter, double& probability, cv::Mat debugImag  = cv::Mat() );

	//loads model file
	void load(std::string& modelFile);

private:
	// Trained AdaBoost classifier
#ifdef OPENCV_24
	cv::Ptr<CvBoost> classifier;
#else
	cv::Ptr<cv::ml::Boost> classifier;
#endif
};

} /* namespace cmp */

#endif /* CHARCLASSIFIER_H_ */
