/*
 * Segmenter.h
 *
 *  Created on: Dec 15, 2015
 *      Author: Michal.Busta at gmail.com
 *
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
#ifndef SEGMENTER_H_
#define SEGMENTER_H_

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "KeyPoints.h"
#include "segm/segmentation.h"
#include "CharClassifier.h"

#include <unordered_map>
#include "FTPyramid.hpp"

namespace cmp
{

#define MIN_COMP_SIZE 12

/**
 * @class cmp::Segmenter
 * 
 * @brief The letter segmentation class
 *
 * Segments the letter components from detected keypoints
 */
class Segmenter
{
public:
	Segmenter(cv::Ptr<CharClassifier> charClassifier = cv::Ptr<CharClassifier> (new CvBoostCharClassifier()), int maxComponentSize = MAX_COMP_SIZE, int minCompSize = MIN_COMP_SIZE);
	virtual ~Segmenter();

	virtual void getLetterCandidates(cv::Mat& img, std::vector<cmp::FastKeyPoint>& img1_keypoints, std::unordered_multimap<int, std::pair<int, int> >& keypointsPixels, std::vector<cmp::LetterCandidate*>& letters, cv::Mat debugImage = cv::Mat(), int minHeight = 5) = 0;


	virtual cv::Mat getSegmenationMap(){
		return segmMap;
	}

	int64 getClassificationTime(){
		return classificationTime;
	};

	cv::Ptr<CharClassifier> getCharClassifier(){
		return charClassifier;
	}

	std::vector<LetterCandidate>& getLetterCandidates(){
		return letterCandidates;
	}

	bool segmentGrad = false;

	int minCompSize;

	int minHeight = 5;

	int minSizeSegmCount = 0;

	int segmentKeyPoints = 3;

	double strokeAreaTime = 0;

	int componentsCount = 0;

	int64 strokesTime = 0;

	std::unordered_map<int, std::vector<std::vector<cv::Ptr<StrokeDir> > > > keypointStrokes;

	int maxStrokeLength = 50;

protected:

	inline void classifyLetters(std::vector<cmp::FastKeyPoint>& img1_keypoints, std::unordered_multimap<int, std::pair<int, int> >& keypointsPixels, vector<double>& scales, std::vector<cmp::LetterCandidate*>& letters, cv::Mat debugImg = cv::Mat());

	int maxComponentSize;

	cv::Mat segmMap;
	cv::Mat idMap;

	std::vector<CvFFillSegment> buffer;
	std::vector<cv::Point> queue;

	cv::Ptr<CharClassifier> charClassifier;

	cv::Ptr<CharClassifier> wordClassifier;

	int64 classificationTime;


	std::vector<LetterCandidate> letterCandidates;

	bool dumpTrainingData = false;

};

struct SegmentOption{

	SegmentOption(int segmentationType, float scoreFactor): segmentationType(segmentationType), scoreFactor(scoreFactor){

	}

	int segmentationType;
	float scoreFactor;
};

class PyramidSegmenter : public Segmenter
{
public:
	PyramidSegmenter(cv::Ptr<cmp::FTPyr> ftDetector, cv::Ptr<CharClassifier> charClassifier = cv::Ptr<CharClassifier>(),
			int maxComponentSize = 2 * MAX_COMP_SIZE, int minCompSize = MIN_COMP_SIZE, float threshodFactor = 1.0,
			int delataIntResegment = 0, int segmentLevelOffset = 0) : Segmenter(charClassifier, maxComponentSize, minCompSize), ftDetector(ftDetector), threshodFactor(threshodFactor), delataIntResegment(delataIntResegment), segmentLevelOffset(segmentLevelOffset)
	{
		segmentOptions.push_back(SegmentOption(0, 1.0));
		//segmentOptions.push_back(SegmentOption(0, 0.4));
	};

	virtual void getLetterCandidates(cv::Mat& img, std::vector<cmp::FastKeyPoint>& img1_keypoints, std::unordered_multimap<int, std::pair<int, int> >& keypointsPixels, std::vector<cmp::LetterCandidate*>& letters, cv::Mat debugImage = cv::Mat(), int minHeight = 5);

	virtual void segmentStrokes(cv::Mat& img, std::vector<cmp::FastKeyPoint>& img1_keypoints, std::unordered_multimap<int, std::pair<int, int> >& keypointsPixels, std::vector<cmp::LetterCandidate*>& letters, cv::Mat debugImage = cv::Mat(), int minHeight = 5);

	virtual cv::Mat getSegmenationMap(){
		return segmPyramid[0];
	}

	static int getSegmIndex(cv::Mat& img, LetterCandidate& letter, int norm)
	{

		int index =  ((letter.bbox.y + letter.bbox.height / 2) / norm) * img.cols + (letter.bbox.x + letter.bbox.width / 2) / norm;
		return index;
	}

private:
	cv::Ptr<cmp::FTPyr> ftDetector;

	std::vector<cv::Mat> segmPyramid;
	std::vector<cv::Mat> idPyramid;
	std::vector<int*> pixelsOffset;

	float threshodFactor;

	std::vector<SegmentOption> segmentOptions;

	int delataIntResegment;

	int segmentLevelOffset;

};

} /* namespace cmp */

#endif /* SEGMENTER_H_ */
