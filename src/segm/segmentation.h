/*
 * segmentation.h
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
#ifndef FASTTEXT_SRC_SEGMENTATION_H_
#define FASTTEXT_SRC_SEGMENTATION_H_

#include <opencv2/core/core.hpp>
#include <set>
#include <iostream>
#include <assert.h>

#include "KeyPoints.h"
#include "flood_fill.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


#define MAX_COMP_SIZE 150

namespace cmp{

class StrokeDir {
public:
	cv::Point center;
	cv::Point direction;
	long threshold;
	int idx;

	StrokeDir(int idx, long threshold,  cv::Point center = cv::Point(), cv::Point direction = cv::Point()) : idx(idx), center(center), direction(direction), threshold(threshold)
	{

	}
};

class LetterCandidate{

public:

	LetterCandidate(cv::Mat mask = cv::Mat(), cv::Rect bbox = cv::Rect(), cv::Scalar cornerPixel = cv::Scalar(), cv::Scalar meanInk = cv::Scalar(),
			int area = 0, cmp::FastKeyPoint keyPoint = FastKeyPoint(), int projection = 0, float scaleFactor = 1.0,
			cv::Point centroid = cv::Point(), float angle = 0, int hullPoints = 0, float quality = 0):
			mask(mask), bbox(bbox), area(area), angle(angle), hullPoints(hullPoints), keyPoint(keyPoint), scaleFactor(scaleFactor), quality(quality),
			duplicate(-1), outputOrder(-1), pointsScaled(false), projection(projection), centroid(centroid), cornerPixel(cornerPixel), meanInk(meanInk) {

		merged = false;
		isValid = true;
	}

	bool contains(LetterCandidate& other );

	void setDuplicate( LetterCandidate& other, int refComp, int thisComp ){
		assert(refComp != -1);
		assert(this->duplicate == -1);
		this->duplicate = refComp;
		other.duplicates.push_back(thisComp);
		other.parents.insert(this->parents.begin(), this->parents.end());
		other.childs.insert(this->childs.begin(), this->childs.end());
		for(auto  it = neibours.begin(); it != neibours.end(); it++)
			other.addNeibour(*it);
	}

	void addNeibour(int refComp){
		if(neibours.find(refComp) == neibours.end())
		{
			neibours.insert(refComp);
		}
	}

	void addChild(int refComp, std::vector<LetterCandidate>& letterCandidates, int refComp2){

		for( auto pid :  parents)
		{
			letterCandidates[pid].childs.insert(refComp);
			//.addChild(refComp, letterCandidates, refComp2);
		}
		this->childs.insert(refComp);
		letterCandidates[refComp].parents.insert(refComp2);

	}

	cv::Mat createChildsImage(const cv::Mat& image, std::vector<LetterCandidate>& letterCandidates);

	cv::Mat generateStrokeWidthMap(std::vector<cmp::FastKeyPoint>& img1_keypoints, std::vector<double>& scales, std::unordered_map<int, std::vector<std::vector<cv::Ptr<StrokeDir> > > >& keypointStrokes);

	cv::Mat generateKeypointImg(const cv::Mat& img, std::vector<cmp::FastKeyPoint>& img1_keypoints, std::unordered_multimap<int, std::pair<int, int> >& keypointsPixels);

	bool isConvex();
	bool isRect();
	cv::Point getCentroid();
	cv::Point getConvexCentroid();

	float getStrokeAreaRatioP(){
		return strokeAreaRatio;
	}

	float getStrokeAreaRatio(std::vector<cmp::FastKeyPoint>& img1_keypoints, std::vector<double>& scales, std::unordered_map<int, std::vector<std::vector<cv::Ptr<StrokeDir> > > >& keypointStrokes);

	float getStrokeAreaRatio(std::vector<cmp::FastKeyPoint>& img1_keypoints, std::unordered_multimap<int, std::pair<int, int> >& keypointsPixels);

	inline void scalePoints(){
		if(pointsScaled)
			return;
		for( size_t i = 0; i < cHullPoints.size(); i++)
		{
			cHullPoints[i].x =  bbox.x + round(cHullPoints[i].x * scaleFactor);
			cHullPoints[i].y = bbox.y + round(cHullPoints[i].y * scaleFactor);
		}
		area *= scaleFactor * scaleFactor;
		rotatedRect.size.width *= scaleFactor;
		rotatedRect.size.height *= scaleFactor;
		pointsScaled = true;
	}

	cv::Mat mask;
	cv::Rect bbox;
	cv::RotatedRect rotatedRect;
	float angle;
	int hullPoints;
	int area;
	int strokeArea = 1;
	float convexHullArea = 0;
	float featuresArea = 0;
	float quality;
	bool isWord = false;
	float scaleFactor;
	bool merged;
	int groupAssigned = -1;
	cmp::FastKeyPoint keyPoint;
	std::vector<std::vector<cv::Point> > contours;
	std::vector<std::vector<cv::Point> > contoursAp;
	std::vector<cv::Vec4i> hierarchy;
	std::vector<cv::Point> cHullPoints;
	std::set<int> parents;
	std::set<int> childs;
	std::set<int> neibours;
	std::vector<int> duplicates;
	int duplicate;

	std::vector<int> leftGroups;
	std::vector<int> rightGroups;

	cv::Scalar intensityOut;
	cv::Scalar intensityInt;

	std::vector<int> keypointIds;

	//std::set<int> groups;
	int outputOrder;
	bool pointsScaled;
	int projection;
	bool isValid;

	cv::Scalar cornerPixel;
	cv::Scalar meanInk;

	int meanStrokeWidth = 0;

	std::vector<wchar_t> textHypotheses;
	std::vector<double> textHypothesesConfidences;

	float strokeAreaRatio = -1;

	int mergedKeypoints = 0;

	bool anglesFiltered = false;

	int gid = -1;

	cv::Mat featureVector;

private:

	cv::Point centroid;
	cv::Point convexCentroid;

};

}//namespace cmp

#endif /* FASTTEXT_SRC_SEGMENTATION_H_ */
