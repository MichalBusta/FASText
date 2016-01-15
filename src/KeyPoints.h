/*
 * KeyPoints.h
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
#ifndef KEYPOINTSFILTERC_H_
#define KEYPOINTSFILTERC_H_

#include <opencv2/features2d/features2d.hpp>
#include <unordered_map>

namespace cmp
{

class CV_EXPORTS_W_SIMPLE FastKeyPoint : public cv::KeyPoint
{
public:

	CV_WRAP FastKeyPoint() : cv::KeyPoint(), count(0), isMerged(false) {}
	//! the full constructor
	CV_WRAP FastKeyPoint(cv::Point2f _pt, float _size, float _angle=-1,
			float _response=0, int _octave=0, int _class_id=-1, uchar count = 0, bool isMerged = false) : cv::KeyPoint(_pt, _size, _angle, _response, _octave, _class_id), count(count), isMerged(isMerged) {}

	CV_WRAP FastKeyPoint(float x, float y, float _size, float _angle=-1,
	            float _response=0, int _octave=0, int _class_id=-1, uchar count = 0, bool isMerged = false): cv::KeyPoint(x, y, _size, _angle, _response, _octave, _class_id), count(count), isMerged(isMerged) {}

	cv::Point2f intensityIn;

	cv::Point2f intensityOut;

	uchar count;

	bool isMerged;

	uchar type = 0;

	uchar channel = 0;

	uchar maxima = 0;
};

/**
 * @class cmp::KeyPointsFilterC
 * 
 * @brief TODO brief description
 *
 * TODO type description
 */
class KeyPointsFilterC
{
public:
	KeyPointsFilterC();
	virtual ~KeyPointsFilterC();

	static void retainBest(std::vector<FastKeyPoint>& keypoints, std::unordered_multimap<int, std::pair<int, int> >& keypointPixels, int n_points);

	static void runByImageBorder( std::vector<FastKeyPoint>& keypoints, cv::Size imageSize, int borderSize );

	static void runByPixelsMask( std::vector<FastKeyPoint>& keypoints, const cv::Mat& mask );
};

} /* namespace cmp */

#endif /* KEYPOINTSFILTERC_H_ */
