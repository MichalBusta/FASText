/*
 * segmentaion.cpp
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
#include "segmentation.h"

#include <unordered_map>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core_c.h>
#include <opencv2/imgproc/types_c.h>

#include <unordered_map>

#include "FASTex.hpp"

#define BORDER_SIZE 5

using namespace std;

namespace cmp{

cv::Point LetterCandidate::getCentroid()
{
	if(centroid.x == 0)
	{
		cv::Moments m = cv::moments(this->mask, true);
		centroid = cv::Point((int) cvRound(bbox.x + ( m.m10 / m.m00 ) * this->scaleFactor ), (int) cvRound(bbox.y + (m.m01 / m.m00 ) * this->scaleFactor));
	}
	return centroid;
}

cv::Point LetterCandidate::getConvexCentroid()
{
	if(convexCentroid.x == 0)
	{
		if(this->cHullPoints.size() > 3){
			cv::Moments m = cv::moments(this->cHullPoints);
			convexCentroid = cv::Point((int) cvRound(m.m10 / m.m00), (int) cvRound(m.m01 / m.m00));
			assert(convexCentroid.x > 0 && convexCentroid.y > 0);
		}else{
			convexCentroid = cv::Point(bbox.x + bbox.width / 2, bbox.y + bbox.height / 2);
		}

	}
	return convexCentroid;
}

float LetterCandidate::getStrokeAreaRatio(std::vector<cmp::FastKeyPoint>& img1_keypoints, std::vector<double>& scales, std::unordered_map<int, std::vector<std::vector<cv::Ptr<StrokeDir> > > >& keypointStrokes)
{
	if( strokeAreaRatio != -1)
		return strokeAreaRatio;
	cv::Mat tmp = cv::Mat::zeros(mask.rows, mask.cols, CV_8UC1);
	for( auto kpid : keypointIds )
	{
		cmp::FastKeyPoint& kp = img1_keypoints[kpid];
		if( kp.type != this->keyPoint.type )
			continue;
		if( abs(kp.octave - this->keyPoint.octave) > 2 )
			continue;
		int radius = 2 / scales[kp.octave] / this->scaleFactor;
		double sf = 1.0 /  scales[kp.octave];
		cv::Scalar color( 255, 255, 255 );
		if( kp.count == 5)
		{
			cv::circle(tmp, cv::Point((kp.pt.x - bbox.x) / this->scaleFactor, (kp.pt.y - bbox.y) / this->scaleFactor), radius, color, -1);
		}
		else
		{
			std::vector<std::vector<cv::Ptr<StrokeDir> > >& storkeDirections = keypointStrokes[kpid];
			for( auto strokes : storkeDirections  )
			{
				int thickness = kp.count / scales[kp.octave] / this->scaleFactor;
				thickness = MAX(1, thickness);
				thickness = MIN(255, thickness);
				for( auto sd : strokes)
				{

					cv::line( tmp,
							cv::Point(roundf((sd->center.x * sf - bbox.x) / this->scaleFactor), roundf((sd->center.y * sf - bbox.y) / this->scaleFactor)) ,
							cv::Point(roundf((sd->direction.x * sf - bbox.x) / this->scaleFactor), roundf((sd->direction.y * sf - bbox.y) / this->scaleFactor)),
							color, thickness );
				}
			}
		}

	}

	cv::Mat strokeArea;
	cv::bitwise_and( tmp, mask, strokeArea);
	int pixels = countNonZero(strokeArea);
	strokeAreaRatio = pixels / (float) countNonZero(mask);
	/*
	cv::imshow("mask", mask);
	cv::imshow("tmp", tmp);
	cv::imshow("strokeArea", strokeArea);
	cv::waitKey(0);
	*/
	return strokeAreaRatio;

}

float LetterCandidate::getStrokeAreaRatio(std::vector<cmp::FastKeyPoint>& img1_keypoints, std::unordered_multimap<int, std::pair<int, int> >& keypointsPixels)
{
	if( strokeAreaRatio != -1)
			return strokeAreaRatio;
	std::pair <std::unordered_multimap<int,std::pair<int, int> >::iterator, std::unordered_multimap<int,std::pair<int, int>>::iterator> ret;
	strokeArea = 0;
	for( auto kpid : keypointIds )
	{
		cmp::FastKeyPoint& kp = img1_keypoints[kpid];
		if( kp.octave !=  this->keyPoint.octave)
			continue;
		ret = keypointsPixels.equal_range(kp.class_id);
		strokeArea += std::distance(ret.first, ret.second);
	}
	strokeAreaRatio = strokeArea / (float) this->area;
	return strokeAreaRatio;
}


bool LetterCandidate::contains(LetterCandidate& other )
{
	return  bbox.contains(other.bbox.tl()) && bbox.contains(other.bbox.br());
}

// helper function:
// finds a cosine of angle between vectors
// from pt0->pt1 and from pt0->pt2
static double angleP( cv::Point pt1, cv::Point pt2, cv::Point pt0 )
{
    double dx1 = pt1.x - pt0.x;
    double dy1 = pt1.y - pt0.y;
    double dx2 = pt2.x - pt0.x;
    double dy2 = pt2.y - pt0.y;
    return (dx1*dx2 + dy1*dy2)/sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
}

bool LetterCandidate::isConvex()
{
	return cv::isContourConvex(contoursAp[0]);
}


bool LetterCandidate::isRect()
{
	if( contoursAp[0].size() == 4)
	{
		double maxCosine = 0;
		for( int j = 2; j < 5; j++ )
		{
			// find the maximum cosine of the angle between joint edges
			double cosine = fabs(angleP(contoursAp[0][j%4], contoursAp[0][j-2], contoursAp[0][j-1]));
			maxCosine = MAX(maxCosine, cosine);
		}

		// if cosines of all angles are small
		// (all angles are ~90 degree) then write quandrange
		// vertices to resultant sequence
		if( maxCosine < 0.3 )
			return true;
	}
	return false;
}

cv::Mat LetterCandidate::createChildsImage(const cv::Mat& image, std::vector<LetterCandidate>& letterCandidates)
{
	cv::Mat tmp = cv::Mat::zeros(image.rows, image.cols, CV_8UC1);


	for( std::set<int>::iterator it = childs.begin(); it != childs.end(); it++ )
	{
		LetterCandidate& ref1 =  letterCandidates[*it];
		cv::Rect rootRect = cv::Rect(ref1.bbox.x, ref1.bbox.y,  ref1.bbox.width, ref1.bbox.height);
		cv::rectangle(tmp, rootRect, cv::Scalar(255, 0, 0));
		cv::Mat mask = ref1.mask;
		if( ref1.scaleFactor != 1)
		{
			cv::resize(mask, mask, cv::Size(ref1.bbox.width, ref1.bbox.height));
		}
		if( (rootRect.x + rootRect.width) >= tmp.cols )
			continue;
		if( (rootRect.y + rootRect.height) >= tmp.rows )
			continue;
		if( rootRect.width != mask.cols || rootRect.height != mask.rows )
			continue;
		cv::bitwise_or(tmp(rootRect), mask, tmp(rootRect));
		for(auto itj : ref1.duplicates)
		{
			LetterCandidate& refd =  letterCandidates[itj];
			rootRect = cv::Rect(refd.bbox.x, refd.bbox.y,  refd.bbox.width, refd.bbox.height);
			mask = refd.mask;
			if( refd.scaleFactor != 1)
			{
				cv::resize(mask, mask, cv::Size(ref1.bbox.width, ref1.bbox.height));
			}
			if( (rootRect.x + rootRect.width) >= tmp.cols )
				continue;
			if( (rootRect.y + rootRect.height) >= tmp.rows )
				continue;
			if( rootRect.width != mask.cols || rootRect.height != mask.rows )
				continue;
			cv::bitwise_or(tmp(rootRect), mask, tmp(rootRect));
		}
	}
	tmp = ~tmp;
	cv::cvtColor(tmp, tmp, cv::COLOR_GRAY2BGR);
	cv::rectangle(tmp, this->bbox, cv::Scalar(0, 255, 0));
	if( tmp.cols > 1024)
		cv::resize(tmp, tmp, cv::Size(tmp.cols / 2, tmp.rows / 2));
	return tmp;
}

cv::Mat LetterCandidate::generateStrokeWidthMap(std::vector<cmp::FastKeyPoint>& img1_keypoints, std::vector<double>& scales, std::unordered_map<int, std::vector<std::vector<cv::Ptr<StrokeDir> > > >& keypointStrokes)
{
	cv::Mat tmp = this->mask.clone();
	cv::cvtColor(tmp, tmp, cv::COLOR_GRAY2BGR);

	for( auto kpid : keypointIds )
	{
		cmp::FastKeyPoint& kp = img1_keypoints[kpid];
		if( abs(kp.octave - this->keyPoint.octave) > 2 )
			continue;
		int radius = 2 / scales[kp.octave] / this->scaleFactor;
		double sf = 1.0 /  scales[kp.octave];
		cv::Scalar color( 0, 0, 255 );
		if( kp.count == 5)
		{
			color = cv::Scalar(255, 0, 0);
			cv::circle(tmp, cv::Point(roundf((kp.pt.x - bbox.x) / this->scaleFactor), roundf((kp.pt.y - bbox.y) / this->scaleFactor)), radius, color, -1);
		}
		else
		{
			std::vector<std::vector<cv::Ptr<StrokeDir> > >& storkeDirections = keypointStrokes[kpid];
			for( auto strokes : storkeDirections  )
			{
				int thickness = kp.count / scales[kp.octave] / this->scaleFactor;
				thickness = MAX(1, thickness);
				thickness = MIN(255, thickness);
				for( auto sd : strokes)
				{

					cv::line( tmp, cv::Point(roundf((sd->center.x * sf - bbox.x) / this->scaleFactor), roundf((sd->center.y * sf - bbox.y) / this->scaleFactor)) ,
							cv::Point(roundf((sd->direction.x * sf - bbox.x) / this->scaleFactor), roundf((sd->direction.y * sf - bbox.y) / this->scaleFactor)), color, thickness );
				}
			}
			//cv::circle(tmp, cv::Point((kp.pt.x - bbox.x) / this->scaleFactor, (kp.pt.y - bbox.y) / this->scaleFactor), 1, cv::Scalar(0, 255, 0), -1);
			//cv::circle(tmp, cv::Point((kp.intensityMin.x - bbox.x) / this->scaleFactor, (kp.intensityMin.y - bbox.y) / this->scaleFactor), 1, cv::Scalar(0, 255, 0), -1);
		}

	}
	return tmp;
}

cv::Mat LetterCandidate::generateKeypointImg(const cv::Mat& img, std::vector<cmp::FastKeyPoint>& img1_keypoints, std::unordered_multimap<int, std::pair<int, int> >& keypointsPixels)
{
	cv::Mat tmp = this->mask.clone();
	cv::cvtColor(tmp, tmp, cv::COLOR_GRAY2BGR);

	std::pair <std::unordered_multimap<int,std::pair<int, int> >::iterator, std::unordered_multimap<int,std::pair<int, int>>::iterator> ret;
	cv::Scalar color(0, 255, 0);
	for( auto kpid : keypointIds )
	{
		cmp::FastKeyPoint& kp = img1_keypoints[kpid];
		if( kp.octave !=  this->keyPoint.octave)
			continue;
		ret = keypointsPixels.equal_range(kp.class_id);
		for (std::unordered_multimap<int,std::pair<int, int> >::iterator it=ret.first; it!=ret.second; it++)
		{
			assert(it->second.first * this->scaleFactor < img.cols);
			assert(it->second.second * this->scaleFactor <= (img.rows + 5));
			cv::circle(tmp, cv::Point((it->second.first - this->bbox.x / this->scaleFactor) , (it->second.second - bbox.y / this->scaleFactor)), 1, color);
		}
	}
	return tmp;
}

float getMinAnglesDiff(float& angle1, float& angle2)
{
	float dif0 = fabs(angle1 - angle2);
	if( angle1 > 45 && angle2 < 45 )
	{
		float angle1N = angle1 - 180;
		float dif1 = fabs(angle1 - angle2);
		if( angle2 < -45 )
		{
			float angle2N = angle2 + 180;
			float dif2 = fabs(angle1N - angle2N);
			if(dif2 < dif0 && dif2 < dif1)
			{
				angle2 = angle2N;
				angle1 = angle1N;
				return dif2;
			}
		}
		if(dif1 < dif0)
		{
			angle1 = angle1N;
			return dif1;
		}

	}else if( angle2 > 45 && angle1 < 45 )
	{
		float angle2N = angle2 - 180;
		float dif1 = fabs(angle1 - angle2N);

		if( angle1 < -45 )
		{
			float angle1N = angle1 + 180;
			float dif2 = fabs(angle1N - angle2N);
			if(dif2 < dif0 && dif2 < dif1)
			{
				angle2 = angle2N;
				angle1 = angle1N;
				return dif2;
			}
		}
		if(dif1 < dif0)
		{
			angle2 = angle2N;
			return dif1;
		}

	}
	return dif0;
}

double constrainAngle(double x)
{
    x = fmod(x + 180,360);
    if (x < 0)
        x += 360;
    return x - 180;
}

static void transformPoint(const cv::Mat& affMat, const cv::Point& input, cv::Rect& bbox, cv::Rect& maxRect, cv::Point& output)
{
	int x = input.x + bbox.x - maxRect.x;
	int y = input.y + bbox.y - maxRect.y;

	float ytr = x * affMat.at<double>(0, 1) + y * affMat.at<double>(1, 1) + affMat.at<double>(2, 1);
	float xtr = x * affMat.at<double>(0, 0) + y * affMat.at<double>(1, 0) + affMat.at<double>(2, 0);
	output.x = xtr;
	output.y = ytr;
}

}//namespace cmp

