/*
 * TextLineDetector.cpp
 *
 *  Created on: Dec 17, 2014
 *      Author: Michal Busta
 */
#include "FastTextLineDetector.h"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>
#include <unordered_set>
#include <mutex>

#include "geometry.h"
#include "HoughTLDetector.h"

using namespace cv;

//#define VERBOSE 1

namespace cmp
{

FastTextLineDetector::FastTextLineDetector()
{

}

FastTextLineDetector::~FastTextLineDetector()
{
	// TODO Auto-generated destructor stub
}

void FastTextLineDetector::findTextLines(const cv::Mat& image, std::vector<LetterCandidate>& letterCandidates, std::vector<double>& scales, std::vector<FTextLine>& textLines)
{
	textLines.clear();

	int index = -1;
	for (std::vector<LetterCandidate>::iterator kv = letterCandidates.begin(); kv != letterCandidates.end(); kv++)
	{
		index += 1;
		if(kv->duplicate != -1)
			continue;
		if(kv->cHullPoints.size() == 0)
		{
			CharClassifier::extractLineFeatures(*kv);
		}
		kv->scalePoints();
	}
	std::vector<LineGroup> hLines;
	double letterHeight = MIN(image.rows, image.cols);
	do{
		HoughTLDetector houghTlDetector;
		houghTlDetector.findTextLines(letterCandidates, image, letterHeight, hLines, 0);
		houghTlDetector.findTextLines(letterCandidates, image, letterHeight, hLines, 1);
		letterHeight /= 2;
	}while( letterHeight > 5 );

	std::vector<LineGroup> hLinesFinal;
	hLinesFinal = hLines;

	//hLinesFinal = hLines;
	std::vector<FTextLine> initialTextLines;
	initialTextLines.reserve(hLinesFinal.size());
	for( size_t i = 0; i < hLinesFinal.size(); i++ )
	{
		LineGroup& group = hLinesFinal[i];
		if( group.conflict)
		{
			continue;
		}
		initialTextLines.push_back(FTextLine(group.theta - M_PI / 2));
		for( auto& rid : group.regionIds ){
			initialTextLines.back().addLetter(rid, letterCandidates);
		}

		initialTextLines.back().getMinAreaRect(letterCandidates);
	}

	for( auto& tl : initialTextLines )
	{
		if(!tl.isSegmentable)
			continue;
		std::vector<cv::Point> centerLine;
		centerLine.reserve(tl.regionSet.size());
		if( tl.regionSet.size() < 3 ) {
			tl.regionSet.clear();
			tl.isSegmentable = false;
			continue;
		}
		for( auto regId : tl.regionSet ){
			centerLine.push_back(letterCandidates[regId].getConvexCentroid());
		}

		cv::fitLine(centerLine, tl.centerLine, CV_DIST_L2, 0,0.01,0.01);
		tl.splitHullLines(letterCandidates);
		if( tl.pointsTop.size() > 3 )
			cv::fitLine(tl.pointsTop, tl.topLine, CV_DIST_L2, 0,0.01,0.01);
		if( tl.pointsBottom.size() > 3 )
			cv::fitLine(tl.pointsBottom, tl.bottomLine, CV_DIST_L2, 0,0.01,0.01);
		double angle2 = atan2(tl.centerLine[1], tl.centerLine[0]);
		tl.angle = angle2;
		tl.minRect.size.width = 0;
		cv::RotatedRect rr = tl.getMinAreaRect(letterCandidates);
		tl.height  = MIN(rr.size.width, rr.size.height);
#ifdef VERBOSE
		cv::Mat tmp = image.clone();
		cvtColor(tmp, tmp, CV_GRAY2BGR);
		cv::line(tmp, cv::Point(tl.centerLine.val[2], tl.centerLine.val[3]), cv::Point(tl.centerLine.val[2] + 100 * tl.centerLine.val[0], tl.centerLine.val[3] + 100 * tl.centerLine.val[1]), cv::Scalar(255, 255, 255) );
		for(auto& center : centerLine  )
			cv::circle(tmp, center, 5, cv::Scalar(255, 255, 255), 2);
		for(auto& center : tl.pointsTop  )
			cv::circle(tmp, center, 5, cv::Scalar(0, 255, 0), 2);
		cv::imshow("tmp", tmp);
		cv::waitKey(0);
#endif

	}

	for( size_t i = 0; i <  initialTextLines.size(); i++)
	{
		if(!initialTextLines[i].isSegmentable)
			continue;
		textLines.push_back(initialTextLines[i]);
	}
}


} /* namespace cmp */
