/*
 * FTextLine.cpp
 *
 *  Created on: Feb 27, 2015
 *      Author: Michal Busta
 */
#include "FastTextLine.h"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#ifdef ANDROID_LOG
#	include <android/log.h>
#endif

#include "geometry.h"

namespace cmp
{

FTextLine::FTextLine()
{
	// TODO Auto-generated constructor stub
	minRect.size.width = 0;
	minRect.size.height = 0;
}

FTextLine::FTextLine(double theta) : theta(theta)
{
	// TODO Auto-generated constructor stub
	minRect.size.width = 0;
	minRect.size.height = 0;
}

FTextLine::~FTextLine()
{
	// TODO Auto-generated destructor stub
}

void FTextLine::addLetter(int letterId, std::vector<LetterCandidate>& letterCandidates)
{
	if( regionSet.find(letterId) != regionSet.end() )
		return;
	LetterCandidate& refR = letterCandidates[letterId];
	if( bbox.width == 0 )
		bbox = refR.bbox;
	else
		bbox |= refR.bbox;
	regionSet.insert(letterId);
	duplicates += refR.duplicates.size() + 1;
}

cv::Mat FTextLine::getNormalizedMask(const cv::Mat& image, std::vector<LetterCandidate>& letterCandidates, double scale)
{
	cv::RotatedRect rr = getMinAreaRect(letterCandidates);
	rr.center.x *= scale;
	rr.center.y *= scale;
	rr.size.width *= scale;
	rr.size.height *= scale;

	if(rr.size.height > rr.size.width)
	{
		std::swap(rr.size.height, rr.size.width);
		rr.angle += 90;
	}

	rext = rr;
	rext.size.width *= 1.5;
	rext.size.height *= 1.2;

	extbox = rext.boundingRect();
	extbox.x = MAX(extbox.x, 0);
	extbox.y = MAX(extbox.y, 0);
	if( (extbox.x + extbox.width) >= image.cols )
		extbox.width = image.cols - extbox.x;
	if( (extbox.y + extbox.height) >= image.rows )
		extbox.height = image.rows - extbox.y;

	cv::Mat tmp = image(extbox);
	cv::Point center = cv::Point(extbox.width / 2, extbox.height / 2);
	cv::Mat rot_mat = getRotationMatrix2D( cv::Point(extbox.width / 2, extbox.height / 2), rr.angle, 1 );

	rot_mat.at<double>(0,2) += rext.size.width/2.0 - center.x;
	rot_mat.at<double>(1,2) += rext.size.height/2.0 - center.y;
	//rot_matI.at<double>(0,2) -= rext.size.width/2.0 - center.x;
	//rot_matI.at<double>(1,2) -= rext.size.height/2.0 - center.y;

	/// Rotate the warped image
	cv::warpAffine( tmp, norm_line, rot_mat, rext.size );
	cv::invertAffineTransform(rot_mat, norm_mat);
	return norm_line;
}

cv::Mat FTextLine::createDebugImage(const cv::Mat& image, std::vector<LetterCandidate>& letterCandidates, bool color, bool drawRect)
{

	cv::Mat tmp = cv::Mat::zeros(image.rows, image.cols, CV_8UC1);
	for( std::set<int>::iterator it = regionSet.begin(); it != regionSet.end(); it++ )
	{
		LetterCandidate& ref1 =  letterCandidates[*it];

		cv::Rect rootRect = cv::Rect(ref1.bbox.x, ref1.bbox.y,  ref1.bbox.width, ref1.bbox.height);
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
	if( color )
	{
		cv::cvtColor(tmp, tmp, cv::COLOR_GRAY2BGR);
	}
	return tmp;
}

cv::RotatedRect FTextLine::getMinAreaRect(std::vector<LetterCandidate>& letterCandidates)
{
	if( minRect.size.width != 0)
		return minRect;
	std::vector<cv::Point> pointsAll;
	for( auto it = regionSet.begin(); it != regionSet.end(); it++ )
	{
		LetterCandidate& ref1 =  letterCandidates[*it];
		if( !ref1.isValid  )
			continue;
		pointsAll.insert(pointsAll.end(), ref1.cHullPoints.begin(), ref1.cHullPoints.end());
	}

#ifdef VERBOSE
	cv::Mat tmp = img.clone();
	if( tmp.channels() == 1)
		cv::cvtColor(tmp, tmp, cv::COLOR_GRAY2BGR);

	for( auto pt : pointsTop )
	{
		cv::circle(tmp, pt, 2, cv::Scalar(255, 0, 0));
	}
	for( auto pt : pointsBottom )
	{
		cv::circle(tmp, pt, 2, cv::Scalar(0, 255, 0));
	}
	for( auto pt : ref0.cHullPoints)
		cv::circle(tmp, pt, 2, cv::Scalar(0, 0, 255));
	for( auto pt : ref1.cHullPoints)
		cv::circle(tmp, pt, 4, cv::Scalar(0, 255, 255));
	cv::imshow("ts", tmp);
	cv::waitKey(0);
#endif

	minRect = minAreaRect( cv::Mat(pointsAll) );
	if(minRect.size.width < minRect.size.height){
		int swp = minRect.size.width;
		minRect.size.width = minRect.size.height;
		minRect.size.height = swp;
		minRect.angle += 90;

	}
	bbox = minRect.boundingRect();
	return minRect;
}

void FTextLine::splitHullLines(std::vector<LetterCandidate>& letterCandidates)
{
	cv::Point start(centerLine[2], centerLine[3] );
	cv::Point end(centerLine[2] + 100 * centerLine[0], centerLine[3] + 100 * centerLine[1]);
	pointsTop.clear();
	pointsBottom.clear();
	for( auto& rid : this->regionSet ){
		LetterCandidate& ref1 =  letterCandidates[rid];
		if(  !ref1.pointsScaled )
			ref1.scalePoints();
		cv::Point top(-1, -1);
		double distTop = 0;
		cv::Point bottom(-1, -1);
		double distBottom = 0;
		for( size_t i = 0; i < ref1.cHullPoints.size(); i++ ){
			int sign = 0;
			double d = distance_to_line(centerLine, ref1.cHullPoints[i], sign );
			if( sign > 0){
				if( d >  distTop ){
					top = ref1.cHullPoints[i];
					distTop = d;
				}
			}else{
				if( d >  distBottom ){
					bottom = ref1.cHullPoints[i];
					distBottom = d;
				}
			}
		}
		if( top.x != -1 ){
			this->pointsTop.push_back(top);

		}
		if( bottom.x != -1 ){
			this->pointsBottom.push_back(bottom);
		}
		if( top.x != -1 && bottom.x != -1  ){
			this->validRegSet.insert(rid);
		}
	}
	this->regionSet = this->validRegSet;
}

} /* namespace cmp */
