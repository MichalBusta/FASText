/*
 * TextLine.h
 *
 *  Created on: Feb 27, 2015
 *      Author: Michal Busta
 */
#ifndef FASTTEXTLINE_H_
#define FASTTEXTLINE_H_

#include <opencv2/core/core.hpp>

#include <list>

#include "Segmenter.h"

namespace cmp
{

inline void expandRoi(const cv::Mat& image, cv::Rect& bbox, int canvas, int& xOffset, int& yOffset, int&xOver, int& yOver )
{
	xOffset = 0;
	xOver = bbox.width;
	if(bbox.x >= canvas)
	{
		xOffset = canvas;
		bbox.x -= xOffset;
		bbox.width += canvas;
	}
	bbox.width += canvas;
	yOffset = 0;
	yOver = bbox.height;
	if(bbox.y >= canvas)
	{
		yOffset = canvas;
		bbox.y -= canvas;
		bbox.height += canvas;
	}
	bbox.height += canvas;

	if( bbox.x + bbox.width > image.cols)
	{
		bbox.width = image.cols - bbox.x;
	}
	if( bbox.y + bbox.height > image.rows)
	{
		bbox.height = image.rows - bbox.y;
	}
	xOver = bbox.width - xOver;
	yOver = bbox.height - yOver;
}

/**
 * @class cmp::TextLine
 * 
 * @brief TODO brief description
 *
 * TODO type description
 */
class FTextLine
{
public:
	FTextLine();

	FTextLine(double theta);

	virtual ~FTextLine();

	void addLetter(int letterId, std::vector<LetterCandidate>& letterCandidates);

	cv::Mat createDebugImage(const cv::Mat& image, std::vector<LetterCandidate>& letterCandidates, bool color, bool drawRect = false);

	cv::Mat getNormalizedMask(const cv::Mat& image, std::vector<LetterCandidate>& letterCandidates, double scale);

	cv::RotatedRect getMinAreaRect(std::vector<LetterCandidate>& letterCandidates);

	void splitHullLines(std::vector<LetterCandidate>& letterCandidates);

	cv::Rect bbox;

	int duplicates = 0;

	double angle = 0;

	std::set<int> regionSet;
	std::set<int> validRegSet;

	std::vector<cv::Point2f> centers;
	std::vector<cv::Point> pointsTop;
	std::vector<cv::Point> pointsBottom;

	cv::Vec4f centerLine;
	cv::Vec4f topLine;
	cv::Vec4f bottomLine;

	bool isSegmentable = true;

	cv::RotatedRect minRect;

	double theta = 0;
	float quality = 0;

	vector<vector<cv::Point> > contours;

	double height = 0;

	std::string text;
	std::vector<double> probs;
	std::vector<int> pos_start;
	std::vector<int> pos_end;

	cv::Mat norm_mat;
	float ocr_scale = 1.0f;
	cv::Mat norm_line;

	cv::RotatedRect rext; //rotated rectangle used for classification
	cv::Rect extbox;

	int type = 0;

	cv::Mat normImage;
};

} /* namespace cmp */

#endif /* FASTTEXTLINE_H_ */
