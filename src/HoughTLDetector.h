/*
 * HoughTLDetector.h
 *
 *  Created on: Jun 17, 2015
 *      Author: Michal Busta
 */
#ifndef HOUGHTLDETECTOR_H_
#define HOUGHTLDETECTOR_H_

#include <opencv2/core/core.hpp>

#include "segm/segmentation.h"

namespace cmp
{

typedef cv::Vec<int, 9> Vec9i;

struct LineGroup{

	LineGroup(double score, double rho, double theta, double scale) : score(score), rho(rho), theta(theta), scale(scale){

		isVertical = !(theta * 180 / M_PI > 45 && theta * 180 / M_PI < 135);
		density = 0;
	}

	LineGroup(std::vector<cv::Point>& pointsTop, std::vector<cv::Point> pointsBottom, float density) : pointsTop(pointsTop), pointsBottom(pointsBottom), density(density), score(0), rho(0), theta(0), scale(0){

	}

	float density;

	std::set<int> groupIds;
	std::set<int> regionIds;

	std::vector<cv::Point> pointsTop;
	std::vector<cv::Point> pointsBottom;

	double score;
	bool processed = false;
	double rho = 0;
	double theta = 0;
	double scale;
	/** the keypoints types */
	int type = 0;
	bool conflict = false;

	cv::Rect bbox;

	void sortIds(std::vector<LetterCandidate>& letterCandidates);

	bool isVertical = false;
};


/**
 * @class cmp::HoughTLDetector
 * 
 * @brief TODO brief description
 *
 * TODO type description
 */
class HoughTLDetector
{
public:

	void findTextLines(std::vector<LetterCandidate>& letterCandidates, const cv::Mat& originalImage, double letterHeight,  std::vector<LineGroup>& lineGroups, int type);

};

} /* namespace cmp */

#endif /* HOUGHTLDETECTOR_H_ */
