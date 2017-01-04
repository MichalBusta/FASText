/*
 * TextLineDetector.h
 *
 *  Created on: Dec 17, 2014
 *      Author: Michal Busta
 */
#ifndef FASTTEXTLINEDETECTOR_H_
#define FASTTEXTLINEDETECTOR_H_

#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>

#include <unordered_map>
#include <unordered_set>
#include <list>

#include "segm/segmentation.h"

#include "CharClassifier.h"
#include "FastTextLine.h"

namespace cmp
{

/**
 * @class cmp::TextLineDetector
 * 
 * @brief TODO brief description
 *
 * TODO type description
 */
class FastTextLineDetector
{
public:
	FastTextLineDetector();
	virtual ~FastTextLineDetector();

	void findTextLines(const cv::Mat& image, std::vector<LetterCandidate>& letterCandidates, std::vector<double>& scales, std::vector<FTextLine>& textLines);

	int minHeight = 5;

};

} /* namespace cmp */

#endif /* FASTTEXTLINEDETECTOR_H_ */
