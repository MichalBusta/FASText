#pragma once
#include "../segm/segmentation.h"
#include "../Segmenter.h"

namespace cmp{

cv::Mat createCSERImage(std::vector<LetterCandidate*>& regions, const std::vector<cmp::FastKeyPoint>& keypoints, std::unordered_multimap<int, std::pair<int, int> >& keypointsPixels, const cv::Mat& sourceImage);


}//namespace cmp
