/*
 * detectors.cpp
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
#include "detectors.h"

namespace cmp
{

class GridAdaptedFeatureDetectorInvoker : public cv::ParallelLoopBody
{
private:
    int gridRows_, gridCols_;
    int maxPerCell_;
    std::vector<FastKeyPoint>& keypoints_;
    std::unordered_multimap<int, std::pair<int, int> >& keypointsPixels;
    const cv::Mat& image_;
    const cv::Mat& mask_;
    const cv::Ptr<FASTextI>& detector_;
    cv::Mutex* kptLock_;

    GridAdaptedFeatureDetectorInvoker& operator=(const GridAdaptedFeatureDetectorInvoker&); // to quiet MSVC

public:

    GridAdaptedFeatureDetectorInvoker(const cv::Ptr<FASTextI>& detector, const cv::Mat& image, const cv::Mat& mask,
                                      std::vector<FastKeyPoint>& keypoints, std::unordered_multimap<int, std::pair<int, int> >& keypointsPixels,
									  int maxPerCell, int gridRows, int gridCols,
                                      cv::Mutex* kptLock)
        : gridRows_(gridRows), gridCols_(gridCols), maxPerCell_(maxPerCell),
          keypoints_(keypoints), keypointsPixels(keypointsPixels), image_(image), mask_(mask), detector_(detector),
          kptLock_(kptLock)
    {

    }

    void operator() (const cv::Range& range) const
    {
        for (int i = range.start; i < range.end; ++i)
        {
            int celly = i / gridCols_;
            int cellx = i - celly * gridCols_;

            cv::Range row_range((celly*image_.rows)/gridRows_, ((celly+1)*image_.rows)/gridRows_);
            cv::Range col_range((cellx*image_.cols)/gridCols_, ((cellx+1)*image_.cols)/gridCols_);
            if(row_range.end < image_.rows - 5)
            {
            	row_range.end += 3;
            }
            if(col_range.end <  image_.cols - 5)
            {
            	col_range.end += 3;
            }

            cv::Mat sub_image = image_(row_range, col_range);
            cv::Mat sub_mask;
            if (!mask_.empty()) sub_mask = mask_(row_range, col_range);

            std::vector<FastKeyPoint> sub_keypoints;
            sub_keypoints.reserve(2 * maxPerCell_);
            std::unordered_multimap<int, std::pair<int, int> > keypointsPixelsSub;
            detector_->segment( sub_image, sub_keypoints, keypointsPixelsSub, sub_mask );
            if( keypointsPixelsSub.size() == 0 )
            	KeyPointsFilterC::retainBest(sub_keypoints, keypointsPixelsSub, 2 * maxPerCell_);

            std::vector<FastKeyPoint>::iterator it = sub_keypoints.begin(), end = sub_keypoints.end();
            for( ; it != end; ++it )
            {
                it->pt.x += col_range.start;
                it->pt.y += row_range.start;
            }

            {
            	cv::AutoLock join_keypoints(*kptLock_);
            	int offset = keypoints_.size();
            	if( keypointsPixelsSub.size() > 0 )
            	{
            		std::vector<FastKeyPoint>::iterator it = sub_keypoints.begin(), end = sub_keypoints.end();
            		for( ; it != end; ++it )
            		{
            			it->class_id += offset;
            		}
            	}

            	keypoints_.insert( keypoints_.end(), sub_keypoints.begin(), sub_keypoints.end() );

            	for (std::unordered_multimap<int, std::pair<int, int> >::iterator itr = keypointsPixelsSub.begin(); itr != keypointsPixelsSub.end(); itr++) {
            		keypointsPixels.insert( std::pair<int, std::pair<int, int> >( itr->first + offset,  std::pair<int, int>(itr->second.first + col_range.start, itr->second.second + row_range.start)));
            	}
            }
        }
    }
};

GridAdaptedFeatureDetector::GridAdaptedFeatureDetector( const cv::Ptr<FASTextI>& detector, int maxTotalKeypoints, int gridRows, int gridCols): detector(detector), maxTotalKeypoints(maxTotalKeypoints), gridRows(gridRows), gridCols(gridCols)
{

}

void GridAdaptedFeatureDetector::detectImpl( const cv::Mat& image, std::vector<FastKeyPoint>& keypoints, const cv::Mat& mask ) const
{
    if (image.empty() )
    {
        keypoints.clear();
        return;
    }

    if(MIN(image.cols, image.rows) < 128 )
    {
    	detector->detect( image, keypoints, mask );
    }else
    {

    	keypoints.reserve(2 * maxTotalKeypoints);
    	int maxPerCell = (maxTotalKeypoints / (gridRows * gridCols));

    	cv::Mutex kptLock;
    	std::unordered_multimap<int, std::pair<int, int> > keypointsPixels;
    	GridAdaptedFeatureDetectorInvoker body(detector, image, mask, keypoints, keypointsPixels, maxPerCell, gridRows, gridCols, &kptLock);
    	//body(cv::Range(0, gridRows * gridCols));
    	cv::parallel_for_(cv::Range(0, gridRows * gridCols), body);
    	//KeyPointsFilterC::retainBest(keypoints, maxTotalKeypoints);
    }
}

void GridAdaptedFeatureDetector::segmentImpl( const cv::Mat& image, std::vector<FastKeyPoint>& keypoints,  std::unordered_multimap<int, std::pair<int, int> >& keypointsPixels, const cv::Mat& mask) const
{
	if (image.empty() )
	{
		keypoints.clear();
		return;
	}

	if(MIN(image.cols, image.rows) < 128 )
	{
		detector->segment( image, keypoints, keypointsPixels, mask );
	}else
	{

		keypoints.reserve(2 * maxTotalKeypoints);
		int maxPerCell = (maxTotalKeypoints / (gridRows * gridCols));

		cv::Mutex kptLock;
		GridAdaptedFeatureDetectorInvoker body(detector, image, mask, keypoints, keypointsPixels, maxPerCell, gridRows, gridCols, &kptLock);
		//body(cv::Range(0, gridRows * gridCols));
		cv::parallel_for_(cv::Range(0, gridRows * gridCols), body);
		//KeyPointsFilterC::retainBest(keypoints, maxTotalKeypoints);
	}
}

} /* namespace cmp */
