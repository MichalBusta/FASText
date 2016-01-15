/*
 * detectors.h
 *
 *  Created on: Dec 15, 2015
 *      Author: Michal.Busta at gmail.com
 *
 *  Copyright (c) 2015, Michal Busta, Lukas Neumann, Jiri Matas.
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
#ifndef DETECTORS_H_
#define DETECTORS_H_

#include "FASTex.hpp"

namespace cmp
{

/*
 * Adapts a detector to partition the source image into a grid and detect
 * points in each cell.
 */
class CV_EXPORTS_W GridAdaptedFeatureDetector : public FASTextI
{
public:
    /*
     * detector            Detector that will be adapted.
     * maxTotalKeypoints   Maximum count of keypoints detected on the image. Only the strongest keypoints
     *                      will be keeped.
     * gridRows            Grid rows count.
     * gridCols            Grid column count.
     */
    CV_WRAP GridAdaptedFeatureDetector( const cv::Ptr<FASTextI>& detector,
                                        int maxTotalKeypoints=1000,
                                        int gridRows=8, int gridCols=8 );


    void setMaxTotalKeypoints(int maxTotalKeypoints){
    	this->maxTotalKeypoints = maxTotalKeypoints;
    }

    virtual bool isColorDetector(){
    	return detector->isColorDetector();
    }

    cv::Ptr<FASTextI> getDetector(){
    	return detector;
    }

    void setThreshold(long threshold){
    	detector->setThreshold(threshold);
    }

protected:
    virtual void detectImpl( const cv::Mat& image, std::vector<FastKeyPoint>& keypoints, const cv::Mat& mask=cv::Mat() ) const;

    virtual void segmentImpl( const cv::Mat& image, std::vector<FastKeyPoint>& keypoints,  std::unordered_multimap<int, std::pair<int, int> >& keypointsPixels, const cv::Mat& mask=cv::Mat() ) const;

    cv::Ptr<FASTextI> detector;
    int maxTotalKeypoints;
    int gridRows;
    int gridCols;
};

} /* namespace cmp */

#endif /* DETECTORS_H_ */
