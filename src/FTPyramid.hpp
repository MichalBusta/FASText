/*
 * FTPyramid.hpp
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
#ifndef FASTTEXT_SRC_FTPYRAMID_HPP_
#define FASTTEXT_SRC_FTPYRAMID_HPP_

#include <vector>

#include <opencv2/features2d/features2d.hpp>

#include "FASTex.hpp"
#include "KeyPoints.h"

using namespace std;

namespace cmp{

/**
 * The FASText pyramid processing implementation
 */
class CV_EXPORTS_W FTPyr
{
public:
    CV_WRAP explicit FTPyr(int nfeatures = 500, float scaleFactor = 1.2f, int nlevels = 8, int edgeThreshold = 31, int keypointTypes = 2,
        int Kmin = 9, int Kmax = 11, bool color = false, bool erodeImages = false, bool createKeypointSegmenter = false);

    void detect( const cv::Mat& image, std::vector<FastKeyPoint>& keypoints, std::unordered_multimap<int, std::pair<int, int> >& keypointsPixels, const cv::Mat& mask = cv::Mat() )
    {
    	keypoints.clear();

    	if( image.empty() )
    		return;

    	CV_Assert( mask.empty() || (mask.type() == CV_8UC1 && mask.size() == image.size()) );

    	detectImpl( image, keypoints, keypointsPixels, mask );
    }

    /**
     * return the image pyramid
     */
    vector<cv::Mat>&  getImagePyramid()
	{
    	return imagePyramid;
	}

    double pyramidTime;
    double fastKeypointTime;

    double getScaleFactor(){
    	return scaleFactor;
    }

    int getEdgeThreshold(){
    	return edgeThreshold;
    }

    double getLevelScale(int level){
    	return scales[level];
    }

    vector<double>& getScales(){
    	return scales;
    }

    vector<int>& getScalesRef(){
    	return scalesRef;
    }

    vector<int>& getThresholds(){
    	return thresholds;
    }

protected:

    void computeFASText(vector<vector<FastKeyPoint> >& allKeypoints,
    		vector<int>& offsets,
			vector<std::unordered_multimap<int, std::pair<int, int> > >& keypointsPixels,
			int nfeatures, vector<int>& thresholds,
			vector<int>& keypointTypes);

    void detectImpl( const cv::Mat& image, vector<FastKeyPoint>& keypoints, std::unordered_multimap<int, std::pair<int, int> >& keypointsPixels, const cv::Mat& mask=cv::Mat() );

    CV_PROP_RW int nfeatures;
    CV_PROP_RW double scaleFactor;
    CV_PROP_RW int nlevels;
    CV_PROP_RW int edgeThreshold;

    int keypointTypes;
    int Kmin;
    int Kmax;

    vector<cv::Mat> imagePyramid;
    vector<cv::Mat> maskPyramid;
    vector<double> scales;
    vector<int> thresholds;
    vector<int> scaleKeypointTypes;
    vector<int> scalesRef;

    cv::Ptr<FASTextI> fastext;

    bool erodeImages;
};

}//namespace cmp

#endif /* FASTTEXT_SRC_FTPYRAMID_HPP_ */
