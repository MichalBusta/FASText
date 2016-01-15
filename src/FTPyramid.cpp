/*
 * FTPyramid.cpp
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
#include "FTPyramid.hpp"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#include <iterator>

#include "TimeUtils.h"
#include "detectors.h"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

using namespace cv;

#define ADJUST_FEATURES 1

namespace cmp
{
//as in ORB
static inline float getScale(int level, double scaleFactor)
{
    return (float)std::pow(scaleFactor, (double)(level));
}

/**
 * Constructor
 */
FTPyr::FTPyr(int nfeatures, float scaleFactor, int nlevels, int edgeThreshold,
        int keypointTypes, int Kmin, int Kmax, bool color, bool erodeImages, bool createKeypointSegmenter) :
	pyramidTime(0), fastKeypointTime(0), nfeatures(nfeatures), scaleFactor(scaleFactor), nlevels(nlevels),
    edgeThreshold(edgeThreshold), keypointTypes(keypointTypes), Kmin(Kmin), Kmax(Kmax),
	erodeImages(erodeImages)
{
	fastext = cv::Ptr<FASTextI> (new GridAdaptedFeatureDetector (cv::Ptr<FASTextI> (new FASTextGray(edgeThreshold, true, keypointTypes, Kmin, Kmax))));
}

void FTPyr::computeFASText(vector<vector<FastKeyPoint> >& allKeypoints,
    		vector<int>& offsets,
			vector<std::unordered_multimap<int, std::pair<int, int> > >& keypointsPixels,
			int nfeatures, vector<int>& thresholds,
			vector<int>& keypointTypes)
{
    int nlevels = (int)imagePyramid.size();
    int levelsDecim = 1;
    for( size_t i = 1; i <  scales.size(); i++)
    {
    	if( scales[i - 1] != scales[i] )
    		levelsDecim++;
    }
    vector<int> nfeaturesPerLevel(nlevels);

    int totalFeatures = nfeatures;
    float factor = scales[1];
    if(factor == 1)
    	factor = scales[3];
#ifdef ADJUST_FEATURES
    for(size_t i = 0; i < imagePyramid.size(); i++  )
    {
    	if( imagePyramid[i].cols > 1024 || imagePyramid[i].rows > 1024 )
    	{
    		totalFeatures /= factor;
    	}else
    		break;
    }
#endif

    // fill the extractors and descriptors for the corresponding scales
    float ndesiredFeaturesPerScale = totalFeatures*(1 - factor)/(1 - (float) pow((double)factor, (double)levelsDecim));
    //float ndesiredFeaturesPerScale2 = totalFeatures / levelsDecim * 3;

    int sumFeatures = 0;
    for( int level = 0; level < nlevels-1; level++ )
    {
        nfeaturesPerLevel[level] = cvRound(ndesiredFeaturesPerScale);
        sumFeatures += nfeaturesPerLevel[level];
        if( scales[level] != scales[level + 1])
        {
        	ndesiredFeaturesPerScale *= factor;
        }
    }
    nfeaturesPerLevel[nlevels-1] = ndesiredFeaturesPerScale;
    //nfeaturesPerLevel[nlevels-1] = std::max(nfeatures - sumFeatures, 0);


    allKeypoints.resize(nlevels);
    keypointsPixels.resize(nlevels);
    offsets.resize(nlevels);
    int keypointsSize = 0;
    double prevsf = -1;
    for (int level = (nlevels - 1); level >= 0; level--)
    {
    	if(keypointsSize > totalFeatures )
    		break;

    	float sf = 1 / scales[level];

        int featuresNum = nfeaturesPerLevel[level];
        allKeypoints[level].reserve(featuresNum*3);

        GridAdaptedFeatureDetector* gaDetector = dynamic_cast<GridAdaptedFeatureDetector*>(&*fastext);
        if(gaDetector != NULL)
        {
        	gaDetector->setMaxTotalKeypoints(2 * featuresNum);
        	FASTextGray* grayDetector = dynamic_cast<FASTextGray*>(&*gaDetector->getDetector());
        	if(grayDetector != NULL)
        	{
        		grayDetector->setKeypointsTypes(keypointTypes[level]);
        	}
        }
        FASTextGray* grayDetector = dynamic_cast<FASTextGray*>(&*fastext);
        if(grayDetector != NULL)
        {
        	grayDetector->setKeypointsTypes(keypointTypes[level]);
        }

        vector<FastKeyPoint> & keypoints = allKeypoints[level];
        fastext->setThreshold( thresholds[level] );
        fastext->segment(imagePyramid[level], keypoints, keypointsPixels[level], maskPyramid[level]);
        offsets[level] = keypoints.size();

        if(keypointsPixels[level].size() == 0)
        	KeyPointsFilterC::retainBest(keypoints, keypointsPixels[level], featuresNum);
        if(prevsf != -1 && prevsf != sf )
        	keypointsSize += keypoints.size();
        prevsf = sf;

        // Set the level of the coordinates
        for (vector<FastKeyPoint>::iterator keypoint = keypoints.begin(),
             keypointEnd = keypoints.end(); keypoint != keypointEnd; keypoint++)
        {
            keypoint->octave = level;
            keypoint->size = sf;
            keypoint->intensityOut += keypoint->pt;
            keypoint->intensityIn += keypoint->pt;

        }
    }
}


void FTPyr::detectImpl( const Mat& image, vector<FastKeyPoint>& keypoints, std::unordered_multimap<int, std::pair<int, int> >& keypointsPixels, const Mat& mask)
{
	if(image.empty() )
		return;

	//ROI handling
	int border = 3;

	if( image.type() != CV_8UC1 && ! fastext->isColorDetector())
		cvtColor(image, image, COLOR_BGR2GRAY);

	int levelsNum = this->nlevels;
	if( levelsNum == -1) //the automatic levels decision
	{
		levelsNum = 1;
		int cols = MAX(image.cols, image.rows);
		while(cols > 30)
		{
			levelsNum++;
			cols /= this->scaleFactor;
		}
	}

	// Pre-compute the scale pyramids
	long long start = TimeUtils::MiliseconsNow();
	int levelsTotal =  levelsNum;
	if( erodeImages )
		levelsTotal += 2 * ( levelsNum );
	if(imagePyramid.size() == 0 || imagePyramid.size() != (size_t) levelsTotal)
	{
		imagePyramid.resize(levelsTotal);
		maskPyramid.resize(levelsTotal);
	}
	scales.clear();
	scalesRef.clear();
	scaleKeypointTypes.clear();
	thresholds.clear();
	int inLevelIndex = 0;
	bool hasErosion = false;
	for (int level = 0; level < levelsNum; ++level)
	{
		float scale = 1/getScale(level, scaleFactor);
		scales.push_back(scale);
		scalesRef.push_back(level);
		thresholds.push_back(this->edgeThreshold);
		Size sz(cvRound(image.cols*scale), cvRound(image.rows*scale));
		Size wholeSize(sz.width + border*2, sz.height + border*2);
		Mat temp;
		Mat tempErode;
		Mat tempDilate;
		Mat masktemp;
		if( !imagePyramid[inLevelIndex].empty() && imagePyramid[inLevelIndex].rows == wholeSize.height && imagePyramid[inLevelIndex].cols == wholeSize.width )
		{
			temp = imagePyramid[inLevelIndex];
		}else
		{
			temp = cv::Mat(wholeSize, image.type());
		}
		imagePyramid[inLevelIndex] = temp(Rect(border, border, sz.width, sz.height));

		if( !mask.empty() )
		{
			masktemp = Mat(wholeSize, mask.type());
			maskPyramid[inLevelIndex] = masktemp(Rect(border, border, sz.width, sz.height));
		}

		// pyramid
		if( level != 0 )
		{

			int step = 1;
			if(hasErosion)
				step = 3;
			resize(imagePyramid[inLevelIndex-step], imagePyramid[inLevelIndex], sz, 0, 0, INTER_LINEAR);
			scaleKeypointTypes.push_back(keypointTypes);
			copyMakeBorder(imagePyramid[inLevelIndex], temp, border, border, border, border,
					BORDER_REFLECT_101+BORDER_ISOLATED);
			if( erodeImages )
			{
				if( !imagePyramid[inLevelIndex + 1].empty() && imagePyramid[inLevelIndex + 1].rows == wholeSize.height && imagePyramid[inLevelIndex + 1].cols == wholeSize.width )
				{
					tempErode = imagePyramid[inLevelIndex + 1];
					tempDilate = imagePyramid[inLevelIndex + 2];
				}else
				{
					tempErode = cv::Mat(wholeSize, image.type());
					tempDilate = cv::Mat(wholeSize, image.type());
				}
				imagePyramid[inLevelIndex + 1] = tempErode(Rect(border, border, sz.width, sz.height));
				Mat element = getStructuringElement( MORPH_CROSS, Size( 3, 3 ), Point( 1, 1 ) );
				cv::erode( temp, tempErode, element );
				scaleKeypointTypes.push_back(1);
				thresholds.push_back(this->edgeThreshold);
				scalesRef.push_back(level);
				imagePyramid[inLevelIndex + 1] = tempErode(Rect(border, border, sz.width, sz.height));

				cv::dilate( temp, tempDilate, element );
				scaleKeypointTypes.push_back(2);
				thresholds.push_back(this->edgeThreshold);
				scalesRef.push_back(level);
				imagePyramid[inLevelIndex + 2] = tempDilate(Rect(border, border, sz.width, sz.height));
				scales.push_back(scale);
				scales.push_back(scale);
				hasErosion = true;
			}
			if (!mask.empty())
			{
				resize(maskPyramid[inLevelIndex-1], maskPyramid[inLevelIndex], sz, 0, 0, INTER_LINEAR);
				threshold(maskPyramid[inLevelIndex], maskPyramid[inLevelIndex], 254, 0, THRESH_TOZERO);
				if( erodeImages )
				{
					maskPyramid[inLevelIndex + 1] = maskPyramid[inLevelIndex];
					maskPyramid[inLevelIndex + 2] = maskPyramid[inLevelIndex];
				}
			}
			if (!mask.empty())
			{
				copyMakeBorder(maskPyramid[level], masktemp, border, border, border, border,
						BORDER_CONSTANT+BORDER_ISOLATED);
			}
			if( erodeImages )
			{
				inLevelIndex += 2;
			}
		}
		else
		{
			copyMakeBorder(image, temp, border, border, border, border,
					BORDER_REFLECT_101);
			scaleKeypointTypes.push_back(keypointTypes);
			if( erodeImages )
			{
				if( !imagePyramid[inLevelIndex + 1].empty() && imagePyramid[inLevelIndex + 1].rows == wholeSize.height && imagePyramid[inLevelIndex + 1].cols == wholeSize.width )
				{
					tempErode = imagePyramid[inLevelIndex + 1];
					tempDilate = imagePyramid[inLevelIndex + 2];
				}else
				{
					tempErode = cv::Mat(wholeSize, image.type());
					tempDilate = cv::Mat(wholeSize, image.type());
				}
				imagePyramid[inLevelIndex + 1] = tempErode(Rect(border, border, sz.width, sz.height));
				Mat element = getStructuringElement( MORPH_CROSS, Size( 3, 3 ), Point( 1, 1 ) );
				cv::erode( imagePyramid[inLevelIndex], imagePyramid[inLevelIndex + 1], element );
				scaleKeypointTypes.push_back(1);
				scalesRef.push_back(level);
				thresholds.push_back(this->edgeThreshold);
				copyMakeBorder(imagePyramid[inLevelIndex + 1], tempErode, border, border, border, border, BORDER_REFLECT_101+BORDER_ISOLATED);
				imagePyramid[inLevelIndex + 1] = tempErode(Rect(border, border, sz.width, sz.height));
				imagePyramid[inLevelIndex + 2] = tempDilate(Rect(border, border, sz.width, sz.height));
				cv::dilate( imagePyramid[inLevelIndex], imagePyramid[inLevelIndex + 2], element );
				scaleKeypointTypes.push_back(2);
				scalesRef.push_back(level);
				thresholds.push_back(this->edgeThreshold);
				copyMakeBorder(imagePyramid[inLevelIndex + 2], tempDilate, border, border, border, border, BORDER_REFLECT_101+BORDER_ISOLATED);
				imagePyramid[inLevelIndex + 2] = tempDilate(Rect(border, border, sz.width, sz.height));
				scales.push_back(scale);
				scales.push_back(scale);
				hasErosion = true;

				inLevelIndex += 2;
			}
			if( !mask.empty() )
				copyMakeBorder(mask, masktemp, border, border, border, border,
						BORDER_CONSTANT+BORDER_ISOLATED);
		}
		inLevelIndex++;
	}
	pyramidTime = TimeUtils::MiliseconsNow() - start;


	// Pre-compute the keypoints (we keep the best over all scales, so this has to be done beforehand
	vector < vector<FastKeyPoint> > allKeypoints;
	vector <int> offsets;
	std::vector<std::unordered_multimap<int, std::pair<int, int> > > allKeypointsPixels;

	start = TimeUtils::MiliseconsNow();

	computeFASText(allKeypoints, offsets, allKeypointsPixels,
			nfeatures, thresholds, scaleKeypointTypes);

	fastKeypointTime = TimeUtils::MiliseconsNow() - start;
	// make sure we have the right number of keypoints keypoints
	/*vector<KeyPoint> temp;

	        for (int level = 0; level < n_levels; ++level)
	        {
	            vector<KeyPoint>& keypoints = all_keypoints[level];
	            temp.insert(temp.end(), keypoints.begin(), keypoints.end());
	            keypoints.clear();
	        }

	        KeyPoint::retainBest(temp, n_features_);

	        for (vector<KeyPoint>::iterator keypoint = temp.begin(),
	             keypoint_end = temp.end(); keypoint != keypoint_end; ++keypoint)
	            all_keypoints[keypoint->octave].push_back(*keypoint);*/

	Mat descriptors;
	vector<Point> pattern;

	keypoints.clear();
	keypointsPixels.clear();
	int offset = 0;
	for (size_t level = 0; level < allKeypoints.size(); ++level)
	{
		// Get the features and compute their orientation
		vector<FastKeyPoint>& kps = allKeypoints[level];
		// Copy to the output data
		bool chekcKpId =  allKeypointsPixels[level].size() > 0;
		if (level != 0)
		{
			float scale = 1 / scales[level];
			int keypointNo = 0;
			for (vector<FastKeyPoint>::iterator keypoint = kps.begin(),
					keypointEnd = kps.end(); keypoint != keypointEnd; ++keypoint)
			{
				if( keypoint->class_id == keypointNo || !chekcKpId  )
				{
					keypoint->pt *= scale;
					keypoint->intensityOut *= scale;
					keypoint->intensityIn *= scale;
					keypoint->class_id += offset;
					keypoints.push_back(*keypoint);
				}
				keypointNo++;
			}

		}else{
			int keypointNo = 0;
			for (vector<FastKeyPoint>::iterator keypoint = kps.begin(),
					keypointEnd = kps.end(); keypoint != keypointEnd; ++keypoint)
			{
				if( keypoint->class_id == keypointNo || !chekcKpId  )
				{
					keypoints.push_back(*keypoint);
				}
				keypointNo++;
			}
		}

		std::unordered_multimap<int, std::pair<int, int> >& keypointsPixelsSub = allKeypointsPixels[level];
		for (std::unordered_multimap<int, std::pair<int, int> >::iterator itr = keypointsPixelsSub.begin(); itr != keypointsPixelsSub.end(); itr++)
		{
			assert(itr->second.first < image.cols);
			assert(itr->second.second < image.rows);
			keypointsPixels.insert( std::pair<int, std::pair<int, int> >( itr->first + offset,  std::pair<int, int>(itr->second.first, itr->second.second)));
		}
		offset += offsets[level];
	}
}

}//namespace cmp
