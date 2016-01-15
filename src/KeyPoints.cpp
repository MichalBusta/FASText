/*
 * KeyPoints.cpp
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
#include "KeyPoints.h"

namespace cmp
{

KeyPointsFilterC::KeyPointsFilterC()
{
	// TODO Auto-generated constructor stub

}

KeyPointsFilterC::~KeyPointsFilterC()
{
	// TODO Auto-generated destructor stub
}


struct KeypointResponseGreaterThanThreshold
{
    KeypointResponseGreaterThanThreshold(float _value) :
    value(_value)
    {
    }
    inline bool operator()(const FastKeyPoint& kpt) const
    {
        return kpt.response >= value;
    }
    float value;
};

struct KeypointResponseGreater
{
    inline bool operator()(const FastKeyPoint& kp1, const FastKeyPoint& kp2) const
    {
        return kp1.response > kp2.response;
    }
};

// takes keypoints and culls them by the response
void KeyPointsFilterC::retainBest(std::vector<FastKeyPoint>& keypoints, std::unordered_multimap<int, std::pair<int, int> >& keypointPixels,  int n_points)
{
    //this is only necessary if the keypoints size is greater than the number of desired points.
    if( n_points >= 0 && keypoints.size() > (size_t)n_points )
    {
        if (n_points==0)
        {
            keypoints.clear();
            return;
        }
        if(keypointPixels.size() == 0)
        {
        	std::sort(keypoints.begin(),  keypoints.end(), KeypointResponseGreater());

        	//this is the boundary response, and in the case of FAST may be ambigous
        	float ambiguous_response = keypoints[n_points - 1].response;
        	//use std::partition to grab all of the keypoints with the boundary response.
        	std::vector<FastKeyPoint>::iterator new_end =
        			std::partition(keypoints.begin() + n_points - 1, keypoints.end(),
        					KeypointResponseGreaterThanThreshold(ambiguous_response));
        	//resize the keypoints, given this new end point. nth_element and partition reordered the points inplace
        	keypoints.erase(new_end, keypoints.end());
        }else{
        	std::pair <std::unordered_multimap<int,std::pair<int, int> >::iterator, std::unordered_multimap<int,std::pair<int, int>>::iterator> ret, ret2;

        	sort(keypoints.begin(), keypoints.end(),
        	    [&](const FastKeyPoint & a, const FastKeyPoint & b)
        	{
        		ret = keypointPixels.equal_range(a.class_id);
        		int dist1 = abs(20 - std::distance(ret.first, ret.second));
        		ret2 = keypointPixels.equal_range(b.class_id);
        		int dist2 = abs(20 - std::distance(ret2.first, ret2.second));
        	    return dist1 < dist2;
        	});
        	keypoints.resize( n_points );
        }
    }
}

struct RoiPredicate
{
    RoiPredicate( const cv::Rect& _r ) : r(_r)
    {}

    bool operator()( const FastKeyPoint& keyPt ) const
    {
        return !r.contains( keyPt.pt );
    }

    cv::Rect r;
};

void KeyPointsFilterC::runByImageBorder( std::vector<FastKeyPoint>& keypoints, cv::Size imageSize, int borderSize )
{
    if( borderSize > 0)
    {
        if (imageSize.height <= borderSize * 2 || imageSize.width <= borderSize * 2)
            keypoints.clear();
        else
            keypoints.erase( std::remove_if(keypoints.begin(), keypoints.end(),
                                       RoiPredicate(cv::Rect(cv::Point(borderSize, borderSize),
                                                         cv::Point(imageSize.width - borderSize, imageSize.height - borderSize)))),
                             keypoints.end() );
    }
}

class MaskPredicate
{
public:
    MaskPredicate( const cv::Mat& _mask ) : mask(_mask) {}
    bool operator() (const FastKeyPoint& key_pt) const
    {
        return mask.at<uchar>( (int)(key_pt.pt.y + 0.5f), (int)(key_pt.pt.x + 0.5f) ) == 0;
    }

private:
    const cv::Mat mask;
    MaskPredicate& operator=(const MaskPredicate&);
};

void KeyPointsFilterC::runByPixelsMask( std::vector<FastKeyPoint>& keypoints, const cv::Mat& mask )
{
    if( mask.empty() )
        return;

    keypoints.erase(std::remove_if(keypoints.begin(), keypoints.end(), MaskPredicate(mask)), keypoints.end());
}

} /* namespace cmp */

