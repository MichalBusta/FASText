/*
 * FASTex.hpp
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
#ifndef CMP_FAST_HPP_
#define CMP_FAST_HPP_

#include "KeyPoints.h"
#include <unordered_map>
#include <vector>

namespace cmp{

inline long ColourDistance(const uchar* e1, const uchar* e2)
{
	int ur1 = e1[2];
	int ur2 = e2[2];
	long rmean = (  ur1 + ur2 ) / 2;
	long r = ur1 - ur2;
	int ug1 = e1[1];
	int ug2 = e2[1];
	long g = ug1 - ug2;
	int ub1 = e1[0];
	int ub2 = e2[0];
	long b = ub1 - ub2;
	return (((512+rmean)*r*r)>>8) + 4*g*g + (((767-rmean)*b*b)>>8);
}

inline long ColourDistanceMAX(const uchar* e1, const uchar* e2, uchar& sign)
{
	int d1 = e1[0] - (int) e2[0];
	int ad1 = abs(d1);
	int d2 = e1[0] - (int) e2[0];
	int ad2 = abs(d2);
	int d3 = e1[0] - (int) e2[0];
	int ad3 = abs(d3);
	if(ad1 > ad2)
	{
		if(ad1 > ad3)
		{
			sign = d1 > 0;
			return ad1;
		}
		sign = d3 > 0;
		return ad3;
	}else
	{
		if(ad2 > ad3)
		{
			sign = d2 > 0;
			return ad2;
		}
		sign = d3 > 0;
		return ad3;
	}
}

inline long ColourDistanceVec(const cv::Vec3b& e1, const cv::Vec3b& e2)
{
	int ur1 = e1[2];
	int ur2 = e2[2];
	long rmean = (  ur1 + ur2 ) / 2;
	long r = ur1 - ur2;
	int ug1 = e1[1];
	int ug2 = e2[1];
	long g = ug1 - ug2;
	int ub1 = e1[0];
	int ub2 = e2[0];
	long b = ub1 - ub2;
	return (((512+rmean)*r*r)>>8) + 4*g*g + (((767-rmean)*b*b)>>8);
}

inline long ColourDistanceGray(const uchar& e1, const uchar& e2)
{
	return e2 - e1;
}

template<int channel>
inline long ColourDistanceRGB(const cv::Vec3b& e1, const cv::Vec3b& e2)
{
	return e2[channel] - e1[channel];
}

template<int channel>
inline long ColourDistanceRGBP(const uchar& e1, const uchar& e2)
{
	return (&e2)[channel] - (&e1)[channel];
}

template<int channel>
inline long ColourDistanceRGBIP(const uchar& e1, const uchar& e2)
{
	return (&e1)[channel] - (&e2)[channel];
}

inline long ColourDistanceGrayABS(const uchar* e1, const uchar* e2)
{
	return abs(((int) *e2) - *e1);
}

inline long ColourDistanceGrayP(const uchar* e1, const uchar* e2)
{
	return (*e2 - *e1);
}

inline long ColourDistanceGrayI(const uchar& e1, const uchar& e2)
{
	return e1 - e2;
}

template<int channel>
inline long ColourDistanceRGBI(const cv::Vec3b& e1, const cv::Vec3b& e2)
{
	return (e1)[channel] - (e2)[channel];
}

inline long ColourDistanceGrayIP(const uchar* e1, const uchar* e2)
{
	return ( *e1 - *e2);
}

inline long ColourDistanceGrayNorm(const uchar* e1, const uchar* e2)
{
	int ur1 = e1[0];
	int ur2 = e2[0];
	long rmean = (  ur1 + ur2 ) / 2;
	long r = ur1 - ur2;
	return (((512+rmean)*r*r)>>8);
}

inline int getValueCorner12(const uchar * ptr, int* pixel, int* corners, const int& k, const int& ks, const uchar& (*dist)(const uchar&, const uchar&) )
{
	int x = ptr[pixel[k]];

	if( k == 3 && ks != 2 && ks != 3 ){
		x = dist(x, ptr[corners[0]]);
	}else if(k == 5 && ks != 4 && ks != 5){
		x = dist(x, ptr[corners[1]]);
	}else if(k == 8 && ks != 7 && ks != 8){
		x = dist(x, ptr[corners[2]]);
	}else if(k == 11 && ks != 11){
		x = dist(x, ptr[corners[3]]);
	}

	return x;
}

inline void getCrossCorner12(const uchar * ptr, int* corners, int* cornersOut, const int& k, int& k1, int& k2, const uchar& (*dist)(const uchar&, const uchar&) )
{
	switch(k){
	case 0:
	case 1:
	case 11:
		k1 = dist(ptr[cornersOut[1]], ptr[corners[1]]);
		k2 = dist(ptr[cornersOut[2]], ptr[corners[2]]);
		break;
	case 2:
	case 3:
	case 4:
		k1 = dist(ptr[cornersOut[2]], ptr[corners[2]]);
		k2 = dist(ptr[cornersOut[3]], ptr[corners[3]]);
		break;
	case 5:
	case 6:
	case 7:
		k1 = dist(ptr[cornersOut[0]], ptr[corners[0]]);
		k2 = dist(ptr[cornersOut[3]], ptr[corners[3]]);
		break;
	case 8:
	case 9:
	case 10:
		k1 = dist(ptr[cornersOut[0]], ptr[corners[0]]);
		k2 = dist(ptr[cornersOut[1]], ptr[corners[1]]);
		break;
	}
}

/**
 * The interface method
 */
class CV_EXPORTS_W FASTextI
{
public:

	enum
	{
		KEY_POINTS_BLACK = 0, KEY_POINTS_WHITE = 1, KEY_POINTS_ALL = 3
	};

    CV_WRAP FASTextI( long threshold = 10, bool nonmaxSuppression=true, int keypointsTypes = KEY_POINTS_ALL, int Kmin = 9, int Kmax = 11);

    virtual ~FASTextI(){

    };

    void detect( const cv::Mat& image, std::vector<FastKeyPoint>& keypoints, const cv::Mat& mask ) const
    {
        keypoints.clear();

        if( image.empty() )
            return;

        CV_Assert( mask.empty() || (mask.type() == CV_8UC1 && mask.size() == image.size()) );
        detectImpl( image, keypoints, mask );
    }

    void segment( const cv::Mat& image, std::vector<FastKeyPoint>& keypoints, std::unordered_multimap<int, std::pair<int, int> >& keypointsPixels, const cv::Mat& mask ) const
    {
    	keypoints.clear();

    	if( image.empty() )
    		return;

    	CV_Assert( mask.empty() || (mask.type() == CV_8UC1 && mask.size() == image.size()) );
    	segmentImpl( image, keypoints, keypointsPixels, mask );
    }

    virtual bool isColorDetector(){
    	return false;
    }

    void setThreshold(long threshold){
    	this->threshold = threshold;
    }

    void setKeypointsTypes(int keypointsTypes){
    	this->keypointsTypes = keypointsTypes;
    }

protected:

    virtual void detectImpl( const cv::Mat& image, std::vector<FastKeyPoint>& keypoints, const cv::Mat& mask=cv::Mat() ) const = 0;

    virtual void segmentImpl( const cv::Mat& image, std::vector<FastKeyPoint>& keypoints,  std::unordered_multimap<int, std::pair<int, int> >& keypointsPixels, const cv::Mat& mask=cv::Mat() ) const
    {
    	detectImpl( image, keypoints, mask);
    }

    long threshold;
    bool nonmaxSuppression;
    int Kmin;
    int Kmax;

    int keypointsTypes;

    std::vector<std::vector<float> > fastAngles;
};

/**
 * Gray level FASText Feature detector
 */
class CV_EXPORTS_W FASTextGray : public FASTextI
{
public:

    CV_WRAP FASTextGray( long threshold=10, bool nonmaxSuppression=true, int keypointsTypes = KEY_POINTS_ALL, int Kmin = 9, int Kmax = 11);

    virtual ~FASTextGray(){

    };

protected:

    virtual void detectImpl( const cv::Mat& image, std::vector<FastKeyPoint>& keypoints, const cv::Mat& mask=cv::Mat() ) const;
};

}//namespace cmp;

#endif /* FAST_HPP_ */
