/*  FASTex.cpp
 *  Created on: Dec 15, 2015
 *      Author: Michal.Busta at gmail.com
 *
 * Copyright 2015, Michal Busta, Lukas Neumann, Jiri Matas.
 *
 * Based on:
 *
 * FASText: Efficient Unconstrained Scene Text Detector,Busta M., Neumann L., Matas J.: ICCV 2015.
 * Machine learning for high-speed corner detection, E. Rosten and T. Drummond, ECCV 2006
 */

#define _USE_MATH_DEFINES

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <map>
#include <assert.h>

#include "FT_common.hpp"

//#define RELATIVE_THRESH 1
#define DO_BENDS 1
#define CHECK_PATH 1

#if defined _MSC_VER
# pragma warning( disable : 4127)
#endif

#ifdef DO_ORB
#undef DO_BENDS
#undef CHECK_PATH
#endif

using namespace cv;

#define PATTERN_SIZE 12

namespace cmp
{

#define MIN_BEND_DIST 1

inline int check_pixel_bend(const uchar* ptr, int* offsets, const uchar* tab, const int& pos, int& vmin, int& vminIdx, int& vmax, int& vmaxIdx )
{
	if( ptr[offsets[pos]] <  vmin )
	{
		vminIdx = pos;
		vmin = ptr[offsets[pos]];
	}
	if( ptr[offsets[pos]] >  vmax )
	{
		vmaxIdx = pos;
		vmax = ptr[offsets[pos]];
	}
	return tab[ptr[offsets[pos]]];
}

bool isBrighter(const int& x, const int& vt ){
	return x > vt;
}

bool isDarker(const int& x, const int& vt ){
	return x < vt;
}

/**
 * Inner loop of FASText
 * @param img source image
 * @param N the pattern to check - most simple impl. uses 2xpattern size
 * @param threshold the edge threshold (relative)
 * @param vt the edge threshold (absolute)
 * @param ptr
 * @param pixel
 * @param corners
 * @param cornersOut
 * @param pixelIndex
 * @param pixelCheck16
 * @param Kmin the keypoint constraints (min count of Other pixels)
 * @param Kmax the keypoint constraints (max count of Other pixels)
 * @param isOther the function defines
 * @param maxdist
 * @param mindist
 * @param distFunction
 * @param func the callback function
 */
template<typename Func>
inline void fastext_inner_loop_12(Mat& img, const int& N, const int& threshold,
		const int& vt, const uchar* ptr, int* pixel, int *corners, int* cornersOut,  int* pixelIndex,
		const int* pixelCheck16,
		const int& Kmin, const int& Kmax, const int& maxValStart,
		bool (*isOther)(const int&, const int&),
		const uchar& (*maxdist)(const uchar&, const uchar&),
		const uchar& (*mindist)(const uchar&, const uchar&),
		long (*distFunction)(const uchar&, const uchar&),
		Func func)
{
	int count = 0;
	int minVal = 255;
	int vminIdx = 0;
	int vmaxVal = 0;
	int vmaxIdx = 0;

	int countAll = 0;
	int countDiff = 0;
	int vmin = maxValStart;
	int x = ptr[0];
	int ks = 0;
	int prevKs = -1;
	for( int k = 0; k < N; k++ )
	{
		//int& cornerIndex = pixelIndex[k];
		x = ptr[pixel[k]];
		int same = 0;

		if( x > vmaxVal )
			vmaxIdx = k;
		vmaxVal = max(x, vmaxVal);
		if( x < minVal )
			vminIdx = k;
		minVal = min(x, minVal);

		if( isOther(x, vt) )
		{

			countDiff = 0;
			++count;
			++countAll;
			if( count > 1 && count < Kmin)
				vmin = mindist(vmin, x);
			if( count >= Kmin )
			{
				//check Kmax
				int countR = 0;
				same++;
				int sameStart = 100;
				int sameEnd = -1;
				for(int check = 1; check < (PATTERN_SIZE - count) + 1; check++)
				{
					int l = k + check;
					int x = ptr[pixel[l]];
					if(isOther(x, vt))
					{
						countR++;
						if(same % 2 == 0)
							same++;
					}else
					{
						sameStart = MIN(sameStart, l);
						sameEnd = MAX(sameEnd, l);
						if(same % 2 == 1)
							same++;
					}
				}

				if((count + countR) <= Kmax && same <= 3)
				{

					int k1 = 0, k2 = 0;
					int sameCheck = ((sameEnd + sameStart) / 2) % 12;
					getCrossCorner12(ptr, corners, cornersOut, sameCheck, k1, k2, maxdist );
					if( !isOther(k1, vt) || !isOther(k2, vt) )
					{
						break;
					}

#ifdef CHECK_PATH
					if( sameStart != vmaxIdx )
						if( !isMostSameAccessible12(ptr, (int)img.step, 1, 0, sameStart, threshold, distFunction) )
							break;
					if( sameEnd != vmaxIdx )
						if( !isMostSameAccessible12(ptr, (int)img.step, 1, 0, sameEnd, threshold, distFunction) )
							break;

#endif
					int countc = 0;
					for(int c = 0; c < 16; c++)
					{
						int xc = ptr[pixelCheck16[c]];
						if(isOther(xc, vt))
							++countc;
					}
					if( countc == 16 )
					{
						countc = 24;
						return;
					}

					uchar kpType;
					if(countc == 24)
					{
						kpType = 6;
					}else
						kpType = sameEnd - sameStart + 1;

					func(kpType, vmaxIdx, vminIdx, vmin);

					break;
				}else
				{
					count = 0;
				}
			}
		}
		else
		{
			countDiff += 1;
			if( countDiff > 4)
				break;
			prevKs = ks;
			ks = k;
#ifdef DO_BENDS
			int ksDist = ks - prevKs;
			if(ksDist >= MIN_BEND_DIST)
			{
				int same = 0;
				int countR = 0;
				int countSame = 0;
				int countSamePrev = 0;
				for(int check = 0; check < (PATTERN_SIZE - k + prevKs); check++ )
				{
					int l = k + check;
					int x = ptr[pixel[l]];
					//int x = getValueCorner12(ptr, pixel, corners, pixelIndex[l], k, maxdist );
					if(isOther(x,vt))
					{
						if( l < 12)
							countR++;
						vmin = mindist(vmin, x);
						if(same % 2 == 0)
							same++;
					}else
					{
						x = ptr[pixel[l]];
						if(same == 0)
							countSame++;
						if(same % 2 == 1)
							same++;
						if(same == 2)
							countSamePrev++;
					}
				}
				if( same == 2 && (countR + countAll) > 7 && (countR + countAll) < 10 && countSame < 4 && countSamePrev < 4)
				{

#ifdef CHECK_PATH
					if( !isMostSameAccessible12(ptr, (int)img.step, 1, 0, prevKs - 1, threshold, distFunction) ||
							!isMostSameAccessible12(ptr, (int)img.step, 1, 0, k, threshold, distFunction))
					{
						count = 0;
						ks++;
						continue;
					}
#endif

					int countc = 0;
					for(int c = 0; c < 16; c++)
					{
						int xc = ptr[pixelCheck16[c]];
						if(isOther(xc,vt))
							++countc;
					}
					if( countc == 16 )
					{
						count = 0;
						ks++;
						continue;
					}

					uchar kpType;
					if(countc == 24)
						kpType = 6;
					else
						kpType = 5;

					func(kpType, vmaxIdx, vminIdx, vmin);
					break;

				}
			}
#endif
			ks++;
			count = 0;
		}
	}

}

void FASText12(cv::Ptr<cv::AutoBuffer<uchar> > _buf, const std::vector<std::vector<float> >& fastAngles,
		Mat& img, std::vector<FastKeyPoint>& keypoints, int threshold, bool nonmax_suppression, int keypointsTypes, const int Kmin = 9, const int Kmax = 11)
{
    const int N = 2 * PATTERN_SIZE;
    int i, j, pixel[34], pixelIndex[34], corners[8], cornersOut[8], pixelCheck[24], pixelCheck16[16];
    cmp::makeOffsets(pixel, corners, cornersOut, (int)img.step, PATTERN_SIZE, pixelIndex, pixelCheck, pixelCheck16);

    keypoints.clear();
    threshold = std::min(std::max(threshold, 0), 255);

#ifdef RELATIVE_THRESH
    float slope = 0.3;
    uchar threshold_tabr[26][512];
    for( j = 0; j < 26; j++)
    {
    	int thresholdr =  std::min(std::max((int) (threshold + slope * j), 0), 255);
    	for( i = -255; i <= 255; i++ )
    		threshold_tabr[j][i+255] = (uchar)(i < -thresholdr ? 1 : i > thresholdr ? 2 : 0);
    }
#else
    uchar threshold_tab[512];
    for( i = -255; i <= 255; i++ )
    	threshold_tab[i+255] = (uchar)(i < -threshold ? 1 : i > threshold ? 2 : 0);
#endif

    if(_buf.empty() )
    {
    	_buf = cv::Ptr<cv::AutoBuffer<uchar> > (new AutoBuffer<uchar>((img.cols+16)*3*(sizeof(int) * 4 + sizeof(uchar)) + 128));
    }else
    {
    	_buf->allocate( (img.cols+16)*3*(sizeof(int) * 4 + sizeof(uchar)) + 128 );
    }
    uchar* buf[3];
    buf[0] = *_buf; buf[1] = buf[0] + img.cols; buf[2] = buf[1] + img.cols;
    int* cpbuf[13];
    cpbuf[0] = (int*)alignPtr(buf[2] + img.cols, sizeof(int)) + 1;
    cpbuf[1] = cpbuf[0] + img.cols + 1;
    cpbuf[2] = cpbuf[1] + img.cols + 1;
    cpbuf[3] = cpbuf[2] + img.cols + 1;
    cpbuf[4] = cpbuf[3] + img.cols + 1;
    cpbuf[5] = cpbuf[4] + img.cols + 1;
    cpbuf[6] = cpbuf[5] + img.cols + 1;
    cpbuf[7] = cpbuf[6] + img.cols + 1;
    cpbuf[8] = cpbuf[7] + img.cols + 1;
    cpbuf[9] = cpbuf[8] + img.cols + 1;
    cpbuf[10] = cpbuf[9] + img.cols + 1;
    cpbuf[11] = cpbuf[10] + img.cols + 1;
    cpbuf[12] = cpbuf[11] + img.cols + 1;
    memset(buf[0], 0, img.cols*3);
    memset(cpbuf[9], 100, img.cols*3*sizeof(int));


    for(int i = 3; i < img.rows-3; i++)
    {
    	const uchar* ptr = img.ptr<uchar>(i) + 3;
    	uchar* curr = buf[(i - 3)%3];
    	int* cornerpos = cpbuf[(i - 3)%3];
    	int* mostSame = cpbuf[(i - 3)%3 + 3];
    	int* mostDiff = cpbuf[(i - 3)%3 + 6];
    	int* kpType = cpbuf[(i - 3)%3 + 9];
    	memset(curr, 0, img.cols);
    	memset(kpType, 100, img.cols * sizeof(int));
    	int ncorners = 0;
    	j = 3;
    	for( ; j < img.cols - 3; j++, ptr++ )
    	{
    		int v = ptr[0];
#ifdef RELATIVE_THRESH
    		const uchar* tab = &threshold_tabr[v / 10][0] - v + 255;
#else
    		const uchar* tab = &threshold_tab[0] - v + 255;
#endif
#ifdef STRAIT_KP
    		int d1 = tab[ptr[pixel[0]]] | tab[ptr[pixel[6]]];
    		int d2 = tab[ptr[pixel[3]]] | tab[ptr[pixel[9]]];

    		if( d1 == 0 && d2 == 0)
    		{
    			continue;
    		}
    		int d = d1;
    		d &= d2;

    		int d3 = (tab[ptr[pixel[1]]]) | (tab[ptr[pixel[7]]]);
    		int d4 = (tab[ptr[pixel[4]]]) | (tab[ptr[pixel[10]]]);

    		if( d3 == 0 && d4 == 0 )
    		{
    			continue;
    		}
    		d &= d3;
    		d &= d4;

    		int d5 = tab[ptr[pixel[2]]] | tab[ptr[pixel[8]]];
    		int d6 = tab[ptr[pixel[5]]] | tab[ptr[pixel[11]]];

    		if( d5 == 0 && d6 == 0 )
    		{
    			continue;
    		}
    		d = d1 & d2 & d3 & d4 & d5 & d6;

    		int white = 0;
    		int black = 0;
    		for( int t = 0; t < patternSize; t++ )
    		{
    			white += (tab[ptr[pixel[t]]] & 1) > 0;
    			black += (tab[ptr[pixel[t]]] & 2) > 0;
    		}


    		if( black == 12 )
    			continue;
    		if( white == 12 )
    			continue;
#else

    		int d = tab[ptr[pixel[0]]] | tab[ptr[pixel[6]]];

    		if( d == 0 )
    			continue;

    		d &= (tab[ptr[pixel[2]]]) | (tab[ptr[pixel[8]]]);
    		d &= tab[ptr[pixel[3]]] | tab[ptr[pixel[9]]];
    		d &= (tab[ptr[pixel[4]]]) | (tab[ptr[pixel[10]]]);

    		if( d == 0 )
    			continue;

    		d &= tab[ptr[pixel[1]]] | tab[ptr[pixel[7]]];
    		d &= tab[ptr[pixel[5]]] | tab[ptr[pixel[11]]];

#endif
    		//white ink
    		if( ((d & 1)) && ( (keypointsTypes & 1) > 0) )
    		{
#ifdef RELATIVE_THRESH
    			int thresholdr = threshold + v / 10 * slope;
    			int vt = v - thresholdr;
#else
    			int vt = v - threshold;
#endif

    			fastext_inner_loop_12(img, N, threshold, vt,
    			    ptr, pixel, corners, cornersOut, pixelIndex, pixelCheck16,
    			    Kmin, Kmax, 0,
    			    isDarker, std::min, std::max, ColourDistanceGrayI,
    			    [&] (uchar& kpT, int& vmaxIdx, int& vminIdx, int &vmin) {

    				kpType[j] = kpT;
    				cornerpos[ncorners++] = j;
    				mostSame[j] = pixelIndex[vmaxIdx];
    				mostDiff[j] = pixelIndex[vminIdx];

    				if(nonmax_suppression)
    				{
    					assert(v > vmin);
    					curr[j] = (uchar) (v - vmin);
    				}
    			} );
    		}

    		//black ink
    		if( ((d & 2) && (keypointsTypes & 2))  )
    		{
#ifdef RELATIVE_THRESH
    			int thresholdr = threshold + v / 10 * slope;
    			int vt = v + thresholdr;
#else
    			int vt = v + threshold;
#endif
    			fastext_inner_loop_12(img, N, threshold, vt,
    					ptr, pixel, corners, cornersOut, pixelIndex, pixelCheck16,
						Kmin, Kmax, 255,
						isBrighter, std::max, std::min, ColourDistanceGray,
						[&] (uchar& kpT, int& vmaxIdx, int& vminIdx, int &vmin) {

    				kpType[j] =  10 + kpT;
    				cornerpos[ncorners++] = j;
    				mostSame[j] = pixelIndex[vminIdx];
    				mostDiff[j] = pixelIndex[vmaxIdx];

    				if(nonmax_suppression)
    				{
    					assert(v < vmin);
    					curr[j] = (uchar) (vmin - v);
    				}
    			} );

    		}
    	}


        cornerpos[-1] = ncorners;

        if( i == 3 )
            continue;

        //non-maxima supression
        const uchar* prev = buf[(i - 4 + 3)%3];
        const uchar* pprev = buf[(i - 5 + 3)%3];

        const int* mostSamePrev = cpbuf[(i - 4 + 3)%3 + 3];
        const int* mostDiffPrev = cpbuf[(i - 4 + 3)%3 + 6];
        const int* kpTypePrev = cpbuf[(i - 4 + 3)%3 + 9];
        const int* kpTypePPrev = cpbuf[(i - 5 + 3)%3 + 9];
        cornerpos = cpbuf[(i - 4 + 3)%3];
        ncorners = cornerpos[-1];


        for( int k = 0; k < ncorners; k++ )
        {
            j = cornerpos[k];
            int score = prev[j];
            int kpTypeC = kpTypePrev[j];

            if( !nonmax_suppression ||
               ((score > prev[j+1] || kpTypePrev[j + 1] > kpTypeC ) && kpTypePrev[j + 1] >= kpTypeC && (score >= prev[j-1] || kpTypePrev[j - 1] > kpTypeC  ) && kpTypePrev[j - 1] >= kpTypeC &&
                (score >= pprev[j-1] ||  kpTypeC < kpTypePPrev[j - 1] ) && kpTypeC <= kpTypePPrev[j - 1] && (score > pprev[j] || kpTypeC < kpTypePPrev[j] ) && kpTypeC <= kpTypePPrev[j] && (score > pprev[j+1] || kpTypeC < kpTypePPrev[j + 1]  )  && kpTypeC <= kpTypePPrev[j + 1] &&
                (score >= curr[j-1] || kpTypeC < kpType[j - 1] ) && kpTypeC <= kpType[j - 1] && (score >= curr[j] || kpTypeC < kpType[j]) && kpTypeC <= kpType[j] && (score > curr[j+1] || kpTypeC < kpType[j + 1]) && kpTypeC <= kpType[j + 1]) )
            {
            	keypoints.push_back(FastKeyPoint((float)j, (float)(i - 1), 7.f, -1, (float)score, 0, keypoints.size()));
            	assert(mostDiffPrev[j] != -1);
            	if(mostDiffPrev[j] != -1)
            	{
            		int pxOffset = (int) pixel[mostDiffPrev[j]];
            		int yOffset = 0;
            		while( pxOffset <= -5 )
            		{
            			yOffset -= 1;
            			pxOffset += img.step;
            		}
            		while( pxOffset >= 5 )
            		{
            			yOffset += 1;
            			pxOffset -= img.step;
            		}
#ifndef NDEBUG
            		if (abs(pxOffset) >= 5)
            		{
            			assert(false);
            			continue;
            		}
            		if (abs(yOffset) >= 5)
            		{

            			assert(false);
            			continue;
            		}
#endif
            		keypoints.back().intensityOut.x = pxOffset;
            		keypoints.back().intensityOut.y = yOffset;
            		if( kpTypeC > 10){
            			keypoints.back().type = 1;
            			kpTypeC -= 10;
            		}
            		keypoints.back().count = kpTypeC;
            	}
            	assert(mostSamePrev[j] != -1);
				if (mostSamePrev[j] != -1)
				{
					int pxOffset = (int) pixel[mostSamePrev[j]];
					int yOffset = 0;
					while( pxOffset <= -5 )
					{
                		yOffset -= 1;
                		pxOffset += img.step;
					}
					while( pxOffset >= 5 )
					{
                		yOffset += 1;
                		pxOffset -= img.step;
					}
#ifndef NDEBUG
					if (abs(pxOffset) >= 5)
					{
						assert(false);
						continue;
					}
					if (abs(yOffset) >= 5)
					{

						assert(false);
						continue;
					}
#endif

					keypoints.back().intensityIn.x = pxOffset;
					keypoints.back().intensityIn.y = yOffset;
					keypoints.back().angle = fastAngles[yOffset + 2][pxOffset + 2];
				}
            }
        }
    }
}

/**
 *   FastFeatureDetector
 */
FASTextI::FASTextI( long _threshold, bool _nonmaxSuppression, int keypointsTypes, int Kmin, int Kmax )
    : threshold(_threshold), nonmaxSuppression(_nonmaxSuppression), keypointsTypes(keypointsTypes), Kmin(Kmin), Kmax(Kmax)
{
	for(int y = -2; y < 3; y++)
	{
		fastAngles.push_back( std::vector<float>() );
		for(int x = -2; x < 3; x++)
		{
			fastAngles[y + 2].push_back( atan2(y, x) * 180 / M_PI);
		}
	}
}

/**
 *   FastFeatureDetector
 */
FASTextGray::FASTextGray( long _threshold, bool _nonmaxSuppression, int keypointsTypes, int Kmin, int Kmax ): FASTextI( _threshold, _nonmaxSuppression, keypointsTypes, Kmin, Kmax )
{

}


void FASTextGray::detectImpl( const Mat& image, std::vector<FastKeyPoint>& keypoints, const Mat& mask ) const
{
    Mat grayImage = image;
    if( image.type() != CV_8UC1 )
    	cvtColor( image, grayImage, COLOR_BGR2GRAY );
    //imwrite("/tmp/fast.png", grayImage);
    cv::Ptr<cv::AutoBuffer<uchar> > autoBuffer;
    cmp::FASText12(autoBuffer, fastAngles, grayImage, keypoints, threshold, nonmaxSuppression, this->keypointsTypes, Kmin, Kmax);
    KeyPointsFilterC::runByPixelsMask( keypoints, mask );
}

}//namespace cmp
