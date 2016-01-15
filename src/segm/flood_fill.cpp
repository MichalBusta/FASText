/*********************************************************************
* Software License Agreement (BSD License)
*
*  Copyright (c) 2009, Willow Garage, Inc.
*  All rights reserved.
*
*  Redistribution and use in source and binary forms, with or without
*  modification, are permitted provided that the following conditions
*  are met:
*
*   * Redistributions of source code must retain the above copyright
*     notice, this list of conditions and the following disclaimer.
*   * Redistributions in binary form must reproduce the above
*     copyright notice, this list of conditions and the following
*     disclaimer in the documentation and/or other materials provided
*     with the distribution.
*   * Neither the name of the Willow Garage nor the names of its
*     contributors may be used to endorse or promote products derived
*     from this software without specific prior written permission.
*
*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
*  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
*  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
*  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
*  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
*  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
*  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
*  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
*  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
*  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
*  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
*  POSSIBILITY OF SUCH DAMAGE.
*********************************************************************/

#include "flood_fill.h"

#include <unordered_map>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core_c.h>
#include <opencv2/imgproc/types_c.h>

#include "FASTex.hpp"

namespace cmp{

#define ICV_PUSH( Y, L, R, PREV_L, PREV_R, DIR )  \
{                                                 \
    tail->y = (ushort)(Y);                        \
    tail->l = (ushort)(L);                        \
    tail->r = (ushort)(R);                        \
    tail->prevl = (ushort)(PREV_L);               \
    tail->prevr = (ushort)(PREV_R);               \
    tail->dir = (short)(DIR);                     \
    if( ++tail == buffer_end )                    \
    {                                             \
        buffer->resize(buffer->size() * 2);       \
        tail = &buffer->front() + (tail - head);  \
        head = &buffer->front();                  \
        buffer_end = head + buffer->size();       \
    }                                             \
}

#define ICV_POP( Y, L, R, PREV_L, PREV_R, DIR )   \
{                                                 \
    --tail;                                       \
    Y = tail->y;                                  \
    L = tail->l;                                  \
    R = tail->r;                                  \
    PREV_L = tail->prevl;                         \
    PREV_R = tail->prevr;                         \
    DIR = tail->dir;                              \
}

#define UP 1
#define DOWN -1

template<typename _Tp>
static void
icvFloodGrad_CnIR( uchar* idImage, int stepId, uchar* image, int stepY, CvSize roi, CvPoint seed, int newVal,
		CvConnectedComp* region, std::vector<CvFFillSegment>* buffer, long threshold, int maxSize, long (*diff)(const _Tp*, const _Tp*), cv::Mat& segmImg )
{
    int* idImg = (int*)(idImage + stepId * seed.y);
    _Tp* imgPtr = (_Tp*)(image + stepY * seed.y);

    threshold = abs(threshold);

    int i, L, R;
    int area = 0;
    int XMin, XMax, YMin = seed.y, YMax = seed.y;
    int _8_connectivity = 1;
    CvFFillSegment* buffer_end = &buffer->front() + buffer->size(), *head = &buffer->front(), *tail = &buffer->front();

    L = R = XMin = XMax = seed.x;

    idImg[L] = newVal;

    while( (R + 1) < roi.width && idImg[R + 1] != newVal && (diff( imgPtr + (R+1), imgPtr + R ) < threshold))
    	idImg[++R] = newVal;

    while( (L - 1) > 0 && idImg[L - 1]  != newVal && (diff( imgPtr + (L-1), imgPtr + L ) < threshold))
    	idImg[--L] = newVal;

    XMax = R;
    XMin = L;
    assert(R < roi.width);
    ICV_PUSH( seed.y, L, R, R + 1, R, UP );

    while( head != tail )
    {
        int k, YC, PL, PR, dir;
        ICV_POP( YC, L, R, PL, PR, dir );
        int data[][3] =
        {
            {-dir, L - _8_connectivity, R + _8_connectivity},
            {dir, L - _8_connectivity, PL - 1},
            {dir, PR + 1, R + _8_connectivity}
        };

        unsigned length = (unsigned)(R-L);

        if( (unsigned)(YC + dir) >= (unsigned)roi.height )
        	continue;

        if( region )
        {
            if(area > maxSize)
            {
            	region->area = -1;
            	return;
            }
            assert(R < roi.width);
            if( XMax < R ) XMax = R;
            if( XMin > L ) XMin = L;
            assert(XMin >= 0);
            assert(YC < roi.height);
            if( YMax < YC ) YMax = YC;
            if( YMin > YC ) YMin = YC;
            assert(YMin >= 0);
        }

        for( k = 0; k < 3; k++ )
        {
            dir = data[k][0];
            if( (unsigned)(YC + dir) >= (unsigned)roi.height )
            	continue;
            assert((YC + dir) < roi.height);
            assert((YC) < roi.height);
            idImg = (int*)(idImage + (YC + dir) * stepId);
#ifndef NDEBUG
            uchar* simg = segmImg.ptr<uchar>((YC + dir));
#endif
            imgPtr = (_Tp*)(image + stepY * (YC + dir));
            _Tp* img1 = (_Tp*)(image + YC * stepY);

            int left = data[k][1];
            int right = data[k][2];

            if( (unsigned)(YC + dir) >= (unsigned)roi.height )
                continue;

            for( i = left; i <= right; i++ )
            {
            	int idx;
            	_Tp val;

            	if( i >= roi.width )
            		continue;

            	if( idImg[i] != newVal &&
            			(((val = imgPtr[i],
            			(unsigned)(idx = i-L-1) <= length) &&
            			(diff( &val, img1 + (i-1)) < threshold)) ||
            			((unsigned)(++idx) <= length &&
            			(diff( &val, img1 + i ) < threshold)) ||
						((unsigned)(++idx) <= length &&
						(diff( &val, img1 + (i+1) ) < threshold)) ))
            	{
            		int j = i;
            		assert(i < roi.width);
            		idImg[i] = newVal;
#ifndef NDEBUG
            		simg[i] = MAX(150, simg[i]);
#endif
            		area++;
            		while( j > 0 && idImg[--j] != newVal && (diff( imgPtr + j, imgPtr + (j+1) ) < threshold) )
            		{
            			assert(j < (roi.width - 1));
            			idImg[j] = newVal;
#ifndef NDEBUG
            			simg[j] = MAX(150, simg[j]);
#endif
            			area++;
            		}

            		while( ++i <  roi.width && idImg[i] != newVal &&
            				((val = imgPtr[i],
            				diff( &val, imgPtr + (i-1) ) < threshold) ||
            				(((unsigned)(idx = i-L-1) <= length &&
            				diff( &val, img1 + (i-1) ) < threshold)) ||
							((unsigned)(++idx) <= length &&
							diff( &val, img1 + i ) < threshold) ||
							((unsigned)(++idx) <= length &&
							diff( &val, img1 + (i+1) ) < threshold)))
            		{
            			assert(i < roi.width);
            			idImg[i] = newVal;
#ifndef NDEBUG
            			simg[i] = MAX(150, simg[i]);
#endif
            			area++;
            		}
            		assert((i-1) < roi.width);
            		assert(R < roi.width);
            		if(i > 0)
            			ICV_PUSH( YC + dir, j+1, i-1, L, R, -dir );
            	}
            }
        }
    }

    if( region )
    {
    	//cv::imshow("segm", segmImg);
    	//cv::waitKey(0);

        region->area = area;
        region->rect.x = XMin;
        region->rect.y = YMin;
        region->rect.width = XMax - XMin + 1;
        region->rect.height = YMax - YMin + 1;
        region->value = cv::Scalar(newVal);
    }
}

template<typename _Tp>
static void
icvFloodFill_CnIR( uchar* idImage, int stepId, uchar* image, int stepY, CvSize roi, CvPoint seed, int newVal,
		CvConnectedComp* region, std::vector<CvFFillSegment>* buffer, long threshold, int maxSize, long (*distFunction)(const _Tp&, const _Tp&), cv::Mat& segmImg )
{
    int* idImg = (int*)(idImage + stepId * seed.y);
    _Tp* imgPtr = (_Tp*)(image + stepY * seed.y);
    _Tp& seedPtr = imgPtr[seed.x];

    int i, L, R;
    int area = 0;
    int XMin, XMax, YMin = seed.y, YMax = seed.y;
    int _8_connectivity = 1;
    CvFFillSegment* buffer_end = &buffer->front() + buffer->size(), *head = &buffer->front(), *tail = &buffer->front();

    L = R = XMin = XMax = seed.x;

    idImg[L] = newVal;

    while( ++R < roi.width && distFunction(seedPtr,  imgPtr[R]) <  threshold )
    {
    	idImg[R] = newVal;
    	area++;
    }

    while( --L >= 0 && distFunction(seedPtr,  imgPtr[L]) <  threshold )
    {
    	idImg[L] = newVal;
    	area++;
    }

    XMax = --R;
    XMin = ++L;

    ICV_PUSH( seed.y, L, R, R + 1, R, UP );

    while( head != tail )
    {
        int k, YC, PL, PR, dir;
        ICV_POP( YC, L, R, PL, PR, dir );
        int data[][3] =
        {
            {-dir, L - _8_connectivity, R + _8_connectivity},
            {dir, L - _8_connectivity, PL - 1},
            {dir, PR + 1, R + _8_connectivity}
        };

        if( region )
        {
            if(area > maxSize)
            {
            	region->area = -1;
            	return;
            }

            if( XMax < R ) XMax = R;
            if( XMin > L ) XMin = L;
            if( YMax < YC ) YMax = YC;
            if( YMin > YC ) YMin = YC;
        }

        for( k = 0; k < 3; k++ )
        {
            dir = data[k][0];
            idImg = (int*)(idImage + (YC + dir) * stepId);
            imgPtr = (_Tp*)(image + stepY * (YC + dir));
            int left = data[k][1];
            int right = data[k][2];

            if( (unsigned)(YC + dir) >= (unsigned)roi.height )
                continue;

            for( i = left; i <= right; i++ )
            {
                if( (unsigned)i < (unsigned)roi.width && idImg[i] != newVal && distFunction(seedPtr, imgPtr[i]) <  threshold)
                {
                    int j = i;
                    idImg[i] = newVal;
                    area++;
#ifndef NDEBUG
                    segmImg.at<uchar>((YC + dir), j) = MAX(150, segmImg.at<uchar>((YC + dir), j));
#endif
                    while( --j >= 0 && distFunction(seedPtr, imgPtr[j]) < threshold )
                    {
                    	idImg[j] = newVal;
                    	area++;
#ifndef NDEBUG
                    	segmImg.at<uchar>((YC + dir), j) = MAX(150, segmImg.at<uchar>((YC + dir), j));
#endif
                    }

                    while( ++i < roi.width && distFunction(seedPtr, imgPtr[i]) < threshold)
                    {
                    	idImg[i] = newVal;
                    	area++;
#ifndef NDEBUG
                    	segmImg.at<uchar>((YC + dir), i) = MAX(150, segmImg.at<uchar>((YC + dir), i));
#endif
                    }

                    ICV_PUSH( YC + dir, j+1, i-1, L, R, -dir );
                }
            }
        }
    }

    if( region )
    {
    	//cv::imshow("segm", segmImg);
    	//cv::waitKey(1);

        region->area = area;
        region->rect.x = XMin;
        region->rect.y = YMin;
        region->rect.width = XMax - XMin + 1;
        region->rect.height = YMax - YMin + 1;
        region->value = cv::Scalar(newVal);
    }
}

static inline CvSize cvGetMatSize( const CvMat* mat )
{
    CvSize size;
    size.width = mat->cols;
    size.height = mat->rows;
    return size;
}

void
floodFillC( std::vector<CvFFillSegment>& buffer, CvArr* idarr, CvArr* arr, CvPoint seed_point,
             int channel, int  newVal, CvScalar lo_diff, CvScalar up_diff,
             CvConnectedComp* comp, long threshold, int maxSize, cv::Mat& segmImg, bool gradFill)
{
    cv::Ptr<CvMat> tempMask;

    if( comp )
        memset( comp, 0, sizeof(*comp) );

    int i, type, cn;
    int buffer_size;

    struct { cv::Vec3b b; cv::Vec3i i; cv::Vec3f f; } ld_buf, ud_buf;
    CvMat stubid, *imgId = cvGetMat(idarr, &stubid);
    CvMat stub, *img = cvGetMat(arr, &stub);
    CvSize size;
    type = CV_MAT_TYPE( img->type );
    cn = CV_MAT_CN(type);

    if ( (cn != 1) && (cn != 3) )
    {
        CV_Error( CV_StsBadArg, "Number of channels in input image must be 1 or 3" );
        return;
    }

    for( i = 0; i < cn; i++ )
    {
        if( lo_diff.val[i] < 0 || up_diff.val[i] < 0 )
            CV_Error( CV_StsBadArg, "lo_diff and up_diff must be non-negative" );
    }

    size = cvGetMatSize( img );

    if( (unsigned)seed_point.x >= (unsigned)size.width ||
        (unsigned)seed_point.y >= (unsigned)size.height )
        CV_Error( CV_StsOutOfRange, "Seed point is outside of image" );

    buffer_size = MAX( size.width, size.height ) * 2;
    buffer.clear();
    buffer.resize( buffer_size );

    if( type == CV_8UC1 )
    {
    	if( gradFill )
    	{
    		if(threshold > 0)
    			icvFloodGrad_CnIR<uchar>(imgId->data.ptr, imgId->step, img->data.ptr, img->step, size, seed_point, newVal, comp, &buffer, threshold / 2, maxSize, &ColourDistanceGrayIP, segmImg);
    		else
    			icvFloodGrad_CnIR<uchar>(imgId->data.ptr, imgId->step, img->data.ptr, img->step, size, seed_point, newVal, comp, &buffer, threshold / 2, maxSize, &ColourDistanceGrayP, segmImg);
    	}else if(threshold > 0)
    	{
    		icvFloodFill_CnIR<uchar>(imgId->data.ptr, imgId->step, img->data.ptr, img->step, size, seed_point, newVal, comp, &buffer, threshold, maxSize, &ColourDistanceGray, segmImg);
    	}else{
    		icvFloodFill_CnIR<uchar>(imgId->data.ptr, imgId->step, img->data.ptr, img->step, size, seed_point, newVal, comp, &buffer, -threshold, maxSize, &ColourDistanceGrayI, segmImg);
    	}
    }
    else if( type == CV_8UC3 )
    {
    	std::vector<CvFFillSegment> buffer2;
    	buffer2.resize( buffer_size );
    	if(threshold > 0)
        {
    		switch(channel){
    		case 0:
    			icvFloodFill_CnIR<cv::Vec3b>(imgId->data.ptr, imgId->step, img->data.ptr, img->step, size, seed_point, newVal, comp, &buffer2, threshold, maxSize, &ColourDistanceRGB<0>, segmImg);
    			break;
    		case 1:
    			icvFloodFill_CnIR<cv::Vec3b>(imgId->data.ptr, imgId->step, img->data.ptr, img->step, size, seed_point, newVal, comp, &buffer2, threshold, maxSize, &ColourDistanceRGB<1>, segmImg);
    			break;
    		case 2:
    			icvFloodFill_CnIR<cv::Vec3b>(imgId->data.ptr, imgId->step, img->data.ptr, img->step, size, seed_point, newVal, comp, &buffer2, threshold, maxSize, &ColourDistanceRGB<2>, segmImg);
    			break;
    		default:
    			CV_Error( CV_StsOutOfRange, "Invalid channel!" );
    			break;
    		}

        }else{
        	switch(channel){
    		case 0:
    			icvFloodFill_CnIR<cv::Vec3b>(imgId->data.ptr, imgId->step, img->data.ptr, img->step, size, seed_point, newVal, comp, &buffer2, -threshold, maxSize, &ColourDistanceRGBI<0>, segmImg);
    			break;
    		case 1:
    			icvFloodFill_CnIR<cv::Vec3b>(imgId->data.ptr, imgId->step, img->data.ptr, img->step, size, seed_point, newVal, comp, &buffer2, -threshold, maxSize, &ColourDistanceRGBI<1>, segmImg);
    			break;
    		case 2:
    			icvFloodFill_CnIR<cv::Vec3b>(imgId->data.ptr, imgId->step, img->data.ptr, img->step, size, seed_point, newVal, comp, &buffer2, -threshold, maxSize, &ColourDistanceRGBI<2>, segmImg);
    			break;
    		default:
    			CV_Error( CV_StsOutOfRange, "Invalid channel!" );
    			break;
        	}
        }
    }
    else
    	CV_Error( CV_StsUnsupportedFormat, "" );
}

int floodFill( std::vector<CvFFillSegment>& buffer, cv::InputOutputArray _imageId, cv::InputOutputArray _image, cv::Point seedPoint, int channel, double scaleFactor,
		int& compCounter, long threshold, int maxSize, int minCompSize, cv::Mat& segmImg, cv::Mat& segmMap, cv::Rect& rect, int& area, std::unordered_map<int, int>& keypointHash, std::vector<int>& keypointIds,  bool resegment,
		bool gradFill, int srcCols, cv::Scalar loDiff, cv::Scalar upDiff)
{
    CvConnectedComp ccomp;
    CvMat c_image = _image.getMat();
    CvMat c_imageId = _imageId.getMat();

    if(! resegment )
    {
    	int* checkRow = (int*) (c_imageId.data.ptr + seedPoint.y * c_imageId.step);
    	if(checkRow[seedPoint.x] > 0)
    		return checkRow[seedPoint.x];
    }

    compCounter++;

    floodFillC(buffer, &c_imageId, &c_image, seedPoint, channel, compCounter, loDiff, upDiff, &ccomp, threshold, maxSize, segmMap, gradFill);
    rect = ccomp.rect;

    if( ccomp.area == -1 )
    	return -2;

    if( ccomp.area < minCompSize )
    	return -1;

    segmImg = cv::Mat::zeros( ccomp.rect.height, ccomp.rect.width, CV_8UC1 );
    for (int y = 0; y < ccomp.rect.height; y++  )
    {
    	int* rowId  = (int*)(c_imageId.data.ptr + c_imageId.step * (y + ccomp.rect.y));
    	uchar* rowSegm = &segmImg.at<uchar>(y * segmImg.step);
    	int ybase = ((int) roundf(((y + ccomp.rect.y)))) * srcCols;
    	for(int x = 0; x <  ccomp.rect.width; x++)
    	{
    		if( rowId[x + ccomp.rect.x] == compCounter)
    		{
    			rowSegm[x] = 255;
    			int index = ( (ybase) + roundf((x + ccomp.rect.x)));
    			if( keypointHash.find( index ) != keypointHash.end()  )
    			{
    				keypointIds.push_back(keypointHash[index]);
    			}
#ifndef NDEBUG
    			segmMap.at<uchar>(y + ccomp.rect.y, x + ccomp.rect.x) = 255;
#endif
    		}
    	}
    }
    area = ccomp.area;
    return compCounter;
}

}//namespace cmp



