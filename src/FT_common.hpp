/*
 * FT_common.hpp
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

#ifndef __CMP_FT_COMMON_HPP__
#define __CMP_FT_COMMON_HPP__

#ifdef __cplusplus

#include "FASTex.hpp"
#include <opencv2/features2d/features2d.hpp>


namespace cmp
{

void makeOffsets(int pixel[34], int* corners, int* cornersOut, int row_stride, int patternSize, int pixelIndex[34], int pixelcheck[24], int pixelcheck16[16]);
void makeOffsetsC(int pixel[34], int pixelCounter[34], int corners[8], int rowStride, int patternSize, int pixelcheck[24], int pixelcheck16[16]);

template<int patternSize>
int cornerScore(const uchar* ptr, const int pixel[], int threshold);

template<typename _Tp>
static inline bool isMostSameAccessible12(const uchar* ptr, int img_step, int xstep, int cn, int mostSameIdx, int threshold, long (*distFunction)(const _Tp&, const _Tp&))
{
	if( mostSameIdx > 11 )
		mostSameIdx -= 12;
	switch(mostSameIdx){
	case 0:
		if( !( distFunction(ptr[cn], ptr[img_step + cn])  <= threshold ) )
			return false;
		break;
	case 1:
		if( !( distFunction(ptr[cn], ptr[img_step + cn])  <= threshold
				|| distFunction(ptr[cn], ptr[img_step + xstep + cn])  <= threshold ) )
			return false;
		break;
	case 11:
		 if( !(distFunction(ptr[cn], ptr[img_step + cn])  <= threshold
				 || distFunction(ptr[cn], ptr[img_step -xstep + cn])  <= threshold) )
			 return false;
		 break;
	case 2:
		if( !(distFunction(ptr[cn], ptr[1 * xstep + cn])  <= threshold
				|| distFunction(ptr[cn], ptr[img_step + xstep + cn])  <= threshold ) )
			return false;
		break;
	case 3:
		if( !(distFunction(ptr[cn], ptr[xstep + cn])  <= threshold) )
			return false;
		break;
	case 4:
		if( !(distFunction(ptr[cn], ptr[1 * xstep + cn])  <= threshold
				|| distFunction(ptr[cn], ptr[-img_step + xstep + cn])  <= threshold ) )
			return false;
		break;
	case 5:
		if( !( distFunction(ptr[cn], ptr[-img_step + cn])  <= threshold
				|| distFunction(ptr[cn], ptr[-img_step + xstep + cn])  <= threshold ))
			return false;
		break;
	case 6:
		if( !(distFunction(ptr[cn], ptr[-img_step + cn])  <= threshold ) )
			return false;
		break;
	case 7:
		if( !(distFunction(ptr[cn], ptr[-img_step + cn])  <= threshold
				|| distFunction(ptr[cn], ptr[-img_step - xstep + cn])  <= threshold) )
			return false;
		break;
	case 8:
		if( !(distFunction(ptr[cn], ptr[-1*xstep + cn])  <= threshold
				|| distFunction(ptr[cn], ptr[-1*xstep - img_step + cn])  <= threshold ) )
			return false;
		break;
	case 9:
		if( !( distFunction(ptr[cn], ptr[-1*xstep + cn])  <= threshold ) )
			return false;
		break;
	case 10:
		if( !(distFunction(ptr[cn],  ptr[-1*xstep + cn])  <= threshold
				|| distFunction(ptr[cn],  ptr[-1*xstep + img_step + cn])  <= threshold) )
			return false;
		break;
	}
	return true;
}

}//namespace cmp

#endif
#endif //__CMP_FT_COMMON_HPP__
