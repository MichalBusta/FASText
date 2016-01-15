/*
 * FT_common.cpp
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

#include "FT_common.hpp"

#define VERIFY_CORNERS 0

namespace cmp {

void makeOffsets(int pixel[34], int* corners, int* cornersOut, int rowStride, int patternSize, int pixelIndex[34],  int pixelcheck[24], int pixelcheck16[16])
{
	static const int offsets24[][2] =
	{
		{0,  4}, { 1,  4}, { 2,  4}, { 3,  3}, { 4, 2}, { 4, 1}, { 4, 0}, { 4, -1},
		{ 4, -2}, { 3, -3}, { 2, -4}, { 1, -4}, {0, -4}, {-1, -4}, {-2, -4}, {-3, -3},
		{-4, -2}, {-4, -1}, {-4, 0}, {-4,  1}, {-4,  2}, {-3,  3}, {-2,  4}, {-1,  4}
	};

    static const int offsets16[][2] =
    {
        {0,  3}, { 1,  3}, { 2,  2}, { 3,  1}, { 3, 0}, { 3, -1}, { 2, -2}, { 1, -3},
        {0, -3}, {-1, -3}, {-2, -2}, {-3, -1}, {-3, 0}, {-3,  1}, {-2,  2}, {-1,  3}
    };

    static const int corners16[][2] =
    {
        { 3,  2}, { 2,  3}, { 3, -2}, { 2, -3},
		{-2, -3}, {-3, -2} , {-3,  2}, {-2,  3}
    };

    static const int offsets12[][2] =
    {
        {0,  2}, { 1,  2}, { 2,  1}, { 2, 0}, { 2, -1}, { 1, -2},
        {0, -2}, {-1, -2}, {-2, -1}, {-2, 0}, {-2,  1}, {-1,  2}
    };

    static const int corners12[][2] =
    {
    	{ 2,  2}, { 2, -2},
		{-2, -2}, {-2,  2}
    };
    static const int cornersOut12[][2] =
    {
    	{ 3,  3}, { 3, -3},
    	{-3, -3}, {-3,  3}
    };


    static const int offsets8[][2] =
    {
        {0,  1}, { 1,  1}, { 1, 0}, { 1, -1},
        {0, -1}, {-1, -1}, {-1, 0}, {-1,  1}
    };

    const int (*offsets)[2] = patternSize == 16 ? offsets16 :
                              patternSize == 12 ? offsets12 :
                              patternSize == 8  ? offsets8  : 0;

    CV_Assert(pixel && offsets);

    int k = 0;
    for( ; k < patternSize; k++ )
    {
        pixel[k] = offsets[k][0] + offsets[k][1] * rowStride;
        pixelIndex[k] = k;
    }
    for( ; k < 34; k++ )
    {
        pixel[k] = pixel[k - patternSize];
        pixelIndex[k] = k - patternSize;
    }
    if(patternSize == 16)
    {
    	for( k = 0; k < 8; k++ )
    		corners[k] = corners16[k][0] + corners16[k][1] * rowStride;
    }else{
    	for( k = 0; k < 4; k++ )
    	{
    		corners[k] = corners12[k][0] + corners12[k][1] * rowStride;
    		cornersOut[k] = cornersOut12[k][0] + cornersOut12[k][1] * rowStride;
    	}
    }
    for( k = 0; k < 24; k++ )
    	pixelcheck[k] =  offsets24[k][0] + offsets24[k][1] * rowStride;
    for( k = 0; k < 16; k++ )
    	pixelcheck16[k] =  offsets16[k][0] + offsets16[k][1] * rowStride;
}

void makeOffsetsC(int pixel[34], int pixelCounter[34], int corners[8], int rowStride, int patternSize, int pixelcheck[24], int pixelcheck16[16])
{
	static const int offsets24[][2] =
	{
		{0,  4}, { 1,  4}, { 2,  4}, { 3,  3}, { 4, 2}, { 4, 1}, { 4, 0}, { 4, -1},
		{ 4, -2}, { 3, -3}, { 2, -4}, { 1, -4}, {0, -4}, {-1, -4}, {-2, -4}, {-3, -3},
		{-4, -2}, {-4, -1}, {-4, 0}, {-4,  1}, {-4,  2}, {-3,  3}, {-2,  4}, {-1,  4}
	};

    static const int offsets16[][2] =
    {
        {0,  3}, { 1,  3}, { 2,  2}, { 3,  1}, { 3, 0}, { 3, -1}, { 2, -2}, { 1, -3},
        {0, -3}, {-1, -3}, {-2, -2}, {-3, -1}, {-3, 0}, {-3,  1}, {-2,  2}, {-1,  3}
    };

    static const int corners16[][2] =
    {
    	{ 3,  2}, { 2,  3}, { 3, -2}, { 2, -3},
		{-2, -3}, {-3, -2} , {-3,  2}, {-2,  3}
    };

    static const int offsets12[][2] =
    {
        {0,  2}, { 1,  2}, { 2,  1}, { 2, 0}, { 2, -1}, { 1, -2},
        {0, -2}, {-1, -2}, {-2, -1}, {-2, 0}, {-2,  1}, {-1,  2}
    };

    static const int corners12[][2] =
    {
    	{ 2,  2}, { 2, -2},
		{-2, -2}, {-2,  2}
    };

    static const int offsets8[][2] =
    {
        {0,  1}, { 1,  1}, { 1, 0}, { 1, -1},
        {0, -1}, {-1, -1}, {-1, 0}, {-1,  1}
    };

    const int (*offsets)[2] = patternSize == 16 ? offsets16 :
                              patternSize == 12 ? offsets12 :
                              patternSize == 8  ? offsets8  : 0;

    CV_Assert(pixel && offsets);

    int k = 0;
    for( ; k < patternSize; k++ )
    {
        pixel[k] = 3 * offsets[k][0] + offsets[k][1] * rowStride;
        pixelCounter[k] = k;
    }
    for( ; k < 34; k++ )
    {
        pixel[k] = pixel[k - patternSize];
        pixelCounter[k] = k - patternSize;
    }

    if(patternSize == 16)
    {
    	for( k = 0; k < 8; k++ )
    		corners[k] = 3 * corners16[k][0] + corners16[k][1] * rowStride;
    }else{
    	for( k = 0; k < 4; k++ )
    		corners[k] = 3 * corners12[k][0] + corners12[k][1] * rowStride;
    }
    for( k = 0; k < 24; k++ )
    	pixelcheck[k] =  3 * offsets24[k][0] + offsets24[k][1] * rowStride;
    for( k = 0; k < 16; k++ )
    	pixelcheck16[k] =  3 * offsets16[k][0] + offsets16[k][1] * rowStride;
}

} // namespace cmp
