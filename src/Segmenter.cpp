/*
 * Segmenter.cpp
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
#include <unordered_map>
#include <stack>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "Segmenter.h"

#include "FT_common.hpp"

using namespace std;

//#define VERBOSE 1

namespace cmp
{

Segmenter::Segmenter(cv::Ptr<CharClassifier> charClassifier, int maxComponentSize, int minCompSize) :
		charClassifier(charClassifier), maxComponentSize(maxComponentSize), minCompSize(minCompSize), classificationTime(0)
{

}

Segmenter::~Segmenter()
{
	// TODO Auto-generated destructor stub
}

void Segmenter::classifyLetters(std::vector<cmp::FastKeyPoint>& img1_keypoints, std::unordered_multimap<int, std::pair<int, int> >& keypointsPixels, vector<double>& scales, std::vector<cmp::LetterCandidate*>& letters, cv::Mat debugImage)
{
	if(letterCandidates.size() == 0)
		return;

	//draw the keypoints groups
	letters.reserve(letterCandidates.size());
	int letterNo = 0;

#ifdef PARALLEL
#pragma omp parallel for
#endif
	for (size_t k = 0; k < letterCandidates.size(); k++)
	{
		LetterCandidate* letter = &letterCandidates[k];
		if(letter->duplicate != -1)
			continue;

		componentsCount += 1;
		int64 startTime = cv::getTickCount();
		if(keypointsPixels.size() > 0)
		{
			letter->quality = letter->getStrokeAreaRatio(img1_keypoints, keypointsPixels);
		}else{
			letter->quality = letter->getStrokeAreaRatio(img1_keypoints, scales, keypointStrokes);
		}

		strokeAreaTime += cv::getTickCount() - startTime;

		if(!charClassifier.empty())
		{
			if(charClassifier->classifyLetter(*letter)){
				double q = letter->quality;
				if(!wordClassifier.empty() && q > 0.5){
					letter->isWord = wordClassifier->isWord(*letter) > 0.5;
					/*
						if(letter->isWord )
						{
							std::cout << "Letter is word with probability: " << letter->quality << std::endl;
							cv::imshow("multiChar", letter->mask);
							cv::waitKey(0);
						}*/
				}
			}
		}

		classificationTime += cv::getTickCount() - startTime;
#ifdef PARALLEL
	}

	for (size_t k = 0; k < letterCandidates.size(); k++)
	{
		LetterCandidate* letter = &letterCandidates[k];
		if(letter->duplicate != -1)
			continue;
#endif
		if(dumpTrainingData)
		{
			if( letter->quality > 0.5)
			{
				ostringstream os;
				os << "/tmp/chars/" << letter->quality << "-" << rand() << "-" << k << ".png";

				cv::Mat tmp;
				cv::resize(letter->mask, tmp, cv::Size(letter->mask.cols * letter->scaleFactor, letter->mask.rows * letter->scaleFactor));

				imwrite(os.str(), tmp);
			}else{
				ostringstream os;
				cv::Mat tmp;
				cv::resize(letter->mask, tmp, cv::Size(letter->mask.cols * letter->scaleFactor, letter->mask.rows * letter->scaleFactor));
				os << "/tmp/nonChars/" << letter->quality << "-" << rand() << "-" << k << ".png";
				imwrite(os.str(), tmp);

			}
		}

		if(!debugImage.empty())
		{
			cv::Scalar color = cv::Scalar(0, 255, 0);
			cv::Rect roi = letter->bbox;
			roi.x -= 2;
			roi.y -= 2;
			roi.width += 4;
			roi.height += 4;
			cv::rectangle(debugImage, roi, color, 1);
		}
		letter->outputOrder = letters.size();
		letters.push_back( &*letter );
		letterNo++;
	}
}


struct Direction{
	uchar dirStart;
	uchar dirEnd;
	int mostSame;

	Direction(uchar dirStart, uchar dirEnd, int mostSame): dirStart(dirStart), dirEnd(dirEnd), mostSame(mostSame){

	}
};


inline void getDirections(long dists[12], int threshold, std::vector<Direction>& directions)
{
	int directionStart = 18;
	int directionEnd = 0;
	int mostSame = 0;
	long minDist = LONG_MAX;
	long minDistFirst = LONG_MAX;
	for(int i = 0; i < 12; i++)
	{
		if( dists[i] < threshold)
		{
			directionStart = MIN(i, directionStart);
			directionEnd = MAX(i, directionEnd);
			if( dists[i] < minDist  )
			{
				mostSame = i;
				minDist = dists[i];
			}
		}else if( directionStart != 18 )
		{
			directions.push_back(Direction(directionStart, directionEnd, mostSame));
			if(minDistFirst == LONG_MAX)
				minDistFirst = minDist;
			mostSame = 0;
			minDist = LONG_MAX;
			directionStart = 18;
			directionEnd = 0;
		}else{
			mostSame = 0;
			minDist = LONG_MAX;
			directionStart = 18;
		}
	}
	if( directionStart != 18 )
	{
		if(directions.size() > 0 && directions[0].dirStart == 0){
			directions[0].dirStart = directionStart;
			if( minDist < minDistFirst)
			{
				directions[0].mostSame = mostSame;
			}
		}else{
			directions.push_back(Direction(directionStart, directionEnd, mostSame));
		}
	}
}

template<typename _Tp>
static bool moveStroke(cv::Mat& img, cv::Mat& segmMap, cv::Mat& idImage, StrokeDir& current,  long (*distFunction)(const _Tp*, const _Tp*), long& threshold, int& compNo, std::vector<std::vector<cv::Point> >& steps )
{
	static const int offsetsStrokes16L[][2] =
	{
			{1,  0}, { 1,  0}, { 1,  0}, { 0,  -1}, { 0, -1}, { 0, -1}, { 0, -1}, { -1, 0},
			{-1, 0}, {-1, 0}, {-1, 0}, {0, -1}, {0, -1}, {0,  -1}, {-1,  0}, {-1,  0}
	};
	static const int offsetsStrokes16R[][2] =
	{
			{-1,  0}, {-1,  0}, {0,  1}, { 0,  1}, {0, 1}, { 0, 1}, { 1, 0}, { 1, 0},
			{1, 0}, {1, 0}, {0, -1}, {0, 1}, {0, 1}, {0,  1}, {0,  1}, {1,  0}
	};
	int xStep = 1;
	if(img.type() == CV_8UC3)
		xStep = 3;
	cv::Point cp = current.center;
	const uchar* ptr = &img.at<uchar>(cp.y, cp.x);

	for(size_t i = 0; i < steps[current.idx].size(); i++)
	{

		bool change = false;
		const uchar* ptrc = img.ptr<uchar>(cp.y) + cp.x * xStep;
		if( distFunction(ptr, ptrc)  < threshold )
		{
			idImage.at<int>(cp.y, cp.x) = compNo;
#ifndef NDEBUG
			uchar* sptr = &segmMap.at<uchar>(cp.y,  cp.x);
			*sptr = MAX(*sptr, 100);
#endif
			change = true;
		}
		cv::Point cl(cp.x + offsetsStrokes16L[current.idx][0], cp.y + offsetsStrokes16L[current.idx][1]);
		cv::Point cr(cp.x + offsetsStrokes16R[current.idx][0], cp.y + offsetsStrokes16R[current.idx][1]);

		uchar* sptr2 = segmMap.ptr<uchar>(cl.y) + cl.x;
		const uchar* ptr2 = img.ptr<uchar>(cl.y) + cl.x * xStep;
		if( distFunction(ptr, ptr2)  < threshold )
		{
			idImage.at<int>(cl.y, cl.x) = compNo;
#ifndef NDEBUG
			uchar* sptr2 = segmMap.ptr<uchar>(cl.y) + cl.x;
			*sptr2 = MAX(*sptr2, 100);
#endif
			change = true;
		}
		uchar* ptr3 = img.ptr<uchar>(cr.y) + cr.x * xStep;
		if( distFunction(ptr, ptr3)  < threshold )
		{
			idImage.at<int>(cr.y, cr.x) = compNo;
#ifndef NDEBUG
			uchar* sptr3 = segmMap.ptr<uchar>(cr.y) + cr.x;
			*sptr3 = MAX(*sptr3, 100);
#endif
			change = true;
		}
		if(!change)
			return false;
		cp +=  steps[current.idx][i];
	}
	return true;
}

static int getIdx12(cv::Point pt)
{
	if( pt.y == 2 )
	{
		if(pt.x == 0)
			return 0;
		if(pt.x == 1)
			return 1;
		if(pt.x == -1)
			return 11;

	}else if( pt.y == 1 )
	{
		if( pt.x == 2 )
			return 2;
		if( pt.x == -2 )
			return 10;
	}else if( pt.y == 0 )
	{
		if( pt.x == 2 )
			return 3;
		else if( pt.x == -2 )
			return 9;
	}
	else if( pt.y == -1 )
	{
		if( pt.x == 2 )
			return 4;
		else if( pt.x == -2 )
			return 8;
	}
	else if( pt.y == -2 )
	{
		if( pt.x == 1 )
			return 5;
		else if( pt.x == 0 )
			return 6;
		else if( pt.x == -1 )
			return 7;
	}
	return -1;
}

template<typename _Tp>
int segmentStroke(cv::Mat& img, cv::Mat& segmMap, cv::Mat& idImage, cmp::FastKeyPoint& keypoint, double scaleFactor, long (*distFunction)(const _Tp&, const _Tp&), long threshold, int& compCounter, cv::Mat& segmImg, int& area, cv::Rect& roi, std::vector<std::vector<cv::Ptr<StrokeDir> > > & strokes, bool single, int pixel[34], int msLength )
{

	int compNo = ++compCounter;

	int maxStrokeLength = msLength * keypoint.count;
	long dists[16];
	static const int offsets12[][2] =
	{
			{0,  2}, { 1,  2}, { 2,  1}, { 2, 0}, { 2, -1}, { 1, -2},
			{0, -2}, {-1, -2}, {-2, -1}, {-2, 0}, {-2,  1}, {-1,  2}
	};

	threshold = abs(threshold);
	float sf = scaleFactor;
	cv::Point center = cv::Point(round(keypoint.pt.x / sf), round(keypoint.pt.y / sf));
	//uchar* ptr = &((img.ptr<uchar>((int) center.y))[((int) center.x) * channels]);
	cv::Point same = cv::Point(round(keypoint.intensityIn.x / sf), round(keypoint.intensityIn.y / sf));
	int idx = getIdx12(same - center);
	assert(idx >= 0);
	int XMin = center.x, XMax = center.x, YMin = center.y, YMax = center.y;

	std::vector<std::vector<cv::Point> > steps;
	//makeSteps(steps);
	std::stack<cv::Ptr<StrokeDir> > stack;
	stack.push(cv::Ptr<StrokeDir> (new StrokeDir(idx, threshold, center, same)));
	cv::Ptr<StrokeDir> next;
	int strokeLength = 0;
	while(stack.size() > 0 || !next.empty())
	{
		cv::Ptr<StrokeDir> current;
		if( !next.empty() )
		{
			current = next;
			strokes.back().push_back(current);
			next = cv::Ptr<StrokeDir>();
			strokeLength++;
			if( strokeLength > maxStrokeLength || stack.size() > 10 || strokes.size() > 5)
				return -1;
		}else{
			current = stack.top();
			stack.pop();
			strokeLength = 0;
			strokes.push_back(std::vector<cv::Ptr<StrokeDir> >());
			strokes.back().push_back(current);

		}
		uchar* ptr = &img.at<_Tp>((int) current->center.y, current->center.x);

		XMin = MIN(XMin, current->center.x);
		XMax = MAX(XMax, current->center.x);
		YMin = MIN(YMin, current->center.y);
		YMax = MAX(YMax, current->center.y);

		for( int k = 0; k < 12; k++ )
		{
			dists[k] = distFunction(*ptr, ptr[pixel[k]]);
		}
		long thresholdc = threshold;
		int repeat = 1;
		while( repeat > 0 )
		{
			repeat = 0;
			bool updateThreshold = false;
			std::vector<Direction> directions;
			getDirections(dists, thresholdc, directions);
			int dCount = 0;
			for(size_t j = 0; j < directions.size(); j++)
			{
				Direction dir = directions[j];
				if(dir.dirStart > dir.dirEnd)
					dir.dirEnd += 12;
				if((dir.dirEnd - dir.dirStart) > 10)
				{
					repeat = 0;
					continue;
				}
				if( (dir.dirEnd - dir.dirStart) > 5 )
				{
					repeat = 0;
					continue;
				}

				int idx =  dir.mostSame;
				if((dir.dirEnd - dir.dirStart) > 3)
					idx = (dir.dirStart + dir.dirEnd) / 2;
				idx = idx % 12;
				cv::Point strokeDir(current->center.x + offsets12[idx][0], current->center.y + offsets12[idx][1]);
				assert(dir.mostSame < 12);

				if(!isMostSameAccessible12(ptr, img.step[0], 1, 0, dir.mostSame % 12, threshold, distFunction))
				{
					repeat = 0;
					continue;
				}

				if(strokeDir.x < 3 || strokeDir.x >= img.cols - 2 || strokeDir.y < 3 || strokeDir.y >= img.rows - 2 )
				{
					repeat = 0;
					continue;
				}
				if( idImage.at<int>(strokeDir.y, strokeDir.x) == compNo)
				{
					strokes.back().push_back(current);
					repeat = 0;
					continue;
				}
				double dist = cv::norm(strokeDir-current->center);
				if(dist >= 2)
				{
					int idxDist = abs(idx - current->idx);
					if( abs(idx - current->idx) < 4 || abs(idx - current->idx) >= 8 )
					{
						if(next.empty())
						{
							next = cv::Ptr<StrokeDir> (new StrokeDir(idx, thresholdc, current->direction, strokeDir));
							repeat = 0;
						}
						else if( idxDist < abs(next->idx - current->idx) )
						{
							if( !single )
								stack.push(next);
							next = cv::Ptr<StrokeDir> (new StrokeDir(idx, thresholdc, current->direction, strokeDir));
						}else if( !single )
						{
							stack.push(cv::Ptr<StrokeDir> (new StrokeDir(idx, thresholdc, current->direction, strokeDir)));
						}

					}else if( !single ){
						stack.push(cv::Ptr<StrokeDir> (new StrokeDir(idx, thresholdc, current->direction, strokeDir)));
					}
					dCount++;
				}
				//cv::imshow("StrokeMap", segmMap);
				//cv::waitKey(1);
			}
			if(dCount > 0)
			{
				repeat = 0;
			}else{
				if(!updateThreshold)
				{
					thresholdc += 11;
					repeat--;
				}
			}
		}
	}

	if(strokes.size() == 0 || strokes[0].size() == 0 )
	{
		return -1;
	}

	if( !single )
	{

		roi = cv::Rect(XMin, YMin, XMax - XMin + 1, YMax - YMin + 1);
		segmImg = cv::Mat::zeros( roi.height, roi.width, CV_8UC1 );

		area = 0;
		for (int y = 0; y < roi.height; y++  )
		{
			int* rowId  = idImage.ptr<int>(y + roi.y);
			uchar* rowSegm = segmImg.ptr<uchar>(y);
			for(int x = 0; x <  roi.width; x++)
			{
				if( rowId[x + roi.x] == compCounter)
				{
					rowSegm[x] = 255;
					area++;
#ifndef NDEBUG
					segmMap.at<uchar>(y + roi.y, x + roi.x) = 255;
#endif
				}
			}
		}
	}
	return compNo;
}

#define INT_OFFSET 2

void PyramidSegmenter::getLetterCandidates(cv::Mat& img, std::vector<cmp::FastKeyPoint>& img1_keypoints, std::unordered_multimap<int, std::pair<int, int> >& keypointsPixels, std::vector<cmp::LetterCandidate*>& letters, cv::Mat debugImage, int minHeight)
{

	int edgeThreshold = ftDetector->getEdgeThreshold();

	classificationTime = 0;
	if(!charClassifier.empty())
		charClassifier->classificationTime = 0;
	letterCandidates.clear();
	letterCandidates.reserve(img1_keypoints.size() / 3);

	std::sort(img1_keypoints.begin(), img1_keypoints.end(), [](const FastKeyPoint & a, const FastKeyPoint & b) -> bool
	{
		if (a.response == b.response )
		{
			if( a.pt.x == b.pt.x )
			{
				//assert(a.pt.y != b.pt.y);
				return a.pt.y < b.pt.y;
			}
			return a.pt.x < b.pt.x;
		}
		return a.response > b.response;
	});

	std::unordered_map<int, int> keypointToSegm;
	int compCounter = 0;
	int lettersCount = 0;
	vector<cv::Mat>& imagePyramid = ftDetector->getImagePyramid();

	std::vector<std::unordered_map<int, int> > keypointHash;
	keypointHash.resize(imagePyramid.size());
	keypointStrokes.clear();
	for(size_t i = 0; i < img1_keypoints.size(); i++)
	{
		if(img1_keypoints[i].count == 6)
			continue;
		double kpscale = ftDetector->getLevelScale(img1_keypoints[i].octave);
		size_t index = ( (int) roundf(img1_keypoints[i].pt.x * kpscale) + roundf((int) img1_keypoints[i].pt.y * kpscale) * img.cols);
		keypointHash[img1_keypoints[i].octave][index] = i;
		if( img1_keypoints[i].octave > 0)
		{
			kpscale = ftDetector->getLevelScale(img1_keypoints[i].octave -1);
			index = ( (int) roundf(img1_keypoints[i].pt.x * kpscale) + roundf((int) img1_keypoints[i].pt.y * kpscale) * img.cols);
			if( keypointHash[img1_keypoints[i].octave - 1].find(index) == keypointHash[img1_keypoints[i].octave - 1].end() )
				keypointHash[img1_keypoints[i].octave - 1][index] = i;
		}
	}
	vector<double> scales = ftDetector->getScales();

	if(segmPyramid.size() != imagePyramid.size() || segmPyramid[0].cols != img.cols || segmPyramid[0].rows != img.rows)
	{
		segmPyramid.resize(imagePyramid.size());
		idPyramid.resize(imagePyramid.size());
		for( size_t k = 0; k < pixelsOffset.size(); k++ )
			delete [] pixelsOffset[k];
		pixelsOffset.resize(imagePyramid.size());
		for(size_t i = 0; i < imagePyramid.size(); i++)
		{
			segmPyramid[i] = cv::Mat::zeros(imagePyramid[i].rows, imagePyramid[i].cols, CV_8UC1);
			idPyramid[i] = cv::Mat::zeros(imagePyramid[i].rows, imagePyramid[i].cols, CV_32SC1);
			int* pixel = new int[34];
			int corners[34], cornersOut[34], pixelcheck[24], pixelIndex[24], pixelcheck16[16], pixelCounter[34];
			if( imagePyramid[0].type() == CV_8UC1 )
				cmp::makeOffsets(pixel, corners, cornersOut, (int)imagePyramid[i].step[0], 12, pixelIndex, pixelcheck, pixelcheck16);
			else
				cmp::makeOffsetsC(pixel, pixelCounter, corners, (int)imagePyramid[i].step, 12, pixelcheck, pixelcheck16);
			pixelsOffset[i] = pixel;
		}
	}else
	{
		for(size_t i = 0; i < imagePyramid.size(); i++)
		{
			segmPyramid[i] = cv::Scalar(0, 0, 0);
			idPyramid[i] = cv::Scalar(-1, -1, -1);
		}
	}

	std::vector<cv::Point> ccomp;
	for(size_t i = 0; i < img1_keypoints.size(); i++)
	{
		LetterCandidate* prev = NULL;
		int prevComp = -1;
		for( SegmentOption& segOpt : segmentOptions )
		{

			if(img1_keypoints[i].count == 6)
				continue;
			cv::Rect roi;
			cv::Mat segmImg;

			int pyramidIndex = img1_keypoints[i].octave;
			int pyramidIndexOffset = pyramidIndex;
			double sf = 1 / ftDetector->getLevelScale(pyramidIndex);
			cv::Point2f ptScaled =  img1_keypoints[i].pt;
			ptScaled.x /= sf;
			ptScaled.y /= sf;
			ptScaled.x = round(ptScaled.x);
			ptScaled.y = round(ptScaled.y);
			cv::Point2f ptMaxDiffScaled =  img1_keypoints[i].intensityOut;
			ptMaxDiffScaled.x /= sf;
			ptMaxDiffScaled.x = round(ptMaxDiffScaled.x);
			ptMaxDiffScaled.y /= sf;
			ptMaxDiffScaled.y = round(ptMaxDiffScaled.y);
			int compNo = 0;
			long threshold  = 0;
			int area = 0;
			cv::Scalar intensityOut;
			cv::Scalar intensityIn;
			int projection = 0;
			int kpCount = img1_keypoints[i].count;
			if(kpCount == 5)
				kpCount = 2;


			intensityIn = imagePyramid[pyramidIndex].at<uchar>((int) ptScaled.y, (int) ptScaled.x);
			int pixVal = 0;
			std::vector<int> keypointIds;
			int maxIntentsity = imagePyramid[pyramidIndex].at<uchar>((int) ptMaxDiffScaled.y, (int) ptMaxDiffScaled.x);
			if(imagePyramid[pyramidIndex].type() == CV_8UC3)
			{
				int strokeCounter = i;
				int strokeArea;

				threshold = img1_keypoints[i].response;
				//std::cout << "Type: " << (int) img1_keypoints[i].type << std::endl;
				if( img1_keypoints[i].type == 1 ){
					threshold -=  INT_OFFSET;
					cv::Mat tmp;
					switch(img1_keypoints[i].channel){
					case 0:
						segmentStroke(imagePyramid[pyramidIndex], segmPyramid[pyramidIndex], idPyramid[pyramidIndex], img1_keypoints[i], sf, ColourDistanceRGBP<0>, threshold, strokeCounter, tmp, strokeArea, roi, keypointStrokes[i], true, pixelsOffset[pyramidIndex], maxStrokeLength );
						break;
					case 1:
						segmentStroke(imagePyramid[pyramidIndex], segmPyramid[pyramidIndex], idPyramid[pyramidIndex], img1_keypoints[i], sf, ColourDistanceRGBP<1>, threshold, strokeCounter, tmp, strokeArea, roi, keypointStrokes[i], true, pixelsOffset[pyramidIndex], maxStrokeLength );
						break;
					case 2:
						segmentStroke(imagePyramid[pyramidIndex], segmPyramid[pyramidIndex], idPyramid[pyramidIndex], img1_keypoints[i], sf, ColourDistanceRGBP<2>, threshold, strokeCounter, tmp, strokeArea, roi, keypointStrokes[i], true, pixelsOffset[pyramidIndex], maxStrokeLength );
						break;
					}

				}else{
					projection = 1;
					threshold = - img1_keypoints[i].response + INT_OFFSET;
					cv::Mat tmp;
					switch(img1_keypoints[i].channel){
					case 0:
						segmentStroke(imagePyramid[pyramidIndex], segmPyramid[pyramidIndex], idPyramid[pyramidIndex], img1_keypoints[i], sf, ColourDistanceRGBIP<0>, threshold, strokeCounter, tmp, strokeArea, roi, keypointStrokes[i], true, pixelsOffset[pyramidIndex], maxStrokeLength );
						break;
					case 1:
						segmentStroke(imagePyramid[pyramidIndex], segmPyramid[pyramidIndex], idPyramid[pyramidIndex], img1_keypoints[i], sf, ColourDistanceRGBIP<1>, threshold, strokeCounter, tmp, strokeArea, roi, keypointStrokes[i], true, pixelsOffset[pyramidIndex], maxStrokeLength );
						break;
					case 2:
						segmentStroke(imagePyramid[pyramidIndex], segmPyramid[pyramidIndex], idPyramid[pyramidIndex], img1_keypoints[i], sf, ColourDistanceRGBIP<2>, threshold, strokeCounter, tmp, strokeArea, roi, keypointStrokes[i], true, pixelsOffset[pyramidIndex], maxStrokeLength );
						break;
					}
				}

				compNo = floodFill( buffer, idPyramid[pyramidIndex], imagePyramid[pyramidIndex], ptScaled, img1_keypoints[i].channel, sf,
						compCounter, threshold, kpCount * maxComponentSize, minCompSize, segmImg, segmPyramid[pyramidIndex], roi, area, keypointHash[pyramidIndex], keypointIds, true, segmentGrad, img.cols);
				keypointIds.push_back(i);

				//cv::imshow("ts", segmPyramid[pyramidIndex]);
				//cv::waitKey(0);

				//compNo = segmentCompRGB(queue, ptScaled, imagePyramid[pyramidIndex], segmPyramid[pyramidIndex], idPyramid[pyramidIndex], threshold, compCounter, ccomp, roi, segmImg, maxComponentSize, true);
			}else
			{
				pixVal = imagePyramid[pyramidIndex].at<uchar>((int) ptScaled.y, (int) ptScaled.x);
				int dp = abs(pixVal - img1_keypoints[i].maxima);
				int threshold = img1_keypoints[i].response;
				threshold =  1 * (maxIntentsity - pixVal) / 3; //TODO remove !!!
				//cv::Scalar intensityIn2 = imagePyramid[pyramidIndex].at<uchar>((int) ptMostSameScaled.y, (int) ptMostSameScaled.x);
				intensityOut = imagePyramid[pyramidIndex].at<uchar>((int) ptMaxDiffScaled.y, (int) ptMaxDiffScaled.x);
				std::vector<std::vector<cv::Ptr<StrokeDir> > > strokes;
				if( intensityIn.val[0] <  maxIntentsity )
				{
					if(keypointsPixels.size() > 0 )
						threshold = img1_keypoints[i].response - MAX(dp, 2);
					else
						threshold = img1_keypoints[i].response - INT_OFFSET;
					if( img1_keypoints[i].count != 5 && prev == NULL && keypointsPixels.size() == 0 )
					{
						keypointStrokes[i] = std::vector<std::vector<cv::Ptr<StrokeDir> > > ();
						int strokeArea;
						cv::Mat tmp;
						int strokeCounter = i;
						int64 startTime = cv::getTickCount();
						segmentStroke(imagePyramid[pyramidIndex], segmPyramid[pyramidIndex], idPyramid[pyramidIndex], img1_keypoints[i], sf, ColourDistanceGray, edgeThreshold, strokeCounter, tmp, strokeArea, roi, keypointStrokes[i], true, pixelsOffset[pyramidIndex], maxStrokeLength );
						strokesTime += cv::getTickCount() - startTime;
					}
					//compNo = segmentStroke(imagePyramid[pyramidIndex], segmPyramid[pyramidIndex], idPyramid[pyramidIndex], img1_keypoints[i], sf, ColourDistanceGrayP, threshold, compCounter, segmImg, area, roi, strokes);
					//compNo = segmentComp(queue, ptScaled, imagePyramid[pyramidIndex], segmPyramid[pyramidIndex], idPyramid[pyramidIndex], img1_keypoints[i].response + pixVal , compCounter, ccomp, roi, segmImg, maxComponentSize * 3 * (img1_keypoints[i].octave + 1 ), true );
				}else
				{
					projection = 1;
					if(keypointsPixels.size() > 0 )
						threshold = -img1_keypoints[i].response + MAX(dp, 2);
					else
						threshold = -img1_keypoints[i].response + INT_OFFSET;
					if( img1_keypoints[i].count != 5 && prev == NULL && keypointsPixels.size() == 0)
					{
						keypointStrokes[i] = std::vector<std::vector<cv::Ptr<StrokeDir> > >();
						int strokeArea;
						cv::Mat tmp;
						int strokeCounter = i;
						int64 startTime = cv::getTickCount();
						segmentStroke(imagePyramid[pyramidIndex], segmPyramid[pyramidIndex], idPyramid[pyramidIndex], img1_keypoints[i], sf, ColourDistanceGrayI, edgeThreshold, strokeCounter, tmp, strokeArea, roi, keypointStrokes[i], true, pixelsOffset[pyramidIndex], maxStrokeLength );
						strokesTime += cv::getTickCount() - startTime;
					}

					//compNo = segmentStroke(imagePyramid[pyramidIndex], segmPyramid[pyramidIndex], idPyramid[pyramidIndex], img1_keypoints[i], sf, ColourDistanceGrayIP, threshold, compCounter, segmImg, area, roi, strokes);
					//compNo = segmentCompNegative(queue, ptScaled, imagePyramid[pyramidIndex], segmPyramid[pyramidIndex], idPyramid[pyramidIndex], img1_keypoints[i].response - pixVal, compCounter, ccomp, roi, segmImg, maxComponentSize * 3 * (img1_keypoints[i].octave + 1 ), true );
				}
				if( abs(threshold) > 70 && segOpt.scoreFactor == 1.0)
					threshold = 1 * threshold / 2;

				if( intensityIn.val[0] <  maxIntentsity && ((this->segmentKeyPoints & 1) == 0 ) )
					continue;
				if( intensityIn.val[0] >  maxIntentsity && ((this->segmentKeyPoints & 2) == 0 ) )
					continue;

				if(img1_keypoints[i].isMerged)
					continue;


				if(pyramidIndex > 0)
				{
					pyramidIndexOffset += segmentLevelOffset;
					sf = 1 / ftDetector->getLevelScale(pyramidIndexOffset);
					ptScaled =  img1_keypoints[i].pt;
					ptScaled.x /= sf;
					ptScaled.y /= sf;
				}

				//compNo = segmentComp(queue, ptScaled, imagePyramid[pyramidIndex], segmPyramid[pyramidIndex], idPyramid[pyramidIndex], threshold, compCounter, ccomp, roi, segmImg, maxComponentSize, true);
				compNo = floodFill( buffer, idPyramid[pyramidIndexOffset], imagePyramid[pyramidIndexOffset], ptScaled, img1_keypoints[i].channel, sf,
						compCounter, threshold * segOpt.scoreFactor, maxComponentSize, minCompSize, segmImg, segmPyramid[pyramidIndexOffset], roi, area, keypointHash[pyramidIndex], keypointIds, true, segOpt.segmentationType, img.cols);
				/*
				std::cout << "Threshold: " << threshold << ", pix val:" << pixVal << ", cn:" << compNo << ", x:" << img1_keypoints[i].pt.x << "," << img1_keypoints[i].pt.y << std::endl;
				cv::imshow("ts", segmPyramid[pyramidIndex]);
				cv::waitKey(0);
				*/
			}

			if(compNo == -1)
			{
				continue;
			}
			if( (area * sf * sf)  < minCompSize )
				continue;
			//if( imagePyramid[pyramidIndex].rows / (roi.height + roi.width) > 80 )
			//	continue;
			LetterCandidate* ref = NULL;
			if(!segmImg.empty())
			{
				roi.x = roundf((roi.x) * sf);
				roi.y = roundf((roi.y) * sf);
				roi.width = roundf(roi.width * sf);
				roi.height = roundf(roi.height * sf);
				if(MAX(roi.height, roi.width) < minHeight)
					continue;
				compNo = letterCandidates.size();
				letterCandidates.push_back(LetterCandidate(segmImg, roi, intensityIn, intensityIn, area, img1_keypoints[i], projection,  sf));
				ref = &letterCandidates[compNo];
				ref->intensityInt = intensityIn;
				ref->intensityOut = intensityOut;
				ref->keypointIds = keypointIds;
				keypointToSegm[i] = compNo;

				for( auto kpid : keypointIds){
					FastKeyPoint& kpCheck = img1_keypoints[kpid];
					assert( kpCheck.pt.x <=  (roi.x + roi.width + 5));
					if(kpCheck.octave != img1_keypoints[i].octave || kpid == (int) i)
						continue;
					cv::Point2f ptScaled2 =  kpCheck.pt;
					ptScaled2.x /= sf;
					ptScaled2.y /= sf;
					cv::Scalar intCheck = imagePyramid[pyramidIndex].at<uchar>((int) ptScaled2.y, (int) ptScaled2.x);
					if( abs(abs( pixVal - intCheck.val[0]) + abs( img1_keypoints[i].response - kpCheck.response) ) <= this->delataIntResegment )
					{
						kpCheck.isMerged = true;
						letterCandidates[compNo].mergedKeypoints += 1;
					}else if( keypointToSegm.find(kpid) != keypointToSegm.end())
					{
						int com2 = keypointToSegm[kpid];
						LetterCandidate* ref2 = &letterCandidates[com2];
						while(ref2->duplicate != -1)
						{
							com2 = ref2->duplicate;
							ref2 = &letterCandidates[com2];
						}
						if( com2 == compNo )
							continue;
						cv::Rect minBox = ref2->bbox & roi;
						cv::Rect maxBox = ref2->bbox | roi;
						float ratio = minBox.area() / (float)  maxBox.area();
						float areaRatio = MIN(area, ref2->area) / (float) MAX(area, ref2->area);
						if( ratio > 0.8 && areaRatio > 0.8)
						{
							if( ref2->area > area)
							{
								ref->setDuplicate(*ref2, com2, compNo);
								ref = ref2;
								compNo = com2;
							}
							else
								ref2->setDuplicate(*ref, compNo, com2);
						}
					}
				}

				if(prev != NULL && ref != NULL)
				{
					cv::Rect overlap = ref->bbox | prev->bbox;
					float ratio1 = ref->bbox.area() / (float) overlap.area();
					float areaRatio = MIN(area, prev->area) / (float) MAX(area, prev->area);
					if( ratio1 > 0.9 && areaRatio > 0.9)
					{
						if( prev->area > area )
						{
							ref->setDuplicate(*prev, prevComp, compNo);
							ref = prev;
							compNo = prevComp;
						}
						else
							prev->setDuplicate(letterCandidates[compNo], compNo, prevComp);

					}else{
						cv::Rect minBox = letterCandidates[compNo].bbox | prev->bbox;
						float ratio = minBox.area() / (float)  letterCandidates[compNo].bbox.area();
						if( ratio > 0.8)
						{
							letterCandidates[compNo].parents.insert(prevComp);
							prev->childs.insert(compNo);
							continue;
						}

						//float area2 = prev->bbox.area();
						if( letterCandidates[compNo].contains(*prev) )
						{
							prev->parents.insert(compNo);
							letterCandidates[compNo].childs.insert(prevComp);
						}else if( prev->contains(letterCandidates[compNo]) )
						{
							letterCandidates[compNo].parents.insert(prevComp);
							prev->childs.insert(compNo);
						}else
						{
							std::cout << "Unknown comp!!\n";
						}

					}
				}

				prev = &letterCandidates[compNo];
				prevComp = compNo;
			}
			//if( charClassifier->classifyLetter(letterCandidates[compNo], debugImage) )
			//if(charClassifier->extractLineFeatures(letterCandidates[compNo]))
			lettersCount++;
		}

	}

	classifyLetters(img1_keypoints, keypointsPixels, scales, letters, debugImage);

}


template <class T>
inline void hash_combine(std::size_t& seed, const T& v)
{
	std::hash<T> hasher;
	seed ^= hasher(v) + 0x9e3779b9 + (seed<<6) + (seed>>2);
}

struct pairhash {
public:
	template <typename T, typename U>
	std::size_t operator()(const std::pair<T, U> &x) const
	{
		std::size_t seed = 0;
		hash_combine(seed, x.first);
		hash_combine(seed, x.second);
		return seed;
	}
};

inline bool less(uchar& a, int& b)
{
	return a < b;
}

inline bool greater(uchar& a, int& b)
{
	return a > b;
}

void PyramidSegmenter::segmentStrokes(cv::Mat& img, std::vector<cmp::FastKeyPoint>& img1_keypoints, std::unordered_multimap<int, std::pair<int, int> >& keypointsPixels, std::vector<cmp::LetterCandidate*>& letters, cv::Mat debugImage, int minHeight)
{
	classificationTime = 0;
	if(!charClassifier.empty())
		charClassifier->classificationTime = 0;
	letterCandidates.clear();
	letterCandidates.reserve(img1_keypoints.size() / 3);

	std::unordered_multimap< std::pair<int, int>,  int, pairhash> revKeypointsPixels;
	for( auto& it : keypointsPixels)
	{
		revKeypointsPixels.insert({it.second, it.first});
	}

	std::unordered_map<int, int> keypointToClassId;
	for(size_t i = 0; i < img1_keypoints.size(); i++)
	{
		keypointToClassId[img1_keypoints[i].class_id] = i;
	}

	vector<cv::Mat>& imagePyramid = ftDetector->getImagePyramid();
	vector<double> scales = ftDetector->getScales();
	std::unordered_multimap<int, int> toMerge;
	std::unordered_map<int, int> keypointToLetter;
	for(size_t i = 0; i < img1_keypoints.size(); i++)
	{
		cmp::FastKeyPoint seed = img1_keypoints[i];
		std::pair <std::unordered_multimap<int,std::pair<int, int> >::iterator, std::unordered_multimap<int,std::pair<int, int>>::iterator> ret;
		ret = keypointsPixels.equal_range(seed.class_id);
		int threshold =  seed.maxima + ftDetector->getEdgeThreshold();
		std::vector< std::pair<int, int> > segmentation;
		cv::Mat& img = imagePyramid[seed.octave];
		size_t pixelsOffset = 0;

#ifdef VERBOSE
		cv::Mat& segm = this->segmPyramid[seed.octave];
#endif

		cv::Point2f ptScaled =  seed.pt;
		double sf = 1 / ftDetector->getLevelScale(seed.octave);
		ptScaled.x /= sf;
		ptScaled.y /= sf;
		ptScaled.x = round(ptScaled.x);
		ptScaled.y = round(ptScaled.y);
		segmentation.push_back({ptScaled.x, ptScaled.y});

		bool (*cmpf)(uchar&, int&) = &less;
		if( seed.type == 0 )
		{
			cmpf = &greater;
			threshold =  seed.maxima - ftDetector->getEdgeThreshold();
		}

		cv::Rect roi(ptScaled.x, ptScaled.y, ptScaled.x, ptScaled.y);
		for (std::unordered_multimap<int,std::pair<int, int> >::iterator it=ret.first; it!=ret.second; ++it)
		{
			segmentation.push_back(it->second);
			roi.x = MIN(roi.x, it->second.first);
			roi.y = MIN(roi.y, it->second.second);
			roi.width = MAX(roi.width, it->second.first);
			roi.height = MAX(roi.height, it->second.second);
#ifdef VERBOSE
			if(seed.type == 1)
				segm.at<uchar>(it->second.second, it->second.first) = 255;
#endif
		}

		static const int steps[][2] =
		{
			{1,  0}, { 0,  1}, {-1,  0}, { 0,  -1}
		};

		for(int k = 0; k < 9; k++ )
		{
			size_t stepCount = segmentation.size();
			for (size_t i = pixelsOffset; i < stepCount; i++)
			{
				for( int s = 0; s < 4; s++)
				{
					std::pair<int, int> pos = segmentation[i];
					pos.first += steps[s][0];
					if( pos.first >= img.cols || pos.first < 0 )
						continue;
					pos.second += steps[s][1];
					if( pos.second >= img.rows || pos.second < 0 )
						continue;
					std::pair <std::unordered_multimap<std::pair<int, int>, int, pairhash >::iterator, std::unordered_multimap<std::pair<int, int>, int, pairhash>::iterator> q;
					q =  revKeypointsPixels.equal_range(pos);
					bool conf = false;
					for (std::unordered_multimap<std::pair<int, int>, int, pairhash >::iterator it=q.first; it!=q.second; it++)
					{
						if(it->second != seed.class_id)
						{
							assert(keypointToClassId[it->second] < (int) img1_keypoints.size());
							cmp::FastKeyPoint merge = img1_keypoints[keypointToClassId[it->second]];
							if(merge.type == seed.type && seed.octave == merge.octave)
							{
								toMerge.insert({seed.class_id, it->second});
								//std::cout << "Merge ... \n";
							}
						}else{
							conf = true;
						}
					}
					if(!conf){
						if(  cmpf(img.at<uchar>(pos.second, pos.first), threshold) )
						{
							segmentation.push_back(pos);
							assert(pos.second >= 0);
							roi.x = MIN(roi.x, pos.first);
							roi.y = MIN(roi.y, pos.second);
							roi.width = MAX(roi.width, pos.first);
							roi.height = MAX(roi.height, pos.second);
#ifdef VERBOSE
							if(seed.type == 1)
								segm.at<uchar>(pos.second, pos.first) = 255;
#endif
						}
					}
				}
				pixelsOffset += 1;
			}
		}
#ifdef VERBOSE
			cv::imshow("segm", segm);
			cv::imwrite("/tmp/segm.png", segm);
			if(seed.type == 1)
				cv::waitKey(0);
#endif

		cv::Scalar intensityIn = imagePyramid[seed.octave].at<uchar>((int) ptScaled.y, (int) ptScaled.x);

		cv::Point2f ptScaledOut =  seed.intensityOut;
		ptScaledOut.x /= sf;
		ptScaledOut.y /= sf;
		ptScaledOut.x = round(ptScaledOut.x);
		ptScaledOut.y = round(ptScaledOut.y);

		cv::Scalar intensityOut = imagePyramid[seed.octave].at<uchar>((int) ptScaledOut.y, (int) ptScaledOut.x);
		roi.width = roi.width - roi.x + 1;
		roi.height = roi.height - roi.y + 1;

		cv::Mat segmImg = cv::Mat::zeros(roi.height, roi.width, CV_8UC1);
		for (size_t i = 0; i < segmentation.size(); i++)
		{
			segmImg.at<uchar>(segmentation[i].second - roi.y, segmentation[i].first - roi.x) = 255;
		}
		keypointToLetter[seed.class_id] = letterCandidates.size();
		assert(roi.x >= 0);
		assert(roi.y >= 0);
		letterCandidates.push_back(LetterCandidate(segmImg, roi, intensityIn, intensityOut, segmentation.size(), img1_keypoints[i], seed.type,  sf));
		letterCandidates.back().keypointIds.push_back(seed.class_id);

		letterCandidates.back().intensityInt = intensityIn;
		letterCandidates.back().intensityOut = intensityOut;
	}

	for( auto it : toMerge  )
	{
		int c1 = keypointToLetter[it.first];
		LetterCandidate* r1 = &letterCandidates[c1];
		int c2 = keypointToLetter[it.second];
		LetterCandidate* r2 = &letterCandidates[c2];
		while( r1->duplicate != -1)
		{
			c1 = r1->duplicate;
			r1 = &letterCandidates[r1->duplicate];
		}
		while( r2->duplicate != -1)
		{
			c2 = r2->duplicate;
			r2 = &letterCandidates[r2->duplicate];
		}
		if( c1 == c2 )
			continue;
#ifdef VERBOSE
		cv::imshow("r1", r1->mask);
		cv::imshow("r2", r2->mask);
#endif
		cv::Rect r = r1->bbox | r2->bbox;
		cv::Mat mask = cv::Mat::zeros(r.height, r.width, CV_8UC1);
		cv::Rect roi1 = r1->bbox;
		roi1.x -= r.x;
		roi1.y -= r.y;
		cv::bitwise_or(mask(roi1), r1->mask, mask(roi1));
		cv::Rect roi2 = r2->bbox;
		roi2.x -= r.x;
		roi2.y -= r.y;
		cv::bitwise_or(mask(roi2), r2->mask, mask(roi2));
#ifdef VERBOSE
		cv::imshow("r21", mask);
		cv::waitKey(0);
#endif

		assert(r.x >= 0);
		assert(r.y >= 0);
		if( r1->area > r2->area ){
			r1->mask = mask;
			r1->bbox = r;
			r1->area = cv::countNonZero(mask);
			r1->keypointIds.insert(r1->keypointIds.end(), r2->keypointIds.begin(), r2->keypointIds.end());
			r1->duplicates.push_back(c1);
			r2->duplicate = c1;

		}else{
			r2->mask = mask;
			r2->bbox = r;
			r2->area = cv::countNonZero(mask);
			r2->keypointIds.insert(r2->keypointIds.end(), r1->keypointIds.begin(), r1->keypointIds.end());
			r2->duplicates.push_back(c2);
			r1->duplicate = c2;
		}
	}

	for(std::vector<LetterCandidate>::iterator i = letterCandidates.begin(); i != letterCandidates.end(); )
	{
		if( i->area  < 4 || (i->area  * i->scaleFactor * i->scaleFactor)  < minCompSize )
		{
			i = letterCandidates.erase(i);
			continue;
		}
		LetterCandidate& r = *i;
		r.bbox.x = roundf((r.bbox.x) * r.scaleFactor);
		r.bbox.y = roundf((r.bbox.y) * r.scaleFactor);
		r.bbox.width = roundf(r.bbox.width * r.scaleFactor);
		r.bbox.height = roundf(r.bbox.height * r.scaleFactor);
		i++;
	}

	classifyLetters(img1_keypoints, keypointsPixels, scales, letters, debugImage);

}


} /* namespace cmp */
