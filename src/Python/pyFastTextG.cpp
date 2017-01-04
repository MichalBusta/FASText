/*
 * pyFastText.cpp
 *
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

#define PY_ARRAY_UNIQUE_SYMBOL cool_ARRAY_API

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <chrono>

#include "pyFastTextG.h"

#include "../FTPyramid.hpp"
#include "IOUtils.h"
#include "Segmenter.h"
#include "../vis/componentsVis.h"
#include "FastTextLineDetector.h"

#include <iostream>

//#define VERBOSE 1

using namespace cmp;

struct DetectionStat{

	DetectionStat(){
		keypointsCount = 0;
		keypointsTime = 0;
		segmentationTime = 0;
		classificationTime = 0;
		classificationTimeTuples = 0;
		tuplesTime = 0;
		rawClsTime = 0;
		textLineTime = 0;
		wallTime = 0;
		strokesTime = 0;
		gcTime = 0;
	}

	long keypointsCount;
	long keypointsTime;
	long segmentationTime;
	long classificationTime;
	long rawClsTime;
	long textLineTime;
	double wallTime;
	long classificationTimeTuples;
	double tuplesTime;
	long strokesTime;
	long gcTime;
};

DetectionStat detectionStat;

PyArrayObject* find_keypoints(PyArrayObject* img, int numOfDims,  npy_intp* img_dims, int scaleFactor, int nlevels, int edgeThreshold, int keypointTypes, int kMin, int kMax)
{
	cv::Ptr<cmp::FTPyr> detector = new cmp::FTPyr(3000, scaleFactor, nlevels, edgeThreshold, 2, 31, keypointTypes, kMin, kMax);

	int type = CV_8UC3;
	if( numOfDims == 2 )
		type = CV_8UC1;

	cv::Mat srcImg = cv::Mat(img_dims[0], img_dims[1], type, PyArray_DATA(img) );

	std::vector<cmp::FastKeyPoint> keypoints;
	std::unordered_multimap<int, std::pair<int, int> > keypointsPixels;
	detector->detect(srcImg,  keypoints, keypointsPixels);

	npy_intp size_pts[2];
	size_pts[0] = keypoints.size();
	size_pts[1] = 10;

	PyArrayObject* out = (PyArrayObject *) PyArray_SimpleNew( 2, size_pts, NPY_FLOAT );

	for(size_t i = 0; i < keypoints.size(); i++)
	{
		cmp::FastKeyPoint& fast = keypoints[i];

		float* ptr = (float *) PyArray_GETPTR2(out, i, 0);
		*ptr++ = fast.pt.x;
		*ptr++ = fast.pt.y;
		*ptr++ = fast.octave;
		*ptr++ = fast.response;
		*ptr++ = fast.angle;
		*ptr++ = fast.intensityOut.x;
		*ptr++ = fast.intensityOut.y;
		*ptr++ = fast.intensityIn.x;
		*ptr++ = fast.intensityIn.y;
		*ptr++ = fast.count;
	}
	return out;
}

struct Intences{
	cv::Ptr<cmp::FTPyr> ftDetector;
	cv::Ptr<cmp::Segmenter> segmenter;
};

std::vector<Intences> instances;

int initialize(float scaleFactor, int nlevels, int edgeThreshold, int keypointTypes, int kMin, int kMax,
		const char* charClsFile, int erode, int segmentGrad, int minComponentSize, int instance, float thresholdFactor, int segmDeltaInt)
{
	Intences newInstance;
	std::cout << "Using FT detector edgeThreshold = " << edgeThreshold << ", kpTypes: " << keypointTypes << std::endl;
	int keypintsCount = 4000;
	newInstance.ftDetector = new cmp::FTPyr(keypintsCount, scaleFactor, nlevels, edgeThreshold, keypointTypes, kMin, kMax, false, erode == 1, false);

	std::cout << "Using PyramidSegmenter, segmDeltaInt: " << segmDeltaInt << std::endl;
	newInstance.segmenter = new cmp::PyramidSegmenter(newInstance.ftDetector, new cmp::CvBoostCharClassifier(charClsFile), MAX_COMP_SIZE, MIN_COMP_SIZE, thresholdFactor, segmDeltaInt);

	if(minComponentSize > 0)
		newInstance.segmenter->minCompSize = minComponentSize;
	newInstance.segmenter->segmentGrad = segmentGrad;

	if(instance >= 0)
	{
		if( (int) instances.size() < instance + 1)
			instances.resize(instance + 1);
		std::cout << "New instance " << instance << " with edge threshold: " << edgeThreshold << std::endl;
		instances[instance] = newInstance;
	}
	else
	{
		instances.push_back(newInstance);
		std::cout << "New instance " << instance << " with edge threshold: " << edgeThreshold << std::endl;
	}

	return instances.size() - 1;
}

template<typename T>
PyObject *toPython(const T& input)
{
  PyObject *p = PyInt_FromLong((long) input);
  return p;
}


// for types which are a number and thus can be stored in 1d numpy array
// (thus no need for specializing and using ObjectToNumpy template)
template<typename T>
PyArrayObject *toPython(vector<T> &input, std::true_type){
	size_t N = input.size();

	npy_intp size_pts[1];
	size_pts[0] =  N;

	PyArrayObject* out = (PyArrayObject *) PyArray_SimpleNew( 1, size_pts, NPY_OBJECT );

	for(size_t i = 0; i <  N; i++)
	{
		char *ptr = (char*) PyArray_GETPTR1(out, i);
		PyObject *v = toPython(input[i]);
		PyArray_SETITEM(out, ptr, v);
		Py_DECREF(v);
	}
	return out;
}

template<typename T>
PyArrayObject *toPython(vector<T> &input){
	return toPython<T>(input, std::is_arithmetic<T>());
}

static inline void exportLetter(PyArrayObject* out, int& i, cmp::LetterCandidate& det)
{
	char* ptr = (char*) PyArray_GETPTR2(out, i, 0);
	PyArray_SETITEM(out, ptr, PyInt_FromLong(det.bbox.x));
	ptr = (char*) PyArray_GETPTR2(out, i, 1);
	PyArray_SETITEM(out, ptr, PyInt_FromLong(det.bbox.y));
	ptr = (char*) PyArray_GETPTR2(out, i, 2);
	PyArray_SETITEM(out, ptr, PyInt_FromLong(det.bbox.width));
	ptr = (char*) PyArray_GETPTR2(out, i, 3);
	PyArray_SETITEM(out, ptr, PyInt_FromLong(det.bbox.height));
	ptr = (char*) PyArray_GETPTR2(out, i, 4);
	PyArray_SETITEM(out, ptr, PyInt_FromLong((int) det.keyPoint.pt.x));
	ptr = (char*) PyArray_GETPTR2(out, i, 5);
	PyArray_SETITEM(out, ptr, PyInt_FromLong((int) det.keyPoint.pt.y));
	ptr = (char*) PyArray_GETPTR2(out, i, 6);
	PyArray_SETITEM(out, ptr, PyInt_FromLong((int) det.keyPoint.octave));
	ptr = (char*) PyArray_GETPTR2(out, i, 7);
	PyArray_SETITEM(out, ptr, PyInt_FromLong((int) det.groupAssigned));
	ptr = (char*) PyArray_GETPTR2(out, i, 8);
	PyArray_SETITEM(out, ptr, PyInt_FromLong((int) det.duplicate));
	ptr = (char*) PyArray_GETPTR2(out, i, 9);
	PyArray_SETITEM(out, ptr, PyFloat_FromDouble( det.quality) );

	PyObject *__obj__ = (PyObject *) toPython(det.keypointIds);
	ptr = (char*) PyArray_GETPTR2(out, i, 10);
	PyArray_SETITEM(out, ptr, __obj__); \
	Py_DECREF(__obj__);


}

std::vector<cmp::FastKeyPoint> keypoints;
std::unordered_multimap<int, std::pair<int, int> > keypointsPixels;
cv::Mat lastImage;

//#define UPDATE_BBOX 1
//#define VERBOSE 1
PyArrayObject* get_char_segmentations(PyArrayObject* img, int numOfDims, npy_intp* img_dims, const char* outputDir, const char* imageName, int instance, int minHeight)
{
	int type = CV_8UC3;
	if( numOfDims == 2 )
		type = CV_8UC1;

	cv::Mat srcImg = cv::Mat(img_dims[0], img_dims[1], type, PyArray_DATA(img) );
	lastImage = srcImg;

	double t = (double) cv::getTickCount();
	auto start = std::chrono::system_clock::now();
	instances[instance].ftDetector->detect(srcImg,  keypoints, keypointsPixels);
	std::chrono::duration<double> elapsed_seconds = std::chrono::system_clock::now() - start;
#ifndef NDEBUG
	std::cout << "Detected Keypoints: " << keypoints.size() << " in " <<  ( (double) cv::getTickCount() - t)/ (cv::getTickFrequency()) * 1000 << std::endl;
	cout << "Keypoints duration: " << instances[instance].ftDetector->fastKeypointTime << " ms" << endl;
#endif
	detectionStat.keypointsCount += keypoints.size();
	detectionStat.keypointsTime +=  ( (double) cv::getTickCount() - t)/ (cv::getTickFrequency()) * 1000;

	double t2 = (double) cv::getTickCount();
	start = std::chrono::system_clock::now();
	std::vector<cmp::LetterCandidate*> letters;
	instances[instance].segmenter->getLetterCandidates( srcImg, keypoints, keypointsPixels, letters, cv::Mat(), minHeight );
	elapsed_seconds += std::chrono::system_clock::now() - start;
	detectionStat.strokesTime = instances[instance].segmenter->strokesTime / (cv::getTickFrequency()) * 1000;
	instances[instance].segmenter->strokesTime = 0;
	detectionStat.segmentationTime +=  ( (double) cv::getTickCount() - t2)/ (cv::getTickFrequency()) * 1000;
	detectionStat.classificationTime += ((double) instances[instance].segmenter->getClassificationTime()) / (cv::getTickFrequency()) * 1000;
	if( !instances[instance].segmenter->getCharClassifier().empty() )
		detectionStat.rawClsTime += ((double) instances[instance].segmenter->getCharClassifier()->classificationTime) / (cv::getTickFrequency()) * 1000;

#ifndef NDEBUG
	std::cout << "Detected Letters: " << letters.size() << "/" << instances[instance].segmenter->componentsCount << " in " <<  ( (double) cv::getTickCount() - t2)/ (cv::getTickFrequency()) * 1000 << std::endl;
	std::cout << "Detection Total in: " << ( (double) cv::getTickCount() - t)/ (cv::getTickFrequency()) * 1000 << std::endl;
#endif


	//if( !instances[instance].textLineDetector->getCharClassifier().empty() )
	//		detectionStat.rawClsTime += ((double) instances[instance].textLineDetector->getCharClassifier()->classificationTime) / (cv::getTickFrequency()) * 1000;

	std::vector<cmp::LetterCandidate>& letterCandidatesMap =  instances[instance].segmenter->getLetterCandidates();

	npy_intp size_pts[2];
	size_pts[0] = letterCandidatesMap.size();
	size_pts[1] = 11;
	PyArrayObject* out = (PyArrayObject *) PyArray_SimpleNew( 2, size_pts, NPY_OBJECT );

	cv::Mat segmImg;
	if(outputDir != NULL)
	{
		segmImg = srcImg.clone();
		if( srcImg.type() == CV_8UC1 )
		{
			cv::cvtColor(srcImg, segmImg, cv::COLOR_GRAY2BGR);
		}

		cv::Mat cser = createCSERImage(letters, keypoints, keypointsPixels, segmImg);
		ostringstream os;
		os << outputDir << "/" << imageName << "_chars.png";
		imwrite(os.str(), cser);
	}

	int i = 0;
	int assignedCount = 0;
	for( std::vector<cmp::LetterCandidate>::iterator it = letterCandidatesMap.begin(); it != letterCandidatesMap.end(); it++ )
	{
		//if(!it->second.groupAssigned)
		//	continue;
		assignedCount += it->groupAssigned;
		exportLetter(out, i, *it);
		i++;
		/*
		for( std::vector<int>::iterator dup = det.duplicates.begin(); dup != det.duplicates.end(); dup++ )
		{
			cmp::LetterCandidate& detDup = letterCandidatesMap[*dup];
			letters.push_back(detDup);
			letterIds.push_back(*dup);
			exportLetter(out, i, detDup);
			i++;
		}*/

		if(!it->groupAssigned)
			continue;
		if(outputDir != NULL)
		{
			cv::rectangle(segmImg, it->bbox, cv::Scalar(255, 0, 0));
		}
	}
	detectionStat.wallTime += elapsed_seconds.count();
#ifndef NDEBUG
	std::cout << "Wall-Time duration: " << elapsed_seconds.count() << "s\n";
#endif
	return out;
}

PyArrayObject* get_keypoint_strokes(int keypointId, int instance)
{
	if(keypointsPixels.size() > 0)
	{
		std::pair <std::unordered_multimap<int,std::pair<int, int> >::iterator, std::unordered_multimap<int,std::pair<int, int>>::iterator> ret;
		ret = keypointsPixels.equal_range(keypoints[keypointId].class_id);

		npy_intp size_pts[2];
		size_pts[0] = std::distance(ret.first, ret.second);
		size_pts[1] = 2;
		PyArrayObject* out = (PyArrayObject *) PyArray_SimpleNew( 2, size_pts, NPY_OBJECT );
		int strokesCount = 0;

		for (std::unordered_multimap<int,std::pair<int, int> >::iterator it=ret.first; it!=ret.second; it++)
		{
			char* ptr = (char*) PyArray_GETPTR2(out, strokesCount, 0);
			PyArray_SETITEM(out, ptr, PyInt_FromLong(it->second.first));
			ptr = (char*) PyArray_GETPTR2(out, strokesCount, 1);
			PyArray_SETITEM(out, ptr, PyInt_FromLong(it->second.second));
			strokesCount++;
		}
		return out;

	}

	std::vector<std::vector<cv::Ptr<StrokeDir> > >& strokes = instances[instance].segmenter->keypointStrokes[keypointId];
	npy_intp size_pts[2];
	int strokesCount = 0;
	for( size_t i = 0; i < strokes.size(); i++ )
	{
		strokesCount += strokes[i].size();
	}
	size_pts[0] = strokesCount;

	size_pts[1] = 4;
	PyArrayObject* out = (PyArrayObject *) PyArray_SimpleNew( 2, size_pts, NPY_OBJECT );
	strokesCount = 0;
	for( size_t i = 0; i < strokes.size(); i++ )
	{
		for( size_t j = 0; j < strokes[i].size(); j++ )
		{
			char* ptr = (char*) PyArray_GETPTR2(out, strokesCount, 0);
			PyArray_SETITEM(out, ptr, PyInt_FromLong(strokes[i][j]->center.x));
			ptr = (char*) PyArray_GETPTR2(out, strokesCount, 1);
			PyArray_SETITEM(out, ptr, PyInt_FromLong(strokes[i][j]->center.y));
			ptr = (char*) PyArray_GETPTR2(out, strokesCount, 2);
			PyArray_SETITEM(out, ptr, PyInt_FromLong(strokes[i][j]->direction.x));
			ptr = (char*) PyArray_GETPTR2(out, strokesCount, 3);
			PyArray_SETITEM(out, ptr, PyInt_FromLong(strokes[i][j]->direction.y));
			strokesCount++;
		}
	}
	return out;
}

PyArrayObject* get_last_detection_keypoints()
{
	npy_intp size_pts[2];
	size_pts[0] = keypoints.size();
	size_pts[1] = 12;

	PyArrayObject* out = (PyArrayObject *) PyArray_SimpleNew( 2, size_pts, NPY_FLOAT );
	for(size_t i = 0; i < keypoints.size(); i++)
	{
		cmp::FastKeyPoint& fast = keypoints[i];

		float* ptr = (float *) PyArray_GETPTR2(out, i, 0);
		*ptr++ = fast.pt.x;
		*ptr++ = fast.pt.y;
		*ptr++ = fast.octave;
		*ptr++ = fast.response;
		*ptr++ = fast.angle;
		*ptr++ = fast.intensityOut.x;
		*ptr++ = fast.intensityOut.y;
		*ptr++ = fast.intensityIn.x;
		*ptr++ = fast.intensityIn.y;
		*ptr++ = fast.count;
		*ptr++ = fast.type;
		*ptr++ = fast.channel;
	}
	return out;
}

PyArrayObject* get_last_detection_orb_keypoints()
{

#ifdef OPENCV_24
	cv::Ptr<cv::ORB> detector = new cv::ORB(4000, 1.6, 8, 13);
#else
	cv::Ptr<cv::ORB> detector = cv::ORB::create(4000, 1.6, 8, 13);
#endif

	double t = (double) cv::getTickCount();
	std::vector<cv::KeyPoint> keypoints;
	detector->detect(lastImage,  keypoints);
	double time =  ( (double) cv::getTickCount() - t)/ (cv::getTickFrequency()) * 1000;
	npy_intp size_pts[2];
	size_pts[0] = keypoints.size();
	size_pts[1] = 10;

	PyArrayObject* out = (PyArrayObject *) PyArray_SimpleNew( 2, size_pts, NPY_FLOAT );
	//Py_INCREF(out);

	for(size_t i = 0; i < keypoints.size(); i++)
	{
		cv::KeyPoint& fast = keypoints[i];

		float* ptr = (float *) PyArray_GETPTR2(out, i, 0);
		*ptr++ = fast.pt.x;
		*ptr++ = fast.pt.y;
		*ptr++ = fast.octave;
		*ptr++ = fast.response;
		*ptr++ = fast.angle;
		*ptr++ = 0;
		*ptr++ = 0;
		*ptr++ = 0;
		*ptr++ = 0;
		*ptr++ = time;
	}
	return out;
}

PyArrayObject* get_detection_stat()
{
	npy_intp size_pts[1];
	size_pts[0] = 11;

	PyArrayObject* out = (PyArrayObject *) PyArray_SimpleNew( 1, size_pts, NPY_FLOAT );
	//Py_INCREF(out);


	float* ptr = (float *) PyArray_GETPTR1(out, 0);
	*ptr++ = (float) detectionStat.keypointsCount;
	*ptr++ = (float) detectionStat.keypointsTime;
	*ptr++ = (float) detectionStat.segmentationTime;
	*ptr++ = (float) detectionStat.classificationTime;
	*ptr++ = (float) detectionStat.rawClsTime;
	*ptr++ = (float) detectionStat.textLineTime;
	*ptr++ = (float) detectionStat.wallTime;
	*ptr++ = (float) detectionStat.classificationTimeTuples;
	*ptr++ = (float) detectionStat.tuplesTime;
	*ptr++ = (float) detectionStat.strokesTime;
	*ptr++ = (float) detectionStat.gcTime;

	detectionStat.keypointsCount = 0;
	detectionStat.keypointsTime = 0;
	detectionStat.segmentationTime = 0;
	detectionStat.classificationTime = 0;
	detectionStat.rawClsTime = 0;
	detectionStat.textLineTime = 0;
	detectionStat.wallTime = 0;
	detectionStat.classificationTimeTuples = 0;
	detectionStat.tuplesTime = 0;
	detectionStat.strokesTime = 0;
	detectionStat.gcTime = 0;
	return out;
}

PyArrayObject* get_image_at_scale(int level, int instance)
{
	cv::Mat img = instances[instance].ftDetector->getImagePyramid()[level];
	cv::Mat out = img.clone();
#ifdef OPENCV_24
	out.refcount += 1;
#else
	out.addref();
#endif

	if( img.type() == CV_8UC1 )
	{
		npy_intp size_pts[2];
		size_pts[0] = img.rows;
		size_pts[1] = img.cols;

		PyArrayObject* imgMask = (PyArrayObject *) PyArray_SimpleNewFromData( 2, size_pts, NPY_UINT8, out.data );
		Py_INCREF(imgMask);
		return imgMask;
	}else
	{
		npy_intp size_pts[3];
		size_pts[0] = img.rows;
		size_pts[1] = img.cols;
		size_pts[2] = 3;

		PyArrayObject* imgMask = (PyArrayObject *) PyArray_SimpleNewFromData( 3, size_pts, NPY_UINT8, out.data );
		Py_INCREF(imgMask);
		return imgMask;
	}
}

PyArrayObject* get_image_scales(int instance)
{
	vector<double> scales = instances[instance].ftDetector->getScales();
	npy_intp size_pts[1];
	size_pts[0] = scales.size();


	PyArrayObject* out = (PyArrayObject *) PyArray_SimpleNew( 1, size_pts, NPY_FLOAT );
	//Py_INCREF(out);

	float* ptr = (float *) PyArray_GETPTR1(out, 0);
	for(size_t i = 0; i < scales.size(); i++)
	{
		*ptr = scales[i];
		ptr++;
	}

	return out;
}

PyArrayObject* get_segmentation_mask(int maskId)
{
	cmp::LetterCandidate& det = instances[0].segmenter->getLetterCandidates()[maskId];
	cv::Mat mask = det.mask;
	cv::Mat out = mask.clone();
#ifdef OPENCV_24
	out.refcount += 1;
#endif
	npy_intp size_pts[2];
	size_pts[0] = out.rows;
	size_pts[1] = out.cols;

	PyArrayObject* imgMask = (PyArrayObject *) PyArray_SimpleNewFromData( 2, size_pts, NPY_UINT8, out.data );


	return imgMask;
}

PyArrayObject* get_segm_features(int segmId)
{
	cmp::LetterCandidate& det = instances[0].segmenter->getLetterCandidates()[segmId];


	std::vector<float> segmFeatures;

	if(det.featureVector.empty())
	{
		extractFeatureVect(det.mask, segmFeatures, det);
	}else{
		for( int i = 0; i < det.featureVector.cols; i++ )
			segmFeatures.push_back(det.featureVector.at<float>(0, i));
	}
	npy_intp size_pts[1];
	size_pts[0] = segmFeatures.size();

	PyArrayObject* out = (PyArrayObject *) PyArray_SimpleNew( 1, size_pts, NPY_FLOAT );
	for( size_t i = 0; i < segmFeatures.size(); i++ )
	{
		float* ptr = (float *) PyArray_GETPTR1(out, i);
		*ptr = segmFeatures[i];
	}
	return out;
}


PyArrayObject* get_last_detection_lines(std::vector<cmp::FTextLine>& textLines, int instance = 0)
{
	npy_intp size_pts[2];
	size_pts[0] = textLines.size();
	size_pts[1] = 13;


	PyArrayObject* out = (PyArrayObject *) PyArray_SimpleNew( 2, size_pts, NPY_FLOAT );

	std::vector<cmp::LetterCandidate>& letterCandidatesMap =  instances[instance].segmenter->getLetterCandidates();
	//Py_INCREF(out);

	int lineNo = 0;
	for(size_t i = 0; i < textLines.size(); i++)
	{
		cmp::FTextLine* line = &textLines[i];

		if(!line->isSegmentable)
		{
			continue;
		}
		cv::Rect& box = line->bbox;

		//std::cout << "Line " << lineNo << " Box: " << box << std::endl;

		cv::RotatedRect rr = line->getMinAreaRect(letterCandidatesMap);

		float* ptr = (float *) PyArray_GETPTR2(out, lineNo, 0);
		*ptr++ = box.x;
		*ptr++ = box.y;
		*ptr++ = box.width;
		*ptr++ = box.height;

		cv::Point2f rect_points[4]; rr.points( rect_points );
		*ptr++ = rect_points[0].x;
		*ptr++ = rect_points[0].y;
		*ptr++ = rect_points[1].x;
		*ptr++ = rect_points[1].y;
		*ptr++ = rect_points[2].x;
		*ptr++ = rect_points[2].y;
		*ptr++ = rect_points[3].x;
		*ptr++ = rect_points[3].y;

    *ptr++ = (int) line->type;
		lineNo++;
	}
	return out;
}

std::vector<cmp::FTextLine> textLines;

PyArrayObject* find_text_lines(const char * outputDir, const char * imageName, int instance)
{
	double t2 = (double) cv::getTickCount();
	textLines.clear();
	//instances[instance].textLineDetector->findTextLines(lastImage, instances[instance].orbDetector, keypoints, instances[instance].orbDetector->getScales(), textLines);

	FastTextLineDetector textLineDetector;

	textLineDetector.findTextLines(lastImage, instances[instance].segmenter->getLetterCandidates(), instances[instance].ftDetector->getScales(), textLines);

	detectionStat.textLineTime += ( (double) cv::getTickCount() - t2)/ (cv::getTickFrequency()) * 1000;
	t2 = (double) cv::getTickCount();

	//instances[instance].lineSegmenter->segmentLines(textLines);
	detectionStat.gcTime += ( (double) cv::getTickCount() - t2)/ (cv::getTickFrequency()) * 1000;

#ifndef NDEBUG
	std::cout << "Text Lines in: " << ( (double) cv::getTickCount() - t2)/ (cv::getTickFrequency()) * 1000 << std::endl;
#endif

	if(outputDir != NULL && strlen(outputDir) > 0)
	{
		cv::Mat lineImage = lastImage.clone();
		if( lineImage.type() == CV_8UC1 )
		{
			cv::cvtColor(lineImage, lineImage, cv::COLOR_GRAY2RGB);
		}
		cv::Mat neiImage = lastImage.clone();
		for(size_t i = 0; i < textLines.size(); i++)
		{
			//cv::rectangle(lineImage, *it, cv::Scalar(255, 0, 0), 1);
			cmp::FTextLine& line = textLines[i];
			cv::RotatedRect rr = line.minRect;
			cv::Point2f rect_points[4]; rr.points( rect_points );
			cv::line(lineImage, rect_points[0], rect_points[1], cv::Scalar(255, 0, 0), 2);
			cv::line(lineImage, rect_points[1], rect_points[2], cv::Scalar(255, 0, 0), 2);
			cv::line(lineImage, rect_points[2], rect_points[3], cv::Scalar(255, 0, 0), 2);
			cv::line(lineImage, rect_points[3], rect_points[0], cv::Scalar(255, 0, 0), 2);
			/*
				int lineLength = rr.size.width / 2;
				cv::line(lineImage, cv::Point(line.topLine[2],line.topLine[3]), cv::Point(line.topLine[2]+line.topLine[0]*lineLength,line.topLine[3]+line.topLine[1]*lineLength), cv::Scalar(0, 0, 255), 2);
				cv::line(lineImage, cv::Point(line.bottomLine[2],line.bottomLine[3]), cv::Point(line.bottomLine[2]+line.bottomLine[0]*lineLength,line.bottomLine[3]+line.bottomLine[1]*lineLength), cv::Scalar(255, 0, 255), 2);
			 */
		}
		ostringstream os;
		os << outputDir << "/" << imageName << "_detectedLines.jpg";
		imwrite(os.str(), lineImage);
	}
	return get_last_detection_lines(textLines, instance);
}

PyArrayObject* get_normalized_line(int lineNo, int instance)
{

	cv::Mat norm = textLines[lineNo].getNormalizedMask(lastImage, instances[instance].segmenter->getLetterCandidates(), 1.0).clone();
	textLines[lineNo].normImage = norm;
	if(norm.type() == CV_8UC3){
		npy_intp size_pts[3];
		size_pts[0] = norm.rows;
		size_pts[1] = norm.cols;
		size_pts[2] = 3;
		//norm.addref();
		PyArrayObject* imgMask = (PyArrayObject *) PyArray_SimpleNewFromData( 3, size_pts, NPY_UINT8, norm.data );
		return imgMask;
	}else{
		npy_intp size_pts[2];
		size_pts[0] = norm.rows;
		size_pts[1] = norm.cols;
		PyArrayObject* imgMask = (PyArrayObject *) PyArray_SimpleNewFromData( 2, size_pts, NPY_UINT8, norm.data );
		return imgMask;
	}
}

