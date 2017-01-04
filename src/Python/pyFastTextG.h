/*
 * pyFastText.h
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
#ifndef PYFASTTEXT_H_
#define PYFASTTEXT_H_

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>

#define PY_ARRAY_UNIQUE_SYMBOL cool_ARRAY_API
#include <numpy/arrayobject.h>

#ifdef __cplusplus
extern "C" {
#endif


PyArrayObject* find_keypoints(PyArrayObject* img, int numOfDims, npy_intp* img_dims, int scaleFactor, int nlevels, int edgeThreshold, int keypointTypes, int kMin, int kMax);


int initialize(float scaleFactor, int nlevels, int edgeThreshold, int keypointTypes, int kMin, int kMax,
		const char* charClsFile, int erode, int segmentGrad, int minComponentSize, int instance, float thresholdFactor, int segmDeltaInt);

PyArrayObject* get_char_segmentations(PyArrayObject* img, int numOfDims, npy_intp* img_dims, const char * outputDir, const char * imageName, int instance, int minHeight);

PyArrayObject* find_text_lines(const char * outputDir, const char * imageName, int instance);

PyArrayObject* get_normalized_line(int lineNo, int instance);

PyArrayObject* get_keypoint_strokes(int keypointId, int instance);

PyArrayObject* get_last_detection_keypoints();

PyArrayObject* get_last_detection_orb_keypoints();

PyArrayObject* get_detection_stat();

PyArrayObject* get_image_at_scale(int level, int instance);

PyArrayObject* get_segmentation_mask(int maskId);

PyArrayObject* get_image_scales(int instance);

#ifdef __cplusplus
}
#endif

#endif /* PYFASTTEXT_H_ */
