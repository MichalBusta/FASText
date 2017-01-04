/*
 * pyFastTextAPI.c
 *
 *  Created on: Oct 17, 2014
 *      Author: Michal Busta
 *
 * Copyright 2015, Michal Busta, Lukas Neumann, Jiri Matas.
 *
 * Based on:
 *
 * FASText: Efficient Unconstrained Scene Text Detector,Busta M., Neumann L., Matas J.: ICCV 2015.
 * Machine learning for high-speed corner detection, E. Rosten and T. Drummond, ECCV 2006
 */

#include "pyFastTextG.h"

static PyObject *FastTextError;



static PyObject* findKeyPoints_cfunc (PyObject *dummy, PyObject *args)
{
	PyObject *arg1=NULL;
	PyArrayObject *out=NULL;
	PyArrayObject *arr1=NULL;
	PyArrayObject* img = NULL;
	npy_intp* img_dims = NULL;

	float scaleFactor = 2.0f;
	int nlevels = 3;
	int edgeThreshold = 12;
	int keypointTypes = 2;
	int kMin = 9;
	int kMax = 16;

	if (!PyArg_ParseTuple(args, "O|fiiiii", &arg1, &scaleFactor, &nlevels, &edgeThreshold, &keypointTypes, &kMin, &kMax))
		return NULL;

	img = (PyArrayObject *) arg1;
	img_dims = PyArray_DIMS(img);
	int numOfDim = PyArray_NDIM(img);

	out =  find_keypoints(img, numOfDim, img_dims, scaleFactor, nlevels, edgeThreshold, keypointTypes, kMin, kMax);

	return (PyObject *) out;
}

static PyObject* initialize_cfunc (PyObject *dummy, PyObject *args)
{
	float scaleFactor = 2.0f;
	int nlevels = 3;
	int edgeThreshold = 12;
	int keypointTypes = 2;
	int kMin = 9;
	int kMax = 11;
	int segmenterType = 0;
	const char * charClsFile;
	const char * outputDir;
	int erode = 1;
	int segmentGrad = 0;
	int minCompSize = 0;
	float thresholdFactor = 1.0;
	float minTupleTopBottomAngle = 0;
	int segmDeltaInt = 0;
	int instance = -1;
	float maxSpaceHeightRatio = -1;
	int createKeypointSegmenter = 0;
	if (!PyArg_ParseTuple(args, "|fiiiiisiiiifi", &scaleFactor, &nlevels, &edgeThreshold, &keypointTypes, &kMin, &kMax, &charClsFile,
			&erode, &segmentGrad, &minCompSize, &instance, &thresholdFactor, &segmDeltaInt))
		return NULL;

	instance = initialize(scaleFactor, nlevels, edgeThreshold, keypointTypes, kMin, kMax, charClsFile, erode, segmentGrad, minCompSize, instance, thresholdFactor, segmDeltaInt);

	return Py_BuildValue("i", instance);
}

static PyObject* get_char_segmentations_cfunc (PyObject *dummy, PyObject *args)
{
	PyObject *arg1=NULL;
	PyArrayObject *out=NULL;
	PyArrayObject *arr1=NULL;
	PyArrayObject* img = NULL;
	npy_intp* img_dims = NULL;

	const char * imageName;
	const char * outputDir = NULL;
	int instance = 0;
	int minHeight = 0;
	if (!PyArg_ParseTuple(args, "O|ssii", &arg1, &outputDir, &imageName, &instance, &minHeight ))
		return NULL;

	img = (PyArrayObject *) arg1;
	img_dims = PyArray_DIMS(img);
	int numOfDim = PyArray_NDIM(img);

	out =  get_char_segmentations(img, numOfDim, img_dims, outputDir, imageName, instance, minHeight);

	return (PyObject *) out;
}

static PyObject* getLastKeyPoints_cfunc (PyObject *dummy, PyObject *args)
{
	PyObject *arg1=NULL;
	PyArrayObject *out=NULL;

	out =  get_last_detection_keypoints();

	return (PyObject *) out;
}

static PyObject* getKeypointStrokes_cfunc (PyObject *dummy, PyObject *args)
{
	PyObject *arg1=NULL;
	PyArrayObject *out=NULL;
	int instance = 0;
	int keypointId = 0;
	if (!PyArg_ParseTuple(args, "i|i", &keypointId, &instance ))
		return NULL;

	out =  get_keypoint_strokes(keypointId, instance);

	return (PyObject *) out;
}

static PyObject* getDetectionStat_cfunc (PyObject *dummy, PyObject *args)
{
	PyObject *arg1=NULL;
	PyArrayObject *out=NULL;
	out =  get_detection_stat();
	return (PyObject *) out;
}

static PyObject* getImageAtScale_cfunc (PyObject *dummy, PyObject *args)
{
	PyObject *arg1=NULL;
	PyArrayObject *out=NULL;
	int imageScale = 0;
	int instance = 0;
	if (!PyArg_ParseTuple(args, "i|i", &imageScale, &instance))
		return NULL;
	out =  get_image_at_scale(imageScale, instance);
	return (PyObject *) out;
}

static PyObject* getSegmMask_cfunc (PyObject *dummy, PyObject *args)
{
	PyObject *arg1=NULL;
	PyArrayObject *out=NULL;
	int maskId = 0;
	if (!PyArg_ParseTuple(args, "i", &maskId))
		return NULL;
	out =  get_segmentation_mask(maskId);
	return (PyObject *) out;
}

static PyObject* getImageScales_cfunc (PyObject *dummy, PyObject *args)
{
	PyObject *arg1=NULL;
	PyArrayObject *out=NULL;
	int instance = 0;
	if (!PyArg_ParseTuple(args, "|i", &instance))
		return NULL;
	out =  get_image_scales(instance);
	return (PyObject *) out;
}

static PyObject* getLastOrbKeyPoints_cfunc (PyObject *dummy, PyObject *args)
{
	PyObject *arg1=NULL;
	PyArrayObject *out=NULL;

	out =  get_last_detection_orb_keypoints();

	return (PyObject *) out;
}

static PyObject* find_text_lines_cfunc (PyObject *dummy, PyObject *args)
{
	PyObject *arg1=NULL;
	PyArrayObject *out=NULL;
	PyArrayObject *arr1=NULL;

	const char * imageName;
	const char * outputDir = NULL;
	int instance = 0;
	int merge_inners = 0;
	if (!PyArg_ParseTuple(args, "|ssi", &outputDir, &imageName, &instance ))
		return NULL;

	out =  find_text_lines(outputDir, imageName, instance);
	return (PyObject *) out;
}

static PyObject* getNormalizedLine_cfunc (PyObject *dummy, PyObject *args)
{
	PyObject *arg1=NULL;
	PyArrayObject *out=NULL;
	int instance = 0;
	int line = 0;
	if (!PyArg_ParseTuple(args, "i|i", &line, &instance))
			return NULL;

	out =  get_normalized_line(line, instance);

	return (PyObject *) out;
}

static PyMethodDef FastTextMethods[] = {

		{"findKeyPoints",  findKeyPoints_cfunc, METH_VARARGS, "Find Keipoints in the image"},
		{"init",  initialize_cfunc, METH_VARARGS, "Initializes FastText detector"},
		{"getCharSegmentations",  get_char_segmentations_cfunc, METH_VARARGS, "Returns the character segmentations"},
		{"getLastDetectionKeypoints",  getLastKeyPoints_cfunc, METH_VARARGS, "Returns the character segmentations"},
		{"getKeypointStrokes",  getKeypointStrokes_cfunc, METH_VARARGS, "Returns the strokes of given keypoint"},
		{"getDetectionStat",  getDetectionStat_cfunc, METH_NOARGS, "Returns the detection statistics"},
		{"getImageAtScale",  getImageAtScale_cfunc, METH_VARARGS, "Returns the detection image in given scale"},
		{"getSegmentationMask",  getSegmMask_cfunc, METH_VARARGS, "Returns the segmentation mask by ID"},
		{"getImageScales",  getImageScales_cfunc, METH_VARARGS, "Returns the image pyramid scales"},
		{"getLastDetectionOrbKeypoints",  getLastOrbKeyPoints_cfunc, METH_VARARGS, "Find ORB keypoints in the image"},
		{"findTextLines",  find_text_lines_cfunc, METH_VARARGS, "Finds and returns text lines in the image"},
		{"getNormalizedLine",  getNormalizedLine_cfunc, METH_VARARGS, "Returns the normalized line segmentation"},
		{NULL, NULL, 0, NULL}        /* Sentinel */
};


PyMODINIT_FUNC
initftext(void)
{
	PyObject *m;

	m = Py_InitModule("ftext", FastTextMethods);
	import_array();
	if (m == NULL)
		return;

	FastTextError = PyErr_NewException((char*) "ftext.error", NULL, NULL);
	Py_INCREF(FastTextError);
	PyModule_AddObject(m, "error", FastTextError);
}

