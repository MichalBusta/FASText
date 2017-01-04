'''
Created on Sep 15, 2015

@author: busta
'''

import os, sys

baseDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print baseDir
#sys.path.append(os.path.join(baseDir, "Release/bin" ))
sys.path.append(os.path.join(baseDir, "Release" ))

sys.path.append("/Users/flipajs/Downloads/temp2/FASText/Release")
#sys.path.append(os.path.join(baseDir, "Debug" ))

import numpy as np

import ftext
import math

defCharClsModel = '{0}/cvBoostChar.xml'.format(baseDir)

class FASTex(object):
    '''
    classdocs
    '''


    def __init__(self, charClsModelFile = defCharClsModel, scaleFactor = 1.6, nlevels = -1, edgeThreshold = 15, keypointTypes = 3, kMin = 9, kMax = 11, erode = 0, segmentGrad = 0, 
                 minCompSize = 0, segmDeltaInt = 1):
        '''
        Constructor 
        '''
        ftext.init(scaleFactor, nlevels, edgeThreshold, keypointTypes, kMin, kMax, charClsModelFile, erode, segmentGrad, minCompSize, 0, 1.0, segmDeltaInt)
        
        self.edgeThreshold = edgeThreshold
    
    def findKeypoints(self, img, outputDir, baseName):
        return ftext.findKeyPoints(img, outputDir, baseName)
        
     
    def getCharSegmentations(self, img, outputDir='', baseName=''):
        '''
        @param img - the source image (numpy arry)
        @param outputDir - the debug directory for visualizations 
        @param baseName
        
        returns the np array where row is:  [bbox.x, bbox.y, bbox.width, bbox.height, keyPoint.pt.x, keyPoint.pt.y, octave, ?, duplicate, quality, [keypointsIds]]
        '''
        return ftext.getCharSegmentations(img, outputDir, baseName)
    
    def findTextLines(self, outputDir='', baseName=''):
        '''
        @param outputDir - the debug directory for visualizations 
        @param baseName
        
        returns the np array where row is:  [bbox.x, bbox.y, bbox.width, bbox.height, rotated rectangle points (pt1.x, pt1.y, ... pt3.y) ]
        '''
        return ftext.findTextLines(outputDir, baseName)
    
    def getNormalizedLine(self, lineNo):
        '''
        @param lineNo - the id of line - row in np array from findTextLines 
        
        returns the line image normalized against the rotation
        '''
        return ftext.getNormalizedLine(lineNo)
    
    def getLastDetectionKeypoints(self):
        return ftext.getLastDetectionKeypoints()
    
    def getImageAtScale(self, scale):
        return ftext.getImageAtScale(scale)

    def getImageScales(self):
        return ftext.getImageScales()
    
    def getDetectionStat(self):
        return ftext.getDetectionStat()
    
    def getLastDetectionOrbKeypoints(self):
        return ftext.getLastDetectionOrbKeypoints()    
        
    def getSegmentationMask(self, maskNo):
        return ftext.getSegmentationMask(maskNo)
    
    def saveKeypints(self, keypoints, outFile):
        
        keypointSegments = {}
        for i in range(keypoints.shape[0]):
            strokes = ftext.getKeypointStrokes(i)
            keypointSegments[i] = strokes
        
        np.savez(outFile, keypoints=keypoints, keypointSegments = keypointSegments)    
    
