'''
Created on Dec 3, 2014

@author: busta
'''
import matplotlib.pyplot as plt

import sys, os
import numpy as np
import cv2
import utls
import utils


baseDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print baseDir
#sys.path.append(os.path.join(baseDir, "Release/bin" ))
sys.path.append(os.path.join(baseDir, "Release" ))


import ftext
import glob
import pylab
import datetime
from collections import defaultdict
from icdarUtils import computeWordOvelap

from ft import FASTex


MIN_SEGM_OVRLAP = 0.6
evalPunctuation = False

def init_ftext(minCompSize = 5):
    
    scaleFactor = 1.6
    nleves = -1
    edgeThreshold = 13
    keypointTypes = 3

    #charClsModelFile = '/tmp/cvBoostChar.xml'

    edgeThreshold = 14
    fastex = FASTex(edgeThreshold = edgeThreshold)

def run_evaluation(inputDir, outputDir, invert = False, isFp = False):
    
    if not os.path.exists(outputDir):
        os.mkdir(outputDir)
    
    images = glob.glob('{0}/*.jpg'.format(inputDir))
    images.extend(glob.glob('{0}/*.JPG'.format(inputDir)))
    images.extend(glob.glob('{0}/*.png'.format(inputDir)))
    segmDir = '{0}/segmentations'.format(inputDir)

    for image in images:
        print('Processing {0}'.format(image))
        
        img = cv2.imread(image, 0)
        imgc = cv2.imread(image)
        imgproc = img
        
        imgKp = np.copy(img)
        imgKp.fill(0)
        
        baseName = os.path.basename(image)
        baseName = baseName[:-4]
        workPoint = 0.3
        segmentations = ftext.getCharSegmentations(imgproc) #, outputDir, baseName)
        segmentations = segmentations[:, 0:10]
        segmentations = np.column_stack( [ segmentations , np.zeros( (segmentations.shape[0], 2), dtype = np.float ) ] )
        maskDuplicates = segmentations[:, 8] == -1
        segmentationsDuplicates = segmentations[maskDuplicates, :]
        maskNoNei = segmentationsDuplicates[:, 9] > workPoint
        segmentationsNoNei = segmentationsDuplicates[maskNoNei, :]
        keypoints = ftext.getLastDetectionKeypoints()
        imgKp[keypoints[:, 1].astype(int), keypoints[:, 0].astype(int)] = 255
        scales = ftext.getImageScales()
        statc =  ftext.getDetectionStat()
        words = ftext.findTextLines()
        segmLine = segmentations[segmentations[:, 7] == 1.0, :]
        segmentations[:, 2] += segmentations[:, 0] 
        segmentations[:, 3] += segmentations[:, 1]
        
        
        if isFp:
            for detId in range(0, segmentations.shape[0]):
                ftext.acummulateCharFeatures(0, detId)
                
            continue 
            
        lineGt = '{0}/gt_{1}.txt'.format(inputDir, baseName)
        if not os.path.exists(lineGt):
            lineGt = '{0}/{1}.txt'.format(inputDir, baseName)

        lineGt = '{0}/gt_{1}.txt'.format(inputDir, baseName)
        if os.path.exists(lineGt):
            try:
                word_gt = utls.read_icdar2013_txt_gt(lineGt)
            except ValueError:
                try:
                    word_gt = utls.read_icdar2013_txt_gt(lineGt, separator = ',')
                except ValueError:
                    word_gt = utls.read_icdar2015_txt_gt(lineGt, separator = ',')
        else:
            lineGt = '{0}/{1}.txt'.format(inputDir, baseName)
            word_gt = utls.read_mrrc_txt_gt(lineGt, separator = ',')
            
            
            
            
        rWcurrent = 0.0
        for gt_box in word_gt:
            if len(gt_box[4]) == 1:
                continue
            best_match = 0
            cv2.rectangle(imgc, (gt_box[0], gt_box[1]), (gt_box[2], gt_box[3]), (0, 255, 0))
            for det_word in words:
                rect_int =  utils.intersect( det_word, gt_box )
                int_area = utils.area(rect_int)
                union_area = utils.area(utils.union(det_word, gt_box))
                
                if union_area == 0:
                    continue
                
                ratio = int_area / float(union_area)
                det_word[11] = max(det_word[11], ratio)
                
                if ratio > best_match:
                    best_match = ratio
            rWcurrent += best_match
            
            best_match = 0
            for detId in range(segmentations.shape[0]):
                rectn = segmentations[detId, :]
                rect_int =  utils.intersect( rectn, gt_box )
                int_area = utils.area(rect_int)
                union_area = utils.area(utils.union(rectn, gt_box))

                ratio = int_area / float(union_area)
                rectn[11] = max(ratio, rectn[11])
                if ratio > best_match:
                    best_match = ratio
                if ratio > 0.7:

                    #print( "Word Match!" )
                    #tmp = ftext.getSegmentationMask(detId)
                    #cv2.imshow("ts", tmp)
                    #cv2.waitKey(0)

                    ftext.acummulateCharFeatures(2, detId)
            
            
        segmImg = '{0}/{1}_GT.bmp'.format(segmDir, baseName)
        if not os.path.exists(segmImg):
            segmImg = '{0}/gt_{1}.png'.format(segmDir, baseName)
        if not os.path.exists(segmImg):
            segmImg = '{0}/{1}.png'.format(segmDir, baseName)
        segmImg = cv2.imread(segmImg, 0)
        if invert and segmImg is not None:
            segmImg = ~segmImg

        gt_rects = []
        miss_rects = []
        segmGt = '{0}/{1}_GT.txt'.format(segmDir, baseName)
        if os.path.exists(segmGt) and False:
            (gt_rects, groups) = utls.read_icdar2013_segm_gt(segmGt)
            segmImg = '{0}/{1}_GT.bmp'.format(segmDir, baseName)
            if not os.path.exists(segmImg):
                segmImg = '{0}/gt_{1}.png'.format(segmDir, baseName)
            segmImg = cv2.imread(segmImg)
        else:
            contours = cv2.findContours(np.copy(segmImg), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)[1]
            for cont in contours:
                rect = cv2.boundingRect( cont )
                rect = [rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3], '?', 0, 0]
                gt_rects.append(rect)

            
        
        for detId in range(segmentations.shape[0]):
            rectn = segmentations[detId, :]
                
            for k in range(len(gt_rects)):
                gt_rect = gt_rects[k]
                best_match = 0
                best_match_line = 0
                if (gt_rect[4] == ',' or gt_rect[4] == '.' or gt_rect[4] == '\'' or gt_rect[4] == ':' or gt_rect[4] == '-') and not evalPunctuation:
                    continue
        
                minSingleOverlap = MIN_SEGM_OVRLAP
                if gt_rect[4] == 'i' or gt_rect[4] == '!':
                    minSingleOverlap = 0.5
         
        
                rect_int =  utils.intersect( rectn, gt_rect )
                int_area = utils.area(rect_int)
                union_area = utils.area(utils.union(rectn, gt_rect))
                ratio = int_area / float(union_area)
                rectn[10] = max(ratio, rectn[10])
                
                if rectn[9] > workPoint:
                    gt_rect[6] =  max(ratio, gt_rect[6])
                
                if ratio > best_match:
                    best_match = ratio
                    
                if ratio > best_match_line and rectn[7] == 1.0 :
                    best_match_line = ratio
                if ratio > minSingleOverlap:
                    ftext.acummulateCharFeatures(1, detId)


                if ratio < minSingleOverlap:
                    if k < len(gt_rects) - 1:
                        gt_rect2 = gt_rects[k + 1]
                        chars2Rect = utils.union(gt_rect2, gt_rect)
                        rect_int = utils.intersect( rectn, chars2Rect )
                        int_area = utils.area(rect_int)
                        union_area = utils.area(utils.union(rectn, chars2Rect))
                        ratio = int_area / float(union_area)
                        rectn[10] = max(ratio, rectn[10])

                        if ratio > 0.8:
                            best_match2 = ratio
                            gt_rect[5] = ratio
                            gt_rect2[5] = ratio
                            ftext.acummulateCharFeatures(2, detId)

                   
                thickness = 1
                color = (255, 0, 255)
                if best_match >= minSingleOverlap:
                    color = (0, 255, 0)
                if best_match > 0.7:
                    thickness = 2
                cv2.rectangle(imgc, (gt_rect[0], gt_rect[1]), (gt_rect[2], gt_rect[3]), color, thickness)
            
            if rectn[10] == 0 and rectn[11] == 0:
                ftext.acummulateCharFeatures(0, detId)
                            
        
        '''
        if len(miss_rects) > 0:
            cv2.imshow("ts", imgc)
            cv2.imshow("kp", imgKp)
            cv2.waitKey(0)
        '''
                
def run_words(inputDir, outputDir, invert = False):
    
    if not os.path.exists(outputDir):
        os.mkdir(outputDir)
    
    #images = glob.glob('{0}/*.png'.format('/datagrid/personal/TextSpotter/evaluation-sets/MS-text_database'))
    #images = glob.glob('{0}/*.jpg'.format('/datagrid/personal/TextSpotter/evaluation-sets/neocr_dataset'))
    images = glob.glob('{0}/*.jpg'.format(inputDir))
    images.extend(glob.glob('{0}/*.JPG'.format(inputDir)))
    images.extend(glob.glob('{0}/*.png'.format(inputDir)))
    
    matched_words = 0
    word_count = 0

    for image in sorted(images):
        print('Processing {0}'.format(image))
        
        img = cv2.imread(image, 0)
        imgc = cv2.imread(image)
        imgproc = img
        
        imgKp = np.copy(img)
        imgKp.fill(0)
        
        baseName = os.path.basename(image)
        baseName = baseName[:-4]
        workPoint = 0.3
        segmentations = ftext.getCharSegmentations(imgproc) #, outputDir, baseName)
        segmentations = segmentations[:, 0:10]
        segmentations = np.column_stack( [ segmentations , np.zeros( (segmentations.shape[0], 2), dtype = np.float ) ] )
        maskDuplicates = segmentations[:, 8] == -1
        segmentationsDuplicates = segmentations[maskDuplicates, :]
        maskNoNei = segmentationsDuplicates[:, 9] > workPoint
        keypoints = ftext.getLastDetectionKeypoints()
        imgKp[keypoints[:, 1].astype(int), keypoints[:, 0].astype(int)] = 255
        scales = ftext.getImageScales()
        statc =  ftext.getDetectionStat()
        words = ftext.findTextLines()
        segmentations[:, 2] += segmentations[:, 0] 
        segmentations[:, 3] += segmentations[:, 1]
        
        
        lineGt = '{0}/gt_{1}.txt'.format(inputDir, baseName)
        if not os.path.exists(lineGt):
            lineGt = '{0}/{1}.txt'.format(inputDir, baseName)

        lineGt = '{0}/gt_{1}.txt'.format(inputDir, baseName)
        if os.path.exists(lineGt):
            try:
                word_gt = utls.read_icdar2013_txt_gt(lineGt)
            except ValueError:
                try:
                    word_gt = utls.read_icdar2013_txt_gt(lineGt, separator = ',')
                except ValueError:
                    word_gt = utls.read_icdar2015_txt_gt(lineGt, separator = ',')
        else:
            lineGt = '{0}/{1}.txt'.format(inputDir, baseName)
            word_gt = utls.read_mrrc_txt_gt(lineGt, separator = ',')
            
        cw = 0
        for detId in range(segmentations.shape[0]):
            best_match = 0
            
            for gt_box in word_gt:
                if len(gt_box[4]) == 1:
                    continue
                if gt_box[4][0] == "#":
                    continue
                cw += 1
                
                rectn = segmentations[detId, :]
                rect_int =  utils.intersect( rectn, gt_box )
                int_area = utils.area(rect_int)
                union_area = utils.area(utils.union(rectn, gt_box))

                ratio = int_area / float(union_area)
                rectn[11] = max(ratio, rectn[11])
                if ratio > best_match:
                    best_match = ratio
                if ratio > 0.7:

                    #print( "Word Match!" )
                    #cv2.rectangle(imgc, (rectn[0], rectn[1]), (rectn[2], rectn[3]), (0, 255, 0))
                    #cv2.imshow("ts", imgc)
                    #cv2.waitKey(0)
                    ftext.acummulateCharFeatures(2, detId)
                    if gt_box[5] != -1:
                        matched_words += 1
                    gt_box[5] = -1
            
            if best_match == 0:
                ftext.acummulateCharFeatures(0, detId)
            
        word_count += cw    
        print("word recall: {0}".format(matched_words / float(word_count)))

if __name__ == '__main__':
    
    init_ftext()
    

    inputDir = '/home/busta/data/icdar2013-Train'
    outputBase = '/mnt/textspotter/FastTextEval/BDT'
    outputDir = '{0}/{1}'.format(outputBase, datetime.date.today().strftime('%Y-%m-%d'))
    run_evaluation(inputDir, outputDir, True)
    
    #run_words('/home/busta/data/icdar2013-Test', '/tmp/ch4')
    run_words('/home/busta/data/icdar2015-Ch4-Train', '/tmp/ch4')
    ftext.trainCharFeatures()
    
    
    