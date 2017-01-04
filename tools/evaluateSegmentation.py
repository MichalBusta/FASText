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

#sys.path.append('/home/busta/git/SmsReader/Debug')

from ft import FASTex

import glob
import datetime
from collections import defaultdict
from icdarUtils import collect_histograms, computeWordOvelap
import pandas

from vis import draw_keypoints


MIN_SEGM_OVRLAP = 0.6
evalPunctuation = False
dumpLines = False
dumpNegative = False

def run_evaluation(inputDir, outputDir, process_color = 0, processTest = 0):
    
    if not os.path.exists(outputDir):
        os.mkdir(outputDir)

    edgeThreshold = 14
    fastex = FASTex(edgeThreshold = edgeThreshold)
    
    images = glob.glob('{0}/*.jpg'.format(inputDir))
    
    segmDir = '{0}/segmentations'.format(inputDir)
    
    precision = 0;
    precisionDen = 0
    recall = 0
    recall05 = 0
    recallNonMax = 0
    recallDen = 0
    wordRecall = 0
    wordRecallDen = 0
    segm2chars = 0 
    
    regionsCount = 0
    regionsCountNonMax = 0
    missing_segmNonMaxCount = 0
    
    letterKeypointHistogram = defaultdict(lambda : defaultdict(float))
    octaveLetterKeypointHistogram = defaultdict(lambda : defaultdict(float))
    missing_letters = {}
    letterHistogram = defaultdict(int)
    missing_segm = {}
    missing_segm2 = {}
    missing_segmNonMax = {}
    diffMaxOctavesMap = {}
    diffScoreOctavesMap = {}
    segmHistogram = []
    segmWordHistogram = []
    
    results = []  
    hist = None
    histFp = None
    histDist = None
    histDistFp = None
    histDistMax = None
    histDistMaxWhite = None
    histDistMaxFp = None
    hist2dDist =None
    hist2dDistFp = None
    hist2dDistScore = None
    hist2dDistScoreFp = None
    histDistMaxWhiteFp = None
    
    histSegm = np.zeros((256), dtype = np.float)
    histSegmCount = np.zeros((256), dtype = np.int)
    stat = np.asarray([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], dtype=np.float)
    times = []
    gtSegmCount = 0
    wordsOk = []
    wordsFp = []
    
    keypointsTotal = 0
    keypointsTotalInside = 0
    orbTime = 0
    
    lineNo = 0
    perfectWords = 0;
    perfectWordsNS = 0;
    
    hasSegm = False
    
    for image in sorted(images):
        print('Processing {0}'.format(image))
        
        img = cv2.imread(image, 0)
        imgc = cv2.imread(image)
        imgcO = cv2.imread(image)
        if process_color == 1:
            imgproc = imgc
        else:
            imgproc = img
        
        baseName = os.path.basename(image)
        
        
        baseName = baseName[:-4]
        workPoint = 0.3
        segmentations = fastex.getCharSegmentations(imgproc, outputDir, baseName)
        segmentations = segmentations[:, 0:10]
    
        segmentations = np.column_stack( [ segmentations , np.zeros( (segmentations.shape[0], 2), dtype = np.float ) ] )
        maskDuplicates = segmentations[:, 8] == -1
        segmentationsDuplicates = segmentations[maskDuplicates, :]
        maskNoNei = segmentationsDuplicates[:, 9] > workPoint
        segmentationsNoNei = segmentationsDuplicates[maskNoNei, :]
        if segmentations.shape[0] > 0:
            print( 'Dupl ratio: {0} - {1}/ {2} - {3}'.format(segmentationsDuplicates.shape[0] / float(segmentations.shape[0]), segmentationsDuplicates.shape[0], segmentations.shape[0], segmentationsNoNei.shape[0] ) )
        keypoints = fastex.getLastDetectionKeypoints()
        keypointsTotal += keypoints.shape[0]
        statc =  fastex.getDetectionStat()
    
        times.append([ statc[1], statc[2], statc[3], statc[4], statc[5], statc[6], statc[7], statc[8], statc[9], statc[10]])
        stat += statc
        values = img[ keypoints[:, 1].astype(int), keypoints[:, 0].astype(int) ]
        valuesMax = img[keypoints[:, 6].astype(int), keypoints[:, 5].astype(int)]
        diffValMax = np.abs(values - valuesMax)
        
        
        regionsCount += segmentations.shape[0]
        regionsCountNonMax += segmentationsNoNei.shape[0]
       
        segmentations[:, 2] += segmentations[:, 0]
        segmentations[:, 3] += segmentations[:, 1]
        
        keypointsOrb = fastex.getLastDetectionOrbKeypoints()
        orbTime += keypointsOrb[0][9]
            
            
        segmGt = '{0}/{1}_GT.txt'.format(segmDir, baseName)
        pden = 0
        rden = 0
        if os.path.exists(segmGt):
            hasSegm = True
            (gt_rects, groups) = utls.read_icdar2013_segm_gt(segmGt)
            segmImg = '{0}/{1}_GT.bmp'.format(segmDir, baseName)
            if not os.path.exists(segmImg):
                segmImg = '{0}/gt_{1}.png'.format(segmDir, baseName)
            segmImg = cv2.imread(segmImg)
            
            try:
                (hist, histFp, histDist, histDistMax, histDistMaxWhite, hist2dDist, hist2dDistScore, histDistFp, histDistMaxFp, histDistMaxWhiteFp, hist2dDistFp, hist2dDistScoreFp, keypointsInside) = collect_histograms(img, segmImg, keypoints, values, diffValMax, keypointsTotalInside, diffMaxOctavesMap, diffScoreOctavesMap, hist, histFp, histDist, histDistMax, histDistMaxWhite, hist2dDist, hist2dDistScore, histDistFp, histDistMaxFp, histDistMaxWhiteFp, hist2dDistFp, hist2dDistScoreFp)
            except:
                pass
                    
            rcurrent = 0
            rcurrent05 = 0
            rcurrentNonMax = 0
            for k in range(len(gt_rects)):
                gt_rect = gt_rects[k]
                best_match = 0
                best_match_line = 0
                if (gt_rect[4] == ',' or gt_rect[4] == '.' or gt_rect[4] == '\'' or gt_rect[4] == ':' or gt_rect[4] == '-') and not evalPunctuation:
                    continue
                
                gtSegmCount += 1
                
                rectMask = np.bitwise_and(np.bitwise_and( keypointsInside[:, 0] >= gt_rect[0], keypointsInside[:, 0] <= gt_rect[2]), np.bitwise_and(keypointsInside[:, 1] >= gt_rect[1], keypointsInside[:, 1] <= gt_rect[3]))
                letterInside =  keypointsInside[rectMask, :]
                
                #make keypoints histogram 
                if letterInside.shape[0] > 0:
                    octaves = np.unique( letterInside[:, 2])
                    maxOctave = np.max(octaves)
                    maxOctavePoints = 0
                    
                    for i in range(int(maxOctave) + 1):
                        octavePoints = letterInside[letterInside[:, 2] == i, :]
                        maxOctavePoints = max(maxOctavePoints, octavePoints.shape[0])
                    if maxOctavePoints > 0:
                        octaveLetterKeypointHistogram[gt_rect[4]][0] += 1
                    if maxOctavePoints > 1:
                        octaveLetterKeypointHistogram[gt_rect[4]][1] += 1
                    if maxOctavePoints > 2:
                        octaveLetterKeypointHistogram[gt_rect[4]][2] += 1
                    if maxOctavePoints > 3:
                        octaveLetterKeypointHistogram[gt_rect[4]][3] += 1
                    
                    
                
                if letterInside.shape[0] == 0:
                    if not missing_letters.has_key(gt_rect[4]):
                        missing_letters[gt_rect[4]] = []
                    missing_letters[gt_rect[4]].append( (image, gt_rect) )  
                if letterInside.shape[0] > 0:
                    letterKeypointHistogram[gt_rect[4]][0] += 1
                if letterInside.shape[0] > 1:
                    letterKeypointHistogram[gt_rect[4]][1] += 1
                if letterInside.shape[0] > 2:
                    letterKeypointHistogram[gt_rect[4]][2] += 1
                if letterInside.shape[0] > 3:
                    letterKeypointHistogram[gt_rect[4]][3] += 1
                     
                letterHistogram[gt_rect[4]] += 1
                
                best_match2 = 0 
                minSingleOverlap = MIN_SEGM_OVRLAP
                if gt_rect[4] == 'i' or gt_rect[4] == '!':
                    minSingleOverlap = 0.5
                 
                for detId in range(segmentations.shape[0]):
                    rectn = segmentations[detId, :]
                    rect_int =  utils.intersect( rectn, gt_rect )
                    int_area = utils.area(rect_int)
                    union_area = utils.area(utils.union(rectn, gt_rect))
                
                    ratio = int_area / float(union_area)
                    rectn[10] = max(ratio, rectn[10])
                    
                    if rectn[9] > workPoint:
                        gt_rect[6] =  max(ratio, gt_rect[6])
                    
                    if ratio > best_match:
                        best_match = ratio
                        best_segm = segmentations[detId, :]
                        
                    if ratio > best_match_line and rectn[7] == 1.0 :
                        best_match_line = ratio
                        
                    if best_match < minSingleOverlap: 
                        if k < len(gt_rects) - 1:
                            gt_rect2 = gt_rects[k + 1]
                            chars2Rect = utils.union(gt_rect2, gt_rect)
                            rect_int = utils.intersect( rectn, chars2Rect )
                            int_area = utils.area(rect_int)
                            union_area = utils.area(utils.union(rectn, chars2Rect))
                            ratio = int_area / float(union_area)
                            rectn[10] = max(ratio, rectn[10]) 
                            if ratio > best_match2:
                                if ratio > MIN_SEGM_OVRLAP:
                                    segm2chars += 1
                                    best_match2 = ratio
                                    gt_rect[5] = ratio
                                    gt_rect2[5] = ratio
                       
                thickness = 1
                color = (255, 0, 255)
                if best_match >= minSingleOverlap:
                    color = (0, 255, 0)
                if best_match > 0.7:
                    thickness = 2
                cv2.rectangle(imgc, (gt_rect[0], gt_rect[1]), (gt_rect[2], gt_rect[3]), color, thickness)
                        
                recall += best_match
                recallNonMax += gt_rect[6]
                if best_match >= minSingleOverlap:
                    recall05 += best_match
                    rcurrent05 += best_match
                else:
                    if not missing_segm.has_key(image):
                        missing_segm[image] = []
                    missing_segm[image].append(gt_rect)
                    
                    if gt_rect[5] < MIN_SEGM_OVRLAP:
                        if not missing_segm2.has_key(image):
                            missing_segm2[image] = []
                        missing_segm2[image].append(gt_rect)
                        segm2chars += 1
                
                if gt_rect[6] < minSingleOverlap:
                    if not missing_segmNonMax.has_key(image):
                        missing_segmNonMax[image] = []
                    missing_segmNonMax[image].append(gt_rect)
                    missing_segmNonMaxCount += 1
                        
                    
                rcurrent += best_match
                rcurrentNonMax += gt_rect[6]
                recallDen +=  1   
                rden += 1
                
                if best_match > 0 and process_color != 1:
                    val = img[best_segm[5], best_segm[4]]
                    histSegm[val] += best_match
                    histSegmCount[val] += 1
                
            pcurrent = 0
            for detId in range(segmentations.shape[0]):
                best_match = 0
                rectn = segmentations[detId, :]
                
                for gt_rect in gt_rects:
                    rect_int =  utils.intersect( rectn, gt_rect )
                    int_area = utils.area(rect_int)
                    union_area = utils.area(utils.union(rectn, gt_rect))
                    
                    ratio = int_area / float(union_area)
                    
                    if ratio > best_match:
                        best_match = ratio
                
                precision += best_match
                pcurrent += best_match
                precisionDen +=  1   
                pden += 1
                
        
        if pden == 0:
            pcurrent = 0
        else:
            pcurrent = pcurrent / pden
            
        if rden == 0:
            rcurrent = 0
            rcurrent05 = 0
            rcurrentNonMax = 0
        else:
            rcurrent = rcurrent / rden
            rcurrent05 = rcurrent05 / rden
            rcurrentNonMax = rcurrentNonMax / rden
        
        
        segmHistogram.append([ segmentations.shape[0], segmentations[segmentations[:, 10] > 0.4].shape[0], segmentations[segmentations[:, 10] > 0.5].shape[0], segmentations[segmentations[:, 10] > 0.6].shape[0], segmentations[segmentations[:, 10] > 0.7].shape[0] ])
        
        segmWordHistogram.append([segmentations.shape[0], segmentations[np.bitwise_or(segmentations[:, 10] > 0.5, segmentations[:, 11] > 0.5 )].shape[0]])
        
        results.append((baseName, rcurrent, pcurrent, rcurrent05))

    
    if precisionDen == 0:
        pcurrent = 0
    else:
        precision = precision / precisionDen
        
    if recallDen == 0:
        rcurrent = 0
    else:
        recall = recall / recallDen
        recall05 = recall05 / recallDen
        recallNonMax = recallNonMax / recallDen
        
    wordRecall = wordRecall / max(1, wordRecallDen)
            
    try:
        histSegm = histSegm / max(1, histSegmCount)
    except ValueError:
        pass
    
    print('Evalation Results:')
    print( 'recall: {0}, precision: {1}, recall 0.5: {2}, recall NonMax: {3}'.format(recall, precision, recall05, recallNonMax) )
    
    kpTimes = np.histogram(np.asarray(times)[:, 0], bins=20)
    print('Keypoint Time Histogram: {0}'.format(kpTimes))
    
    
    print('Detection statistics:')    
    print(stat)
    
    for letter in letterKeypointHistogram.keys():
        for num in letterKeypointHistogram[letter].keys():
            letterKeypointHistogram[letter][num] = letterKeypointHistogram[letter][num] / float(letterHistogram[letter])
        for num in octaveLetterKeypointHistogram[letter].keys():
            octaveLetterKeypointHistogram[letter][num] = octaveLetterKeypointHistogram[letter][num] / float(letterHistogram[letter])
        letterKeypointHistogram[letter] = dict(letterKeypointHistogram[letter])
        octaveLetterKeypointHistogram[letter] = dict(octaveLetterKeypointHistogram[letter])
    
    print('Perfect words: {0}'.format(perfectWords))
        
    eval_date = datetime.date.today()
    np.savez('{0}/evaluation'.format(outputDir), recall=recall, recall05 = recall05, recallNonMax=recallNonMax, precision=precision, eval_date=eval_date, regionsCount=regionsCount, inputDir = inputDir, hist = hist, histSegm = histSegm, stat=stat, letterKeypointHistogram = dict(letterKeypointHistogram), missing_letters=missing_letters, octaveLetterKeypointHistogram=dict(octaveLetterKeypointHistogram), missing_segm=missing_segm, 
             times=np.asarray(times), histFp = histFp, gtSegmCount = gtSegmCount, wordRecall=wordRecall, histDist=histDist, histDistFp = histDistFp, histDistMax=histDistMax, histDistMaxFp=histDistMaxFp, hist2dDist=hist2dDist, hist2dDistFp=hist2dDistFp, hist2dDistScore=hist2dDistScore, hist2dDistScoreFp=hist2dDistScoreFp, histDistMaxWhite=histDistMaxWhite, histDistMaxWhiteFp=histDistMaxWhiteFp, wordsOk=wordsOk, wordsFp=wordsFp, diffMaxOctavesMap = diffMaxOctavesMap, diffScoreOctavesMap = diffScoreOctavesMap, 
             missing_segm2=missing_segm2, segmHistogram=segmHistogram, segmWordHistogram=segmWordHistogram, regionsCountNonMax=regionsCountNonMax, missing_segmNonMax=missing_segmNonMax)
    
    print( "GT segmentations count {0}".format(gtSegmCount) )
    print('FasTex Inside {0}/{1} ({2})'.format(keypointsTotalInside, keypointsTotal, keypointsTotalInside / float(keypointsTotal) ))
    print('FasText time: {0}, Orb time: {1} '.format( np.sum(times, 0)[0], orbTime))
    print('2 Chars Segmentation: {0}'.format(segm2chars) )
    print('NonMax Regions Count: {0}/{1}'.format(regionsCountNonMax, missing_segmNonMaxCount))
    


def plot_time_evaluation(input_dir='/datagrid/personal/TextSpotter/FastTextEval/ICDAR-Train'):
    '''
    Plots the time evaluation 
    '''
    d=input_dir
    subdirs = [os.path.join(d,o) for o in os.listdir(d) if os.path.isdir(os.path.join(d,o))]
    subdirs = np.sort(subdirs)
    
    
    dates = []
    rt = []
    pt = []
    ht = []
    regions_count = []
    regions_countNonMax = []
    keypoints_count = []
    miss_letters = []
    miss_segm_count = []
    miss_segm_count2 = []
    miss_segmnm_count = []
    miss_segml_count = []
    timesAll = [] 
    dirNames = []
    dirNamesTime = []
    gtSegmCount = 0
    for dir in subdirs:
        inputFile = '{0}/evaluation.npz'.format(dir)
        if not os.path.exists(inputFile):
            continue
        vars_dict = np.load(inputFile)
        rt.append( vars_dict['recall'] )
        pt.append(vars_dict['precision'])
        ht.append((vars_dict['recall'] + vars_dict['precision']) / 2)
        dates.append(vars_dict['eval_date'])
        if 'stat' in vars_dict.keys():
            stat = vars_dict['stat']
        else:
            stat = [0, 0, 0, 0, 0]
        keypoints_count.append(stat[0])
        regions_count.append(vars_dict['regionsCount'])
        if 'gtSegmCount' in vars_dict.keys():
            gtSegmCount = vars_dict['gtSegmCount']
        
        rcnm = 0
        if 'regionsCountNonMax' in vars_dict.keys():
            rcnm = vars_dict['regionsCountNonMax']
        regions_countNonMax.append(rcnm)
            
        ms = 0
        if 'missing_segm' in vars_dict.keys():
            missing_segm = vars_dict['missing_segm']
            missing_segm = dict(missing_segm.tolist())
            for img in missing_segm.keys():
                ms += len(missing_segm[img])
        miss_segm_count.append(ms)
        
        msnm = 0
        if 'missing_segmNonMax' in vars_dict.keys():
            missing_segm = vars_dict['missing_segmNonMax']
            missing_segm = dict(missing_segm.tolist())
            for img in missing_segm.keys():
                msnm += len(missing_segm[img])
        miss_segmnm_count.append(msnm)
        
        
        
        ms2 = 0
        if 'missing_segm2' in vars_dict.keys():
            missing_segm = vars_dict['missing_segm2']
            missing_segm = dict(missing_segm.tolist())
            for img in missing_segm.keys():
                ms2 += len(missing_segm[img])
        miss_segm_count2.append(ms2)
            
        ml = 0
        if 'missing_letters' in vars_dict.keys():
            missing_segm = vars_dict['missing_letters']
            missing_segm = dict(missing_segm.tolist())
            for img in missing_segm.keys():
                ml += len(missing_segm[img])
        miss_letters.append(ml)
        
        if 'times' in vars_dict.keys():
            times = vars_dict['times']
            times = np.average(times, 0)
            if times.shape[0] == 8:     
                times = np.hstack((times, [0, 0]))
            if times.shape[0] == 9:     
                times = np.hstack((times, [0]))
            timesAll.append(times)  
            dirNamesTime.append(os.path.basename(dir))
        dirNames.append(os.path.basename(dir))
    
    N = len(dates)    
    ind = np.arange(N)
    #fig = plt.figure(figsize=(16, 8))
    _, ax = plt.subplots(figsize=(16, 8))
    ax.set_ylim([0.1, 0.9])
    ax.plot(ind, rt, 'o-', label='Recall')
    ax.plot(ind, pt, 'o-', label='Precision')
    ax.plot(ind, ht, 'o-', label='h-Mean')
    plt.xticks(ind, dirNames, rotation=45)
    plt.legend(loc='upper left', shadow=True)
    ax.set_title('Time Evaluation')
    
    ax2 = ax.twinx()
    ax2.plot(ind, regions_count, 'mo-', label='Reg. Count')
    ax2.plot(ind, regions_countNonMax, 'o-', label='Reg. Count Non-Max')
    
    _, ax = plt.subplots(figsize=(16, 8))
    times = np.asarray(timesAll)
    
    indTimes = np.arange(len(times))
    ax.plot(indTimes, times[:, 0], 'o-', label='Keypoints')
    ax.plot(indTimes, times[:, 1], 'o-', label='Segmentation')
    ax.plot(indTimes, times[:, 2], 'o-', label='Regions Classification')
    ax.plot(indTimes, times[:, 0] + times[:, 1] , 'o-', label='Total')
    ax.plot(indTimes, times[:, 5] * 1000, 'o-', label='Wall Time')
    plt.legend(loc='upper left', shadow=True)
    ax.set_title('Processing Time')
    plt.xticks(indTimes, dirNamesTime, rotation=45)
    
    diffMaxOctavesMap = vars_dict['diffMaxOctavesMap']
    dictMax = diffMaxOctavesMap.all()
    diffMax = []
    for k in dictMax.keys():
        diffMax.append(dictMax[k])
    
    diffScoreOctavesMap = vars_dict['diffScoreOctavesMap']    
    dictScore = diffScoreOctavesMap.all()
    diffScore = []
    for k in dictScore.keys():
        diffScore.append(dictScore[k])
    
    data = np.transpose(
        np.vstack(
            (np.vstack(
                (np.vstack(
                    (np.vstack((np.vstack((rt, pt)), ht)), keypoints_count)), miss_letters)), 
                                   np.vstack((miss_segm_count, 
                                             np.vstack((regions_count, 1 -  np.asarray(miss_segm_count, dtype=np.float ) / gtSegmCount)))
                )
            )
        )
    )
    print("Missing Segm2: {0} ({1})".format( miss_segm_count2[len(miss_segm_count2) - 1], gtSegmCount) )
    print("Missing Segm - NonMax: {0}/{1}".format( miss_segmnm_count[len(miss_segmnm_count) - 1], regions_countNonMax[len(miss_segmnm_count) - 1]) )
    data = np.round(data, 3)
    df = pandas.DataFrame(data = data, columns=['r', 'p', 'HMean', 'KP', 'M.Let.', 'M.Seg.', 'SegC', 'RecR' ])
    print(df)
    times = np.round(times, 3)
    df = pandas.DataFrame(data = times, columns=['KP', 's', 'c', 'rawCls', 'tl', 'Wt', 'tc', 'tt', 'StrSeg', 'GCTime'])
    
    print("Times:")
    print(df)
    plt.show()

def show_last_failures(input_dir='/datagrid/personal/TextSpotter/FastTextEval/ICDAR-Train'): 
    '''
    Show last fail images
    '''
    d=input_dir
    subdirs = [os.path.join(d,o) for o in os.listdir(d) if os.path.isdir(os.path.join(d,o))]
    subdirs = np.sort(subdirs)
    lastDir = ''
    for dir_names in subdirs:
        file_name = '{0}/evaluation.npz'.format(dir_names)
        if not os.path.exists(file_name):
            continue
        vars_dict = np.load(file_name)
        results =  vars_dict['results'] 
        lastDir = dir_names
     
    print('Last Results:')
    print( 'recall: {0}, precision: {1}'.format(vars_dict['recall'], vars_dict['precision']) )
    
    for i in range(10):
        result = results[i]
        img = cv2.imread('{0}/{1}_chars.png'.format(lastDir, result[0]))
        fig = plt.figure(figsize=(12, 12))
        a=fig.add_subplot(1,1,1)
        plt.imshow(img)
        a.set_title('{0}.jpg - Recall: {1}, Precision: {2}'.format(result[0], result[1], result[2]))
        plt.show()

if __name__ == '__main__':
    
    process_color = 0
    processTest = 1
    
    inputDir = '/textspotter/evaluation-sets/icdar2013-Train'
    #inputDir = '/home/busta/data/cvut/textspotter/datasets/icdar2013-Train'
    outputBase = '/datagrid/personal/TextSpotter/FastTextEval/ICDAR-Train'
    if processTest == 1: 
        inputDir = '/home/busta/data/icdar2013-Test'
        outputBase = '/tmp/test'
    outputBase = '/textspotter/experiments/FasTex/Train/'
    #outputBase = '/tmp/eval/'
    if processTest == 1: 
        outputBase = '/home/busta/tmp/evalTest'
        
    if processTest == 2: 
        inputDir = '/textspotter/evaluation-sets/icdar-2015-Ch4/Train'
        outputBase = '/tmp/evalCh4'
    #if process_color == 1:
    #    outputBase = '/datagrid/personal/TextSpotter/FastTextEval/ICDAR-Train-Color'
    outputDir = '{0}/{1}'.format(outputBase, datetime.date.today().strftime('%Y-%m-%d'))
    run_evaluation(inputDir, outputDir, process_color, processTest)
    
    plot_time_evaluation(outputBase)
    #show_last_failures(outputBase)
    
    #draw_missed_letters()
    #show_last_failures('/datagrid/personal/TextSpotter/FastTextEval/ICDAR-Train')
    #show_image()
    
    
