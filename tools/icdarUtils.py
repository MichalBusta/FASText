'''
Created on Jan 7, 2015

@author: busta
'''

import os
import numpy as np
import cv2

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import utils

from ft import FASTex

import pylab
import pandas

def draw_missed_letters(input_dir='/datagrid/personal/TextSpotter/FastTextEval/ICDAR-Train', color = 0, edgeThreshold = 12, inter = True, scalingFactor=0.5):
    
    ft = FASTex(process_color = color, edgeThreshold = edgeThreshold)
    
    d=input_dir
    subdirs = [os.path.join(d,o) for o in os.listdir(d) if os.path.isdir(os.path.join(d,o))]
    subdirs = np.sort(subdirs)
    lastDir = ''
    for dir_name in subdirs:
        file_name = '{0}/evaluation.npz'.format(dir_name)
        if not os.path.exists(file_name):
            continue
        vars_dict = np.load(file_name) 
        inputDir = vars_dict['inputDir']
        lastDir = dir_name
        if 'letterKeypointHistogram' in vars.keys():
            letterKeypointHistogram = vars_dict['letterKeypointHistogram']
            letterKeypointHistogram = dict(letterKeypointHistogram.tolist())
        
    print(lastDir)
    
    missing_letters = vars['missing_letters']
    missing_letters = dict(missing_letters.tolist())
    
    segmDir = '{0}/segmentations'.format(inputDir)
    
    keys = []
    ticks = []
    values = []
    values.append([])
    values.append([])
    values.append([])
    values.append([])
    ticks.append([])
    ticks.append([])
    ticks.append([])
    ticks.append([])
    listlen = 0
    for letter in letterKeypointHistogram.keys():
        keys.append(letter)
        values[0].append(0)
        ticks[0].append(listlen)
        values[1].append(0)
        ticks[1].append(listlen + 0.2)
        values[2].append(0)
        ticks[2].append(listlen + 0.4)
        values[3].append(0)
        ticks[3].append(listlen + 0.6)
        for num in letterKeypointHistogram[letter].keys():
            values[num][listlen] = letterKeypointHistogram[letter][num]
            
        listlen += 1
        
    indices = sorted(range(len(values[0])),key=lambda x:values[0][x])
    indices.reverse()
    
    border = 25
    for letter in  np.asarray(keys)[np.asarray(indices)]:
        if not missing_letters.has_key(letter):
            continue
        arr =  missing_letters[letter]
        for i in range(len(arr)):
            miss = arr[i]
            gt0 = miss[1]
            gt = [gt0[0] - border, gt0[1] - border, gt0[2] + border, gt0[3] + border ]
            gt[0] = max(0, gt[0])
            gt[1] = max(0, gt[1])
            if color == 1:
                img = cv2.imread(miss[0])
            else:
                img = cv2.imread(miss[0], 0)
                
            gt[2] = min(img.shape[1], gt[2])
            gt[3] = min(img.shape[0], gt[3])
                
            baseName = os.path.basename(miss[0])
            baseName = baseName[:-4]
            segmImg = '{0}/{1}_GT.bmp'.format(segmDir, baseName)
            segmImg = '{0}/{1}_GT.bmp'.format(segmDir, baseName)
            if not os.path.exists(segmImg):
                segmImg = '{0}/gt_{1}.png'.format(segmDir, baseName)
            segmImg = cv2.imread(segmImg)
            
            segmentations = ft.getCharSegmentations(img)
            keypoints = ft.getLastDetectionKeypoints()
            scales = ft.getImageScales()
            
            centers = segmImg[keypoints[:, 1].astype(int), keypoints[:, 0].astype(int)]
            keypointsInsideMask = centers == (255, 255, 255)
            keypointsInsideMask = np.invert(np.bitwise_and(np.bitwise_and(keypointsInsideMask[:, 0], keypointsInsideMask[:, 1]), keypointsInsideMask[:, 2]))
            keypointsInside = keypoints[keypointsInsideMask, :]
            
            octaves = np.unique( keypointsInside[:, 2])
            maxOctave = 0
            if octaves.shape[0] > 0:
                maxOctave = np.max(octaves)
            
            mask = (keypoints[:, 0] > gt[0]) * (keypoints[:, 0] < gt[2]) * (keypoints[:, 1] > gt[1]) * (keypoints[:, 1] <  gt[3])
            
            images = []
            octPoints = []
            octScales  = []
            keypointsInRect = keypoints[mask, :]
            for i in range(int(maxOctave) + 1):
                scale = scales[i]
                ft = FASTex(process_color = color, edgeThreshold = edgeThreshold)
                octavePoints = keypointsInRect[keypointsInRect[:, 2] == i, :].copy()
                if octavePoints.shape[0] > 0:
                    dst = ft.getImageAtScale(i)
                    images.append(dst)
                    octavePoints[:, 0] *= scales[i]
                    octavePoints[:, 1] *= scales[i]
                    octavePoints[:, 5] *= scales[i]
                    octavePoints[:, 6] *= scales[i]
                    octavePoints[:, 7] *= scales[i]
                    octavePoints[:, 8] *= scales[i]
                    octPoints.append(octavePoints)
                    octScales.append(scale)
            
            f, axes = plt.subplots(1, 1 + len(images), figsize=(16, 3))
            if len(images) > 0:
                ax = axes[0]
            else:
                ax = axes
            
            if color == 1:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            zoom = img[gt[1]:gt[3], gt[0]:gt[2]]
            
            ax.imshow(zoom, cmap=pylab.gray(), interpolation='nearest')
            
    
            kpMask = keypoints[mask]
            kpMask[:, 0] = kpMask[:, 0] - gt[0]
            kpMask[:, 1] = kpMask[:, 1] - gt[1]
            kpMask[:, 7] = kpMask[:, 7] - gt[0]
            kpMask[:, 8] = kpMask[:, 8] - gt[1]
            
            ax.plot(kpMask[:, 0], kpMask[:, 1], 'ro')
            
            for k in range(kpMask.shape[0]):
                ax.plot([kpMask[k,0], kpMask[k,7]], [kpMask[k,1], kpMask[k,8]], 'r-')
            
            style = 'rx'
            if kpMask.shape[1] > 9:
                for k in range(3):
                    maski = kpMask[:, 9] == k + 1
                    if k == 1:
                        style = "rv"
                    if k == 2:
                        style = "rs"
                    if k == 4:
                        style = "bo"
                    if k == 5:
                        style = "yo"
                    
                    ax.plot([kpMask[maski,7]], [kpMask[maski,8]], style)
            
                
                
            
            mask = (keypointsInside[:, 0] > gt[0]) * (keypointsInside[:, 0] < gt[2]) * (keypointsInside[:, 1] > gt[1]) * (keypointsInside[:, 1] <  gt[3])
            kpMask = keypointsInside[mask]
            keypointsInside[:, 0] = keypointsInside[:, 0] - gt[0]
            keypointsInside[:, 1] = keypointsInside[:, 1] - gt[1]
            keypointsInside[:, 7] = keypointsInside[:, 7] - gt[0]
            keypointsInside[:, 8] = keypointsInside[:, 8] - gt[1]
            
            ax.plot(keypointsInside[:, 0], keypointsInside[:, 1], 'go')
            for k in range(keypointsInside.shape[0]):
                ax.plot([keypointsInside[k,0], keypointsInside[k,7]], [keypointsInside[k,1], keypointsInside[k,8]], 'g-')
                
            
            ax.set_xlim(0, gt[2] - max(0, gt[0]))
            ax.set_ylim((gt[3] - max(0, gt[1]), 0))
            
            line = mlines.Line2D(np.array([gt0[0] - gt[0], gt0[2] - gt[0], gt0[2] - gt[0], gt0[0] - gt[0], gt0[0] - gt[0]]), np.array([gt0[1] - gt[1], gt0[1] - gt[1], gt0[3] - gt[1], gt0[3] - gt[1], gt0[1] - gt[1]]), lw=5., alpha=0.4, color='b')
            ax.add_line(line)
            
            f.suptitle('Missing letter: {0} ({1})'.format(gt0[4], miss[0]))
            
            for ai in range(len(images)):
                ax = axes[ai + 1]
                scale = octScales[ai]
                gts = (gt[0] * scale, gt[1] * scale, gt[2] * scale, gt[3] * scale) 
                
                ax.plot(octPoints[ai][:, 0] - gts[0], octPoints[ai][:, 1] - gts[1], 'ro')
                
                zoom = images[ai][int(gt[1] * scale):int(gt[3] * scale), int(gt[0] * scale):int(gt[2] * scale)]
                ax.imshow(zoom, cmap=pylab.gray(), interpolation='nearest')
                ax.set_xlim(0, gts[2] - max(0, gts[0]))
                ax.set_ylim((gts[3] - max(0, gts[1]), 0))
            plt.show()
            
def draw_missed_segm(input_dir='/datagrid/personal/TextSpotter/FastTextEval/ICDAR-Train', color = 0, edgeThreshold = 12, inter = True, scalingFactor=0.5):
    
    ft = FASTex(process_color = color, edgeThreshold = edgeThreshold)
    
    d=input_dir
    subdirs = [os.path.join(d,o) for o in os.listdir(d) if os.path.isdir(os.path.join(d,o))]
    subdirs = np.sort(subdirs)
    for dir_name in subdirs:
        file_name = '{0}/evaluation.npz'.format(dir_name)
        if not os.path.exists(file_name):
            continue
        vars_dict = np.load(file_name) 
        inputDir = vars_dict['inputDir']
    
    missing_segm = vars['missing_segm']
    missing_segm = dict(missing_segm.tolist())
    
    segmDir = '{0}/segmentations'.format(inputDir)
    
    for image in  missing_segm.keys():
        arr =  missing_segm[image]
        if color == 1:
            img = cv2.imread(image)
        else:
            img = cv2.imread(image, 0)
        
        segmentations = ft.getCharSegmentations(img)
        keypoints = ft.getLastDetectionKeypoints()
        
        baseName = os.path.basename(image)
        baseName = baseName[:-4]
        
        segmImg = '{0}/{1}_GT.bmp'.format(segmDir, baseName)
        if not os.path.exists(segmImg):
            segmImg = '{0}/gt_{1}.png'.format(segmDir, baseName)
        segmImg = cv2.imread(segmImg)
        
        centers = segmImg[keypoints[:, 1].astype(int), keypoints[:, 0].astype(int)]
        keypointsInsideMask = centers == (255, 255, 255)
        keypointsInsideMask = np.invert(np.bitwise_and(np.bitwise_and(keypointsInsideMask[:, 0], keypointsInsideMask[:, 1]), keypointsInsideMask[:, 2]))
        keypointsInside = keypoints[keypointsInsideMask, :]
        
        
        f = plt.figure(num = 110)
        ax = f.add_subplot(111)
        ax.imshow(img, cmap=pylab.gray(), interpolation='nearest')
        
        style = "rx"
        for k in range(6):
            maski = keypoints[:, 9] == k + 1
            if k == 1:
                style = "rv"
            if k == 2:
                style = "ro"
            if k == 4:
                style = "bo"
            if k == 5:
                style = "yo"
            
            
            ax.plot(keypoints[maski, 0], keypoints[maski, 1], style)
        
        ax.plot(keypointsInside[:, 0], keypointsInside[:, 1], 'go')
        
        ax.set_xlim(0, img.shape[1])
        ax.set_ylim(img.shape[0], 0)
        
        for i in range(len(arr)):
            miss_gt = arr[i]
            
            line = mlines.Line2D(np.array([miss_gt[0], miss_gt[2], miss_gt[2], miss_gt[0], miss_gt[0]]), np.array([miss_gt[1], miss_gt[1], miss_gt[3], miss_gt[3], miss_gt[1]]), lw=5., alpha=0.4, color='b')
            ax.add_line(line)
            
            ax.set_title('Missing segmentation: {0}'.format(image))
            
            
        plt.show()

def plot_keypoints_histograms(vars_dict):
    
    f, ax = plt.subplots(2, sharex=True)
    hist = vars_dict['hist']
    ax[0].plot(hist)
    ax[0].set_title('FAST Keypoints Histogram')
    plt.xlabel('Intensity')
    plt.ylabel('Keypoints Count')
    ax[0].set_xlim([0, 255])
    hist = vars_dict['histFp']
    ax[1].plot(hist)
    ax[1].set_title('FAST Keypoints Histogram - False Positives')
    ax[1].set_xlim([0, 255])
        
    f, ax = plt.subplots(2, sharex=True)
    histDist = vars_dict['histDist']
    ax[0].plot(histDist)
    ax[0].set_title('FAST Keypoints Scores')
    plt.xlabel('Score')
    plt.ylabel('Keypoints Count')
    ax[0].set_xlim([0, 255])
    histDistFp = vars_dict['histDistFp']
    ax[1].plot(histDistFp)
    ax[1].set_title('FAST Keypoints Scores')
    ax[1].set_xlim([0, 255])
    
    f, ax = plt.subplots(4, sharex=True)
    histDist = vars_dict['histDistMax']
    ax[0].plot(histDist)
    ax[0].set_title('Keypoints on Letter')
    plt.xlabel('Distance')
    ax[0].set_ylabel('Keypoints Count')
    ax[0].set_xlim([0, 255])
    histDistFp = vars_dict['histDistMaxFp']
    ax[1].plot(histDistFp)
    ax[1].set_title('Keypoints Outside Letter')
    ax[1].set_ylabel('Keypoints Count')
    ax[1].set_xlim([0, 255])
    histDistMaxWhite = vars_dict['histDistMaxWhite']
    ax[2].plot(histDistMaxWhite)
    ax[2].set_title('Black Ink Keypoints')
    ax[2].set_ylabel('Keypoints Count')
    ax[2].set_xlim([0, 255])
    histDistMaxWhiteFp = vars_dict['histDistMaxWhiteFp']
    ax[3].plot(histDistMaxWhiteFp)
    ax[3].set_title('Black Ink Keypoints - Outside')
    ax[3].set_ylabel('Keypoints Count')
    ax[3].set_xlim([0, 255])
    
    
    hist2dDist = vars_dict['hist2dDist']
    hist2dDistFp = vars_dict['hist2dDistFp']
    fig, ax = plt.subplots(1, 2, sharey=True, figsize=(17, 8))
    ax[0].set_xlabel('Intensity')
    ax[0].set_ylabel('Max Distance')
    ax[0].set_xlim([0, 255])
    ax[0].set_ylim([0, 255])
    imgplot = ax[0].imshow(hist2dDist, interpolation='nearest', origin='low')
    ax[0].set_title('Kepoints Inside')
    imgplot.set_cmap('hot')
    ax[1].set_title('Kepoints Ouside')
    ax[1].set_xlabel('Intensity')
    ax[1].set_ylabel('Max Distance')
    imgplot = ax[1].imshow(hist2dDistFp, interpolation='nearest', origin='low')
    imgplot.set_cmap('hot')
    ax[1].set_xlim([0, 255])
    ax[1].set_ylim([0, 255])
    
    hist2dDist = vars_dict['hist2dDistScore']
    hist2dDistFp = vars_dict['hist2dDistScoreFp']
    fig, ax = plt.subplots()
    ax.set_xlabel('Score')
    ax.set_ylabel('DistMax')
    imgplot = ax.imshow(hist2dDist, interpolation='nearest', origin='low')
    ax.set_title('Kepoints Inside')
    imgplot.set_cmap('hot')
    fig, ax = plt.subplots()
    ax.set_title('Kepoints Ouside')
    ax.set_xlabel('Score')
    ax.set_ylabel('DistMax')
    imgplot = ax.imshow(hist2dDistFp, interpolation='nearest', origin='low')
    imgplot.set_cmap('hot')
    
    
def collect_histograms(img, segmImg, keypoints, values, diffValMax, keypointsTotalInside, diffMaxOctavesMap, diffScoreOctavesMap, hist, histFp, histDist, histDistMax, histDistMaxWhite, hist2dDist, hist2dDistScore, histDistFp, histDistMaxFp, histDistMaxWhiteFp, hist2dDistFp, hist2dDistScoreFp):
    
    centers = segmImg[keypoints[:, 1].astype(int), keypoints[:, 0].astype(int)]
    keypointsInsideMask = (centers == (255, 255, 255))
    keypointsInsideMask = np.invert( np.bitwise_and(np.bitwise_and(keypointsInsideMask[:, 0], keypointsInsideMask[:, 1]), keypointsInsideMask[:, 2]) )
    keypointsTotalInside += np.count_nonzero(keypointsInsideMask)
    
    centers2 = segmImg[keypoints[:, 8].astype(int), keypoints[:, 7].astype(int)]
    keypointsInsideMask2 = (centers2 == (255, 255, 255))
    keypointsInsideMask2 = np.invert( np.bitwise_and(np.bitwise_and(keypointsInsideMask2[:, 0], keypointsInsideMask2[:, 1]), keypointsInsideMask2[:, 2]) )
    
    keypointsInsideMask = np.bitwise_or(keypointsInsideMask, keypointsInsideMask2)
    keypointsInside = keypoints[keypointsInsideMask, :]
    maskBlackInk = img[keypoints[:, 8].astype(int), keypoints[:, 7].astype(int)] <= img[keypoints[:, 6].astype(int), keypoints[:, 5].astype(int)]
    maskWhiteInk = np.invert(maskBlackInk)
    
    
    octaves = np.unique( keypointsInside[:, 2])
    if len(octaves) > 0:
        maxOctave = np.max(octaves)
        difMaxInside = diffValMax[keypointsInsideMask]        
        for i in range(int(maxOctave) + 1):
            difMaxInsideOctave = difMaxInside[keypointsInside[:, 2] == i]
            keypointsOctaveScore = keypointsInside[keypointsInside[:, 2] == i, 3]
            if difMaxInsideOctave.shape[0] > 0:
                if diffMaxOctavesMap.has_key(i):
                    diffMaxOctavesMap[i] = np.hstack( (diffMaxOctavesMap[i], np.copy(difMaxInsideOctave)))
                    diffScoreOctavesMap[i] = np.hstack( (diffScoreOctavesMap[i], np.copy(keypointsOctaveScore) ) )
                else:
                    diffMaxOctavesMap[i] = np.copy(difMaxInsideOctave)
                    diffScoreOctavesMap[i] = np.copy(keypointsOctaveScore)
           
         
    bins = np.arange(255)
      
    if hist is None:
        hist = np.histogram(values[keypointsInsideMask], bins=bins)[0]
        histDist = np.histogram(keypointsInside[:, 3], bins=bins)[0]
        histDistMax = np.histogram(diffValMax[keypointsInsideMask], bins=bins)[0]
        histDistMaxWhite = np.histogram(diffValMax[np.bitwise_and(keypointsInsideMask, maskWhiteInk)], bins=bins)[0]
        hist2dDist = np.histogram2d(values[keypointsInsideMask], diffValMax[keypointsInsideMask], [bins, bins])[0]
        hist2dDistScore = np.histogram2d(keypointsInside[:, 3].astype(np.uint8), diffValMax[keypointsInsideMask], [bins, bins])[0]
    else:
        hist = np.add(hist,  np.histogram(values[keypointsInsideMask], bins)[0])
        histDist = np.add(histDist,  np.histogram(keypointsInside[:, 3], bins=bins)[0])
        histDistMax = np.add(histDistMax,  np.histogram(diffValMax[keypointsInsideMask], bins=bins)[0])
        histDistMaxWhite = np.add(histDistMaxWhite,  np.histogram(diffValMax[np.bitwise_and(keypointsInsideMask, maskWhiteInk)], bins=bins)[0])
        hist2dDist = np.add(hist2dDist,  np.histogram2d(values[keypointsInsideMask], diffValMax[keypointsInsideMask], [bins, bins])[0])
        hist2dDistScore = np.add(hist2dDistScore, np.histogram2d(keypointsInside[:, 3].astype(np.uint8), diffValMax[keypointsInsideMask], [bins, bins])[0])
    
    outsideMask = np.invert(keypointsInsideMask)
    keypointsOutside = keypoints[outsideMask, :]
    valuesFp = img[keypointsOutside[:, 1].astype(int), keypointsOutside[:, 0].astype(int)]
    
    if histFp is None:
        histFp = np.histogram(valuesFp, bins=bins)[0]
        histDistFp = np.histogram(keypointsOutside[:, 3], bins=bins)[0]
        histDistMaxFp = np.histogram(diffValMax[outsideMask], bins=bins)[0]
        histDistMaxWhiteFp = np.histogram(diffValMax[np.bitwise_and(outsideMask, maskWhiteInk)], bins=bins)[0]
        hist2dDistFp = np.histogram2d(values[outsideMask], diffValMax[outsideMask], [bins, bins])[0]
        hist2dDistScoreFp = np.histogram2d(keypointsOutside[:, 3], diffValMax[outsideMask], [bins, bins])[0]
    else:
        histFp = np.add(histFp,  np.histogram(valuesFp, bins)[0])
        histDistFp = np.add(histDistFp,  np.histogram(keypointsOutside[:, 3], bins=bins)[0])
        histDistMaxFp = np.add(histDistMaxFp,  np.histogram(diffValMax[outsideMask], bins=bins)[0])
        histDistMaxWhiteFp = np.add(histDistMaxWhiteFp,  np.histogram(diffValMax[np.bitwise_and(outsideMask, maskWhiteInk)], bins=bins)[0])
        hist2dDistFp = np.add(hist2dDistFp,  np.histogram2d(values[outsideMask], diffValMax[outsideMask], [bins, bins])[0])
        hist2dDistScoreFp = np.add(hist2dDistScoreFp, np.histogram2d(keypointsOutside[:, 3], diffValMax[outsideMask], [bins, bins])[0])
    
    return (hist, histFp, histDist, histDistMax, histDistMaxWhite, hist2dDist, hist2dDistScore, histDistFp, histDistMaxFp, histDistMaxWhiteFp, hist2dDistFp, hist2dDistScoreFp, keypointsInside)
    
    
def draw_missed_letters_figure(input_dir='/datagrid/personal/TextSpotter/FastTextEval/ICDAR-Train', color = 0, edgeThreshold = 13, inter = True, scalingFactor=0.5, segmList=[]):
    
    ft = FASTex(process_color = color, edgeThreshold = edgeThreshold)
    
    d=input_dir
    subdirs = [os.path.join(d,o) for o in os.listdir(d) if os.path.isdir(os.path.join(d,o))]
    subdirs = np.sort(subdirs)
    lastDir = ''
    for dir_name in subdirs:
        file_name = '{0}/evaluation.npz'.format(dir_name)
        if not os.path.exists(file_name):
            continue
        vars_dict = np.load(file_name) 
        inputDir = vars_dict['inputDir']
        lastDir = dir_name
        if 'letterKeypointHistogram' in vars_dict.keys():
            letterKeypointHistogram = vars_dict['letterKeypointHistogram']
            letterKeypointHistogram = dict(letterKeypointHistogram.tolist())
        
    print(lastDir)
    
    missing_letters = vars_dict['missing_letters']
    missing_letters = dict(missing_letters.tolist())
    
    keys = []
    ticks = []
    values = []
    values.append([])
    values.append([])
    values.append([])
    values.append([])
    ticks.append([])
    ticks.append([])
    ticks.append([])
    ticks.append([])
    listlen = 0
    for letter in letterKeypointHistogram.keys():
        keys.append(letter)
        values[0].append(0)
        ticks[0].append(listlen)
        values[1].append(0)
        ticks[1].append(listlen + 0.2)
        values[2].append(0)
        ticks[2].append(listlen + 0.4)
        values[3].append(0)
        ticks[3].append(listlen + 0.6)
        for num in letterKeypointHistogram[letter].keys():
            values[num][listlen] = letterKeypointHistogram[letter][num]
            
        listlen += 1
        
    indices = sorted(range(len(values[0])),key=lambda x:values[0][x])
    indices.reverse()
    
    missLetter = []
    imagesMiss = {}
    for letter in  np.asarray(keys)[np.asarray(indices)]:
        if not missing_letters.has_key(letter):
            continue
        arr =  missing_letters[letter]
        for i in range(len(arr)):
            miss = arr[i]
            
            if len(segmList) > 0:
                base = os.path.basename(miss[0])
                if not base in segmList:
                    continue
            
            missLetter.append(miss) 
            
            if imagesMiss.has_key(miss[0]):
                imagesMiss[miss[0]].append( miss[1] )
            else:
                imagesMiss[miss[0]] = []
                imagesMiss[miss[0]].append( miss[1] )
    
    for image in imagesMiss.keys():
        
        
        f = plt.figure(num = 250)    
        ax = f.add_subplot(111)
        imgc2 = cv2.imread(image)
        img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        imgc2 = cv2.cvtColor(imgc2, cv2.COLOR_BGR2RGB)
        ax.imshow(imgc2)
        segmentations = ft.getCharSegmentations(img)
        keypoints = ft.getLastDetectionKeypoints()
        
        octaves = np.unique( keypoints[:, 2])
        maxOctave = np.max(octaves)
        scales = ft.getImageScales()
        for i in range(int(maxOctave) + 1):
            octavePoints = keypoints[keypoints[:, 2] == i, :]
            c = 'red'
            if len(octavePoints) > 0 and octavePoints.shape[1] > 9:
                for k in range(6):
                    maski = octavePoints[:, 9] == k + 1
                    if k == 1:
                        style = "rv"
                    if k == 2:
                        style = "ro"
                    if k == 4:
                        style = "bo"
                        c = 'blue'
                    if k == 5:
                        style = "yo"
                        continue
                    
                    s = 10 / scales[i]
                    ax.scatter(octavePoints[maski, 0], octavePoints[maski, 1],c=c, s=s )
        
        for i in range(len(imagesMiss[image])):
            gt0 = imagesMiss[image][i]
            line = mlines.Line2D(np.array([gt0[0], gt0[2], gt0[2], gt0[0], gt0[0]]), np.array([gt0[1], gt0[1], gt0[3], gt0[3], gt0[1]]), lw=5., alpha=0.6, color='r')
            ax.add_line(line)
            
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
        plt.margins(0,0)
        ax.set_xlim([0, imgc2.shape[1]])
        ax.set_ylim([imgc2.shape[0], 0])
        ax.axes.get_xaxis().set_ticks([])
        ax.axes.get_yaxis().set_ticks([])
        plt.show()   
    
def draw_missed_letters_tile(input_dir='/datagrid/personal/TextSpotter/FastTextEval/ICDAR-Train', color = 0, edgeThreshold = 13, inter = True, scalingFactor=1.6, segmList=[]):
    
    ft = FASTex(process_color = color, edgeThreshold = edgeThreshold)
    
    d=input_dir
    subdirs = [os.path.join(d,o) for o in os.listdir(d) if os.path.isdir(os.path.join(d,o))]
    subdirs = np.sort(subdirs)
    lastDir = ''
    for dir_name in subdirs:
        file_name = '{0}/evaluation.npz'.format(dir_name)
        if not os.path.exists(file_name):
            continue
        vars_dict = np.load(file_name) 
        inputDir = vars_dict['inputDir']
        lastDir = dir_name
        if 'letterKeypointHistogram' in vars_dict.keys():
            letterKeypointHistogram = vars_dict['letterKeypointHistogram']
            letterKeypointHistogram = dict(letterKeypointHistogram.tolist())
        
    print(lastDir)
    
    missing_letters = vars_dict['missing_letters']
    missing_letters = dict(missing_letters.tolist())
    
    segmDir = '{0}/segmentations'.format(inputDir)
    segmDir = '/datagrid/personal/TextSpotter/evaluation-sets/icdar2013-Test/segmentations'
    
    keys = []
    ticks = []
    values = []
    values.append([])
    values.append([])
    values.append([])
    values.append([])
    ticks.append([])
    ticks.append([])
    ticks.append([])
    ticks.append([])
    listlen = 0
    for letter in letterKeypointHistogram.keys():
        keys.append(letter)
        values[0].append(0)
        ticks[0].append(listlen)
        values[1].append(0)
        ticks[1].append(listlen + 0.2)
        values[2].append(0)
        ticks[2].append(listlen + 0.4)
        values[3].append(0)
        ticks[3].append(listlen + 0.6)
        for num in letterKeypointHistogram[letter].keys():
            values[num][listlen] = letterKeypointHistogram[letter][num]
            
        listlen += 1
        
    indices = sorted(range(len(values[0])),key=lambda x:values[0][x])
    indices.reverse()
    
    border = 15
    
    missLetter = []
    imagesMiss = {}
    for letter in  np.asarray(keys)[np.asarray(indices)]:
        if not missing_letters.has_key(letter):
            continue
        arr =  missing_letters[letter]
        for i in range(len(arr)):
            miss = arr[i]
            
            if len(segmList) > 0:
                base = os.path.basename(miss[0])
                if not base in segmList:
                    continue
            
            missLetter.append(miss) 
            
            if imagesMiss.has_key(miss[0]):
                imagesMiss[miss[0]].append( miss[1] )
            else:
                imagesMiss[miss[0]] = []
                imagesMiss[miss[0]].append( miss[1] )
    
    rowSize = len(imagesMiss.keys())    
    f, axes = plt.subplots(2, len(imagesMiss.keys()))
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
    
    figNo = 0
    
    
    for image in imagesMiss.keys():
        if len(imagesMiss.keys()) > 1:
            ax0 = axes[0][figNo]
            ax = axes[1][figNo]
        else:
            ax0 = axes[figNo]
            ax = axes[figNo]
            
        figNo += 1
        if color == 1:
            img = cv2.imread(image)
        else:
            img = cv2.imread(image, 0)
        
        baseName = os.path.basename(image)
        baseName = baseName[:-4]
        segmImg = '{0}/{1}_GT.bmp'.format(segmDir, baseName)
        if not os.path.exists(segmImg):
            segmImg = '{0}/gt_{1}.png'.format(segmDir, baseName)
        segmImg = cv2.imread(segmImg)
            
        segmentations = ft.getCharSegmentations(img)
        keypoints = ft.getLastDetectionKeypoints()
        
        if color == 1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
        for i in range(len(imagesMiss[image])):
            if i == 0:
                orBox = imagesMiss[image][0]
            else:
                orBox = utils.union(orBox, imagesMiss[image][i])
            
        gt0 = orBox
        gt = [gt0[0] - border, gt0[1] - border, gt0[2] + border, gt0[3] + border ]
        gt[0] = max(0, gt[0])
        gt[1] = max(0, gt[1])
        gt[2] = min(img.shape[1], gt[2])
        gt[3] = min(img.shape[0], gt[3])
        zoom = img[gt[1]:gt[3], gt[0]:gt[2]]
        ax.imshow(zoom, cmap=pylab.gray(), interpolation='nearest')
        ax0.imshow(zoom, cmap=pylab.gray(), interpolation='nearest')
        
        centers = segmImg[keypoints[:, 1].astype(int), keypoints[:, 0].astype(int)]
        keypointsInsideMask = centers == (255, 255, 255)
        keypointsInsideMask = np.invert(np.bitwise_and(np.bitwise_and(keypointsInsideMask[:, 0], keypointsInsideMask[:, 1]), keypointsInsideMask[:, 2]))
        keypointsInside = keypoints[keypointsInsideMask, :]
        
        mask = (keypoints[:, 0] > gt[0]) * (keypoints[:, 0] < gt[2]) * (keypoints[:, 1] > gt[1]) * (keypoints[:, 1] <  gt[3])        
        
        kpMask = keypoints[mask]
        kpMask[:, 0] = kpMask[:, 0] - gt[0]
        kpMask[:, 1] = kpMask[:, 1] - gt[1]
        kpMask[:, 7] = kpMask[:, 7] - gt[0]
        kpMask[:, 8] = kpMask[:, 8] - gt[1]
        
        ax.plot(kpMask[:, 0], kpMask[:, 1], 'ro')
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        ax0.xaxis.set_ticklabels([])
        ax0.yaxis.set_ticklabels([])
        
        for k in range(kpMask.shape[0]):
            ax.plot([kpMask[k,0], kpMask[k,7]], [kpMask[k,1], kpMask[k,8]], 'r-')
        
        style = 'rx'
        if kpMask.shape[1] > 9:
            for k in range(3):
                maski = kpMask[:, 9] == k + 1
                if k == 1:
                    style = "rv"
                if k == 2:
                    style = "rs"
                if k == 4:
                    style = "bo"
                if k == 5:
                    style = "yo"
                
                ax.plot([kpMask[maski,7]], [kpMask[maski,8]], style)
        
        
        
        for i in range(len(imagesMiss[image])):
            
            gt0 = imagesMiss[image][i]
                        
            mask = (keypointsInside[:, 0] > gt[0]) * (keypointsInside[:, 0] < gt[2]) * (keypointsInside[:, 1] > gt[1]) * (keypointsInside[:, 1] <  gt[3])
            kpMask = keypointsInside[mask]
            keypointsInside[:, 0] = keypointsInside[:, 0] - gt[0]
            keypointsInside[:, 1] = keypointsInside[:, 1] - gt[1]
            keypointsInside[:, 7] = keypointsInside[:, 7] - gt[0]
            keypointsInside[:, 8] = keypointsInside[:, 8] - gt[1]
            
            ax.plot(keypointsInside[:, 0], keypointsInside[:, 1], 'go')
            for k in range(keypointsInside.shape[0]):
                ax.plot([keypointsInside[k,0], keypointsInside[k,7]], [keypointsInside[k,1], keypointsInside[k,8]], 'g-')
                
            
            ax.set_xlim(0, gt[2] - max(0, gt[0]))
            ax.set_ylim((gt[3] - max(0, gt[1]), 0))
            
            line = mlines.Line2D(np.array([gt0[0] - gt[0], gt0[2] - gt[0], gt0[2] - gt[0], gt0[0] - gt[0], gt0[0] - gt[0]]), np.array([gt0[1] - gt[1], gt0[1] - gt[1], gt0[3] - gt[1], gt0[3] - gt[1], gt0[1] - gt[1]]), lw=5., alpha=0.6, color='r')
            ax0.add_line(line)
            
    plt.show()    
    
def computeWordOvelap(imgc, word_gt, words, wordsOk, wordsFp):
    
    best_match = 0
    best_match2 = 0
    for det_word in words:
        try:
            cv2.rectangle(imgc, (det_word[0], det_word[1]), (det_word[2], det_word[3]), (0, 0, 255))
            for gt_box in word_gt:
                rect_int =  utils.intersect( det_word, gt_box )
                int_area = utils.area(rect_int)
                union_area = utils.area(utils.union(det_word, gt_box))
                
                ratio = int_area / float(union_area)
                ratio2 = int_area / utils.area(gt_box)
                if ratio > best_match:
                    best_match = ratio
                    w = det_word
                    best_match2 = ratio2
                    
            if best_match2 > 0.3:
                wordsOk.append(det_word)
            elif best_match == 0:
                wordsFp.append(det_word)
        except:
            pass
            
    return (best_match, best_match2)

evalPunctuation = False

def computeSegmOverlap(gt_rects, segmentations, MIN_SEGM_OVRLAP = 0.6):
    
    segm2chars = 0
    
    for k in range(len(gt_rects)):
        gt_rect = gt_rects[k]
        best_match = 0
        best_match_line = 0
        if (gt_rect[4] == ',' or gt_rect[4] == '.' or gt_rect[4] == '\'' or gt_rect[4] == ':' or gt_rect[4] == '-') and not evalPunctuation:
            continue    

        best_match2 = 0 
        for detId in range(segmentations.shape[0]):
            rectn = segmentations[detId, :]
            rect_int =  utils.intersect( rectn, gt_rect )
            int_area = utils.area(rect_int)
            union_area = utils.area(utils.union(rectn, gt_rect))
        
            ratio = int_area / float(union_area)
        
            if ratio > best_match:
                best_match = ratio
                
            if ratio > best_match_line and rectn[7] == 1.0 :
                best_match_line = ratio
            
            gt_rect[5] = best_match
            if best_match < MIN_SEGM_OVRLAP: 
                if k < len(gt_rects) - 1:
                    gt_rect2 = gt_rects[k + 1]
                    chars2Rect = utils.union(gt_rect2, gt_rect)
                    rect_int = utils.intersect( rectn, chars2Rect )
                    int_area = utils.area(rect_int)
                    union_area = utils.area(utils.union(rectn, chars2Rect))
                    ratio = int_area / float(union_area)
                    if ratio > best_match2:
                        if ratio > MIN_SEGM_OVRLAP:
                            segm2chars += 1
                            best_match2 = ratio
                            gt_rect[5] = ratio
                            gt_rect2[5] = ratio
    
def read_segm_data(input_dir, prefix = ""):
    
    d=input_dir
    subdirs = [os.path.join(d,o) for o in os.listdir(d) if os.path.isdir(os.path.join(d,o))]
    subdirs = np.sort(subdirs)  
    
    ms = []
    dirs = []
    
    for dir_name in subdirs:
        inputFile = '{0}/evaluation.npz'.format(dir_name)
        if not os.path.exists(inputFile):
            continue    
        vars_dict = np.load(inputFile)
    
        missing_segm = vars_dict['missing_segm']
        missing_segm = dict(missing_segm.tolist())
        
        ms.append(missing_segm) 
        dirs.append(prefix + os.path.basename(dir_name))
        
    return (ms, dirs)             
                            
def compare_missed_segm(input_dir='/datagrid/personal/TextSpotter/FastTextEval/experiments/segmentation', input_dir2='/datagrid/personal/TextSpotter/FastTextEval/experiments/segmentationg', showPictures = False):
    
    ft = FASTex()
    
    (ms, dirs) = read_segm_data(input_dir)
    (ms2, dirs2) = read_segm_data(input_dir2, 'g')
    
    ms.extend(ms2)
    dirs.extend(dirs2)
    
    sumHash = {}
    for j in np.arange(0, len(ms)):
        missing_segm = ms[j]
        for image in  missing_segm.keys():
            arr =  missing_segm[image]
            if not sumHash.has_key(image):
                sumHash[image] = arr
                continue
            for i in range(len(arr)):
                miss_gt = arr[i]
                check = sumHash[image]
                hasGt = False
                for k in range(len(check)):
                    miss_gt2 = check[k]
                    if miss_gt == miss_gt2:
                        hasGt = True 
                    
                if not hasGt:
                    sumHash[image].append(miss_gt)
                        
        
    missing_segm = ms[0]    
    
    data = []
    dataf = []
    gt_id = 0
    columns = ['Img', 'GT Id']
    for image in  sumHash.keys():
        arr =  sumHash[image]
        f = None
        for i in range(len(arr)):
            orValue = False
            miss_gt = arr[i]
            row = []
            row.append(os.path.basename(image))
            row.append(gt_id)
            gt_id += 1
            rowf = []
            
            for j in np.arange(0, len(ms)):
                if gt_id == 1:
                    columns.append(dirs[j])
                msj =  ms[j]
                hasSegmj = True
                val = 1
                if msj.has_key(image):
                    arrj =  msj[image]
                    for k in range(len(arrj)):
                        miss_gtj = arrj[k]
                        if miss_gtj == miss_gt:
                            hasSegmj = False
                            val = 0
                            break
                        
                row.append(hasSegmj)
                rowf.append(val)
                
                orValue = orValue or hasSegmj
            if orValue:
                rowf.append(1)
                    
            else:
                rowf.append(0)
                if showPictures:
                    img = cv2.imread(image)
                    imgg = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
                    if f == None:
                        
                        f, axes = plt.subplots(1, 2, figsize=(16, 3))
                        f.suptitle('Missing segmentation: {0}'.format(image))
                        ax = axes[0]
                        ax.imshow(img, cmap=pylab.gray(), interpolation='nearest')
                        ax = axes[1]
                        ax.imshow(imgg, cmap=pylab.gray(), interpolation='nearest')
                        orBox = miss_gt
                        
                        segmentations = ft.getCharSegmentations(imgg)
                        keypoints = ft.getLastDetectionKeypoints()
                        
                        style = 'rx'
                        for k in range(5):
                            maski = keypoints[:, 9] == k + 1
                            if k == 1:
                                style = "rv"
                            if k == 2:
                                style = "ro"
                            if k == 4:
                                style = "bo" 
                        
                            ax.plot(keypoints[maski, 0], keypoints[maski, 1], style)
            
                        for k in range(keypoints.shape[0]):
                            ax.plot([keypoints[k,0], keypoints[k,7]], [keypoints[k,1], keypoints[k,8]], 'r-')
                        ax = axes[0]
                        
                    else:
                        orBox = utils.union(orBox, miss_gt)    
            
                    line = mlines.Line2D(np.array([miss_gt[0], miss_gt[2], miss_gt[2], miss_gt[0], miss_gt[0]]), np.array([miss_gt[1], miss_gt[1], miss_gt[3], miss_gt[3], miss_gt[1]]), lw=5., alpha=0.6, color='r')
                    ax.add_line(line)
            
                    
            row.append(orValue)
                
            data.append(row)
            dataf.append(rowf)
        
        if f != None:    
            ax = axes[0]
            ax.set_xlim(orBox[0] - 20, orBox[2] + 20)
            ax.set_ylim(orBox[3] + 20, orBox[1] - 20)
            ax = axes[1]
            ax.set_xlim(orBox[0] - 20, orBox[2] + 20)
            ax.set_ylim(orBox[3] + 20, orBox[1] - 20)
            plt.show()
                               
            
    columns.append("OR")
    data = np.array(data)
    dataf = np.array(dataf)        
            
    df = pandas.DataFrame(data = data, columns=columns)
    #print(df)
    sumCols = dataf.sum(0)
    sumCols = dataf.shape[0] - sumCols
    print("Missing Segmentations:")
    print(sumCols)
    
    indices = np.argsort(sumCols)
    
    bestFactor = indices[1]
    missing_segm = ms[bestFactor]
    print( "Best factor: {0}".format(dirs[bestFactor])  )
    maskBest = dataf[:, bestFactor] == 0
    datafSec = dataf[maskBest, :]
    sumCols = datafSec.sum(0)
    sumCols = datafSec.shape[0] - sumCols      
        
    print("Missing Segmentations 2 best:")
    print(sumCols)
    
    indices = np.argsort(sumCols)
    bestFactor2 = indices[1]
    print( "Best factor 2: {0}, missing segmentations: {1} -> {2}".format(dirs[bestFactor2], datafSec.shape[0], sumCols[indices[1]])  )
    
    maskBest = datafSec[:, bestFactor2] == 0
    dataf3 = datafSec[maskBest, :]
    sumCols = dataf3.sum(0)
    sumCols = dataf3.shape[0] - sumCols      
    
    indices = np.argsort(sumCols)
    bestFactor2 = indices[1]
    print( "Best factor 3: {0}, missing segmentations: {1} -> {2}".format(dirs[bestFactor2], dataf3.shape[0], sumCols[indices[1]])  )
    
    

    
        
if __name__ == '__main__':
    
    draw_missed_letters('/tmp/evalTest')
    
    segmList = []
    segmList.append( 'img_49.jpg' )
    segmList.append( 'img_168.jpg' )
    segmList.append( 'img_206.jpg' )
    segmList.append( 'img_86.jpg' )
    segmList.append( 'img_205.jpg' )
    segmList.append( 'img_232.jpg' )
    segmList.append( 'img_34.jpg' )
    segmList.append( 'img_230.jpg' )
    draw_missed_letters_figure(input_dir='/datagrid/personal/TextSpotter/FastTextEval/ICDAR-Test', color = 0, edgeThreshold = 13, inter = True, segmList=segmList)
    
    '''
    compare_missed_segm(input_dir='/datagrid/personal/TextSpotter/FastTextEval/experiments/segmentation', input_dir2='/datagrid/personal/TextSpotter/FastTextEval/experiments/segmentationg', showPictures = True)
    
    plotSegmRoc('/datagrid/personal/TextSpotter/FastTextEval/experiments/segmentation')
    '''   
