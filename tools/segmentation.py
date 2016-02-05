'''
Created on Jan 8, 2015

@author: busta
'''

import numpy as np
import cv2
import sys
from ft import FASTex
from vis import draw_keypoints


if __name__ == '__main__':
    
    outputDir = '/tmp'
    edgeThreshold = 13
    
    ft = FASTex(edgeThreshold= edgeThreshold, nlevels=-1, minCompSize = 4)
    
    imgName = '/datagrid/personal/TextSpotter/evaluation-sets/bornDigital/img_100.png'
        
    if len(sys.argv) > 1:
        if sys.argv[1].endswith(".png") or sys.argv[1].endswith(".jpg"):
            imgName = sys.argv[1]
        
    img = cv2.imread(imgName, 0)
    imgc = cv2.imread(imgName)
    
    #print(out)
    segmentations = ft.getCharSegmentations(img, outputDir, 'base')
    print segmentations
    for i in range(segmentations.shape[0]):
        rectn = segmentations[i, :]
        rectn[2] += rectn[0]
        rectn[3] += rectn[1]
        
        mask = ft.getSegmentationMask(i)
        
    '''
    for i in range(lines.shape[0]):
            line = lines[i]
            if line[25] == 0:
                continue
            lineSegm = ft.getNormalizedLine(i)
            cv2.imshow("ts", lineSegm)
            cv2.waitKey(0)
    '''
    
    keypoints = ft.getLastDetectionKeypoints()    
    draw_keypoints(imgc, keypoints, edgeThreshold, inter = True, color = 0)
    
    
    while imgc.shape[1] > 1024:
        shape = img.shape
        shapet = ( shape[0] / 2, shape[1] / 2) 
        dst = np.zeros(shapet, dtype=np.uint8)
        dst = cv2.resize(imgc, (0,0), fx=0.5, fy=0.5)
        imgc = dst
        
        
    pass
        