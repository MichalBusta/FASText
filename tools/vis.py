'''
Created on Dec 12, 2014

@author: busta
'''

import numpy as np
import cv2

import pylab
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector

from matplotlib.text import Text

from _collections import defaultdict
import sys
import ftext
import os

added_objects = []
last_selection = ()

fastOffsetsC = [[0,  4], [ 1,  4], [ 2,  4], [ 3,  3], [ 4, 2], [ 4, 1], [ 4, 0], [ 4, -1], [ 4, -2], [ 3, -3], [ 2, -4], [ 1, -4], [0, -4], [-1, -4], [-2, -4], [-3, -3], [-4, -2], [-4, -1], [-4, 0], [-4,  1], [-4,  2], [-3,  3], [-2,  4], [-1,  4]]
fastOffsets = [[0,  3], [ 1,  3], [ 2,  2], [ 3,  1], [ 3, 0], [ 3, -1], [ 2, -2], [ 1, -3], [0, -3], [-1, -3], [-2, -2], [-3, -1], [-3, 0], [-3,  1], [-2,  2], [-1,  3]]
fastCorners = [[ 2,  3], [ 3,  2], [ 3, -2], [ 2, -3], [-3, -2], [-2, -3], [-3,  2], [-2,  3]]
fastAll = fastOffsets
fastAll.extend(fastCorners)

drawGrayScale = False

def segment_fasts(img, point, distFunction, ax, threshold):
    
    pointmarkers = np.asarray(fastOffsets)
    pointmarkers[:, 0] += point[0]
    pointmarkers[:, 1] += point[1]
    negative = False
    
    if img[point[1], point[0]] <= img[point[6], point[5]]:
        negative = True
    
    otsu_t = otsu_threshold(img, point, pointmarkers, negative=negative)
    
    
    

def floodFill(img, x, y, distFunction, ax, threshold):
    global added_objects
    for k in range( len(added_objects ) ) :
        plotObject =  added_objects[k]
        if type(plotObject) is matplotlib.collections.PathCollection:
            plotObject.remove()
        elif type(plotObject) is matplotlib.text.Text:
            plotObject.remove()
        else:
            plotObject[0].remove()
    added_objects = []
    
    added_objects = []
    # Starting at x and y, changes any adjacent
    # characters that match oldChar to newChar.
    worldWidth = img.shape[1]
    worldHeight = img.shape[0]
    visited = {}
    seedx = x
    seedy = y

    theStack = [ (x, y) ]
    fillx = []
    filly = []
    while len(theStack) > 0:
        x, y = theStack.pop()
        if len(fillx) > 800:
            break
        visited[(x, y)] = 1

        if not distFunction(img[y, x], threshold):
            # Base case. If the current x, y character is not the oldChar,
            # then do nothing.
            continue

        # Change the character at world[x][y] to newChar
        fillx.append(x)
        filly.append(y)

        if x > 0: # left
            if not visited.has_key((x-1, y)):
                theStack.append( (x-1, y) )

        if y > 0: # up
            if not visited.has_key((x, y-1)):
                theStack.append( (x, y-1) )

        if x < worldWidth-1: # right
            if not visited.has_key( (x+1, y) ):
                theStack.append( (x+1, y) )

        if y < worldHeight-1: # down
            if not visited.has_key( (x, y+1) ):
                theStack.append( (x, y+1) )
    
    added_objects.append(ax.plot(fillx, filly, 'wo')) 
    print('area: {0}'.format(len(fillx)))    
    ax.figure.canvas.draw()

def line_select_callback(eclick, erelease):
    'eclick and erelease are the press and release events'
    x1, y1 = eclick.xdata, eclick.ydata
    x2, y2 = erelease.xdata, erelease.ydata
    print ("(%3.2f, %3.2f) --> (%3.2f, %3.2f)" % (x1, y1, x2, y2))
    for i in range(len(toggle_selector.RS)):
        if toggle_selector.RS[i].to_draw.get_height() > 1:
            
            mask = np.logical_and( np.logical_and( toggle_selector.RS[i].keypoints[:, 0] > x1,  toggle_selector.RS[i].keypoints[:, 0] < x2), np.logical_and(toggle_selector.RS[i].keypoints[:, 1] > y1, toggle_selector.RS[i].keypoints[:, 1] < y2))
            idx = np.nonzero(mask)[0]
            inpoints = toggle_selector.RS[i].keypoints[mask, :]
            for j in range( inpoints.shape[0] ):
                drawFast(inpoints[j,:], toggle_selector.RS[i].ax, toggle_selector.RS[i].image, toggle_selector.RS[i].threshold, toggle_selector.RS[i].octIdx[0][idx[j]])
            
            #toggle_selector.RS[i].ax.plot(inpoints[:, 5], inpoints[:, 6], 'go')
            #toggle_selector.RS[i].to_draw.set_height(0)
            toggle_selector.RS[i].ax.figure.canvas.draw()

class FastDrawer:
    def __init__(self, ax, img, threshold):
        
        self.ax = ax
        self.img = img
        self.threshold = threshold
        
        self.cid = plt.connect('button_press_event', self)

    def __call__(self, event):
        if event.dblclick:
            if event.inaxes!=self.ax: return
            point = [event.xdata, event.ydata]
            drawFast(point, self.ax, self.img, self.threshold, -1)
        elif event.button == 3:
            if event.inaxes!=self.ax: return
            if len(last_selection) > 0:
                point = last_selection[0]
                threshold = last_selection[1]
                distFunction = last_selection[2] 
                floodFill(self.img, point[0], point[1], distFunction, self.ax, threshold)
                
            

def toggle_selector(event):
    if event.dblclick:
        #print event
        return 
    
def cdist(e1, e2):
    ur1 = int(e1[2])
    ur2 = int(e2[2])
    rmean = (  ur1 + ur2 ) / 2
    r = ur1 - ur2
    ug1 = int(e1[1])
    ug2 = int(e2[1])
    g = ug1 - ug2
    ub1 = int(e1[0])
    ub2 = int(e2[0])
    b = ub1 - ub2;
    return (((512+rmean)*r*r)>>8) + 4*g*g + (((767-rmean)*b*b)>>8)

def lower_then_b(e1, threshold):
    return e1[2] < threshold

def lower_then(e1, threshold):
    return e1 < threshold

def greater_then(e1, threshold):
    return e1 > threshold

def greater_then_b(e1, threshold):
    return e1[2] > threshold

from skimage.exposure import histogram
def otsu_threshold(img, point, offsets, negative = False):
    
    hist = defaultdict(int)
    max = 0
    sum = 0
    hist[img[point[1], point[0]]] += 1
    sum += img[point[1], point[0]]
    for  i in range(16):
        hist[img[offsets[i, 1], offsets[i, 0]]] += 1
        sum += img[offsets[i, 1], offsets[i, 0]]
    wB = 0
    threshold1 = 0
    next = -1
    count = 0
    sumB = 0
    for val in sorted(hist.keys()):
        wB += hist[val]
        wF = 17 - wB
        if next == -1:
            next = val
            
        if wF == 0:
            break
        sumB += val * hist[val]
        mB = sumB / float(wB)
        mF = (sum - sumB) / float(wF)
        between = wB * wF * pow(mB - mF, 2)
        
    
        
        if between >= max:
            threshold1 = val
            max = between
            count = wB
            next = -1
        
    
    if negative:
        return (2 * next + threshold1) / 3
    
    return (2 * threshold1 + next) / 3

fastOffsets12 = [[0,  2], [ 1,  2], [ 2,  1], [ 2, 0], [ 2, -1], [ 1, -2], [0, -2], [-1, -2], [-2, -1], [-2, 0], [-2,  1], [-1,  2]]
fastCorners12 = [[ 2, 2], [2, -2], [-2,  -2], [-2,  2]]
fastOffsets8 = [[0,  1], [ 1,  1], [1, 0], [ 1, -1], [0, -1], [-1, -1], [-1, 0], [-1,  1]]
showFast8 = True
showFast16 = False
s = 50

from matplotlib.patches import Wedge

def drawFast(point, ax, img, threshold, idx):
    #s = 2000
    point[0] = round(point[0])
    point[1] = round(point[1])
    global added_objects
    for k in range( len(added_objects ) ) :
        plotObject =  added_objects[k]
        if type(plotObject) is matplotlib.collections.PathCollection:
            plotObject.remove()
        elif type(plotObject) is matplotlib.text.Text:
            plotObject.remove()
        elif type(plotObject) is Wedge:
            plotObject.remove()
        else:
            plotObject[0].remove()
    added_objects = []
    
    if idx > -1:
        strokes = ftext.getKeypointStrokes(idx)
        if strokes.shape[0] > 0:
            added_objects.append( ax.scatter(strokes[:,0], strokes[:,1], c='yellow', s=s) )
    
    global last_selection
    pointmarkers = np.asarray(fastAll)
    pointmarkers[:, 0] += int(round(point[0]))
    pointmarkers[:, 1] += int(round(point[1]))
    pointmarkers12 = np.asarray(fastOffsets12)
    pointmarkers12[:, 0] += int(round(point[0]))
    pointmarkers12[:, 1] += int(round(point[1]))
    if len(point) > 4:
        print('point: {0} - {1} - {2} - {3}'.format(point, img[round(point[1]), round(point[0])], img[round(point[6]), round(point[5])], img[round(point[8]), round(point[7])]))
    else:
        print('point: {0} - {1}'.format(point, img[round(point[1]), round(point[0])]))
    
    np.round(pointmarkers, 0, pointmarkers)
    if len(img.shape) == 3:
        
        diff = (img[pointmarkers12[:, 1], pointmarkers12[:, 0]]).astype(np.int) - img[point[1], point[0]]
        if len(point) > 10:
            diff = diff[:, point[11]]
        else:
            diff = diff[:, 0]
        '''
        diff = np.zeros((pointmarkers12.shape[0]), dtype=np.int)
        for i in range(pointmarkers12.shape[0]):
            diff[i] = cdist(img[round(point[1]), round(point[0])], img[pointmarkers12[i, 1], pointmarkers12[i, 0]])
        print(diff)
        '''    
        masksame = diff < threshold
        maskDiff = np.invert(masksame)
        added_objects.append( ax.scatter(pointmarkers12[masksame,0], pointmarkers12[masksame,1], c='black', s=s) )
        added_objects.append( ax.scatter(pointmarkers12[maskDiff,0], pointmarkers12[maskDiff,1], c='white', s=s) )
        added_objects.append( ax.scatter([round(point[0])], [round(point[1])], c='black') )
        
        if len(point) > 4:
            #added_objects.append(  )
            added_objects.append(ax.plot([point[0], point[7]], [point[1], point[8]], 'r-'))
            #added_objects.append( ax.plot([point[0], point[5]], [point[1], point[6]], 'g-') )
        
        if point[10] == 0:   
            last_selection = (point, img[round(point[1]), round(point[0])][0] + threshold * 4, greater_then_b)
        else :
            last_selection = (point, point[3], greater_then_b)
        return
    diff = (img[pointmarkers12[:, 1], pointmarkers12[:, 0]]).astype(np.int) - img[point[1], point[0]]
    print((img[pointmarkers12[:, 1], pointmarkers12[:, 0]]).astype(np.int))
    if len(point) < 3:
        added_objects.append( ax.scatter( [round(point[0])], [round(point[1])], c='black' ) )
        maxDiff = np.argmax(np.abs(diff))
        minDiff = np.argmin(np.abs(diff))
        point = ( point[0], point[1], point[0], point[1], 0, pointmarkers12[maxDiff, 0], pointmarkers12[maxDiff, 1], pointmarkers12[minDiff, 0], pointmarkers12[minDiff, 1])
        print(diff)    
    
    added_objects.append( ax.plot([point[0], point[7]], [point[1], point[8]], 'r-') )
    added_objects.append( ax.plot([point[0], pointmarkers12[0, 0]], [point[1], pointmarkers12[0, 1]], 'r-') )
    #added_objects.append( ax.plot([point[0], pointmarkers12[0, 0]], [point[1], pointmarkers12[0, 1]], 'r-') ) 
    #ax.plot([point[0], point[7]], [point[1], point[8]], 'r-')
    #added_objects.append( ax.plot([point[0], point[5]], [point[1], point[6]], 'g-') )
    #added_objects.append( ax.plot([point[0], pointmarkers12[7, 0]], [point[1], pointmarkers12[7, 1]], 'b-') )
    #added_objects.append( ax.plot([point[0], pointmarkers12[11, 0]], [point[1], pointmarkers12[11, 1]], 'b-') )
    
    if img[point[8], point[7]] <= img[point[6], point[5]]:
        print(diff)
        masksame = diff <= threshold
        maskDiff = np.invert(masksame)
        if showFast16:
            added_objects.append( ax.scatter(pointmarkers[masksame,0], pointmarkers[masksame,1], c='black') )
            added_objects.append( ax.scatter(pointmarkers[maskDiff,0], pointmarkers[maskDiff,1], c='white') )
        otsu_t = otsu_threshold(img, point, pointmarkers, negative=True)
        print('otsu: {0}'.format(otsu_t))
        print('otsur: {0}'.format( point[3] +  img[point[1], point[0]] ))
        last_selection = (point, otsu_t, lower_then)
        
        if True:
            pointmarkers12 = np.asarray(fastOffsets12)
            pointmarkers12[:, 0] += int(round(point[0]))
            pointmarkers12[:, 1] += int(round(point[1]))
            
            '''
            pointc12 = np.asarray(fastCorners12)
            pointc12[:, 0] += round(point[0])
            pointc12[:, 1] += round(point[1])
            '''
            
            added_objects.append( ax.scatter([point[0]], [point[1]],c='red', s=s ) )
            
            diff12 = ( img[pointmarkers12[:, 1], pointmarkers12[:, 0]].astype(np.int) - (img[point[1], point[0]])).astype(np.int)
            #diffc12 = ( img[pointc12[:, 1], pointc12[:, 0]].astype(np.int) - (img[point[1], point[0]])).astype(np.int)
            print('diff12: {0} ({1})'.format(diff12, threshold)) 
            masksame = diff12 <= threshold
            #masksamec = diffc12 <= threshold
            added_objects.append( ax.scatter(pointmarkers12[masksame,0], pointmarkers12[masksame,1],c='red', s=s ) )
            #added_objects.append( ax.scatter(pointc12[masksamec,0], pointc12[masksamec,1],c='red', s=s ) )
            masksame = np.invert(masksame)
            #masksamec = np.invert(masksamec)
            added_objects.append( ax.scatter(pointmarkers12[masksame,0], pointmarkers12[masksame,1],c='white', s=s ) )
            
            #added_objects.append( ax.scatter(pointc12[masksamec,0], pointc12[masksamec,1],c='white', s=s ) )
            xoffset = -0.05
            for i in range( pointmarkers12.shape[0]):
                if i > 9:
                    xoffset = -0.1
                added_objects.append(ax.text(pointmarkers12[i][0] + xoffset, pointmarkers12[i][1] + 0.05, '{0}'.format(i)))
            
            #added_objects.append(ax.text(pointc12[0][0] -0.05, pointc12[0][1] + 0.05, '{0}\''.format(2)))
            #added_objects.append(ax.text(pointc12[1][0] -0.05, pointc12[1][1] + 0.05, '{0}\''.format(5)))
            #added_objects.append(ax.text(pointc12[2][0] -0.05, pointc12[2][1] + 0.05, '{0}\''.format(8)))
            #added_objects.append(ax.text(pointc12[3][0] -0.1, pointc12[3][1] + 0.05, '{0}\''.format(11)))
            
            #w1 = Wedge((round(point[0]), round(point[1])), 2.6, 145, 72, fc='none', ec='green')
            #added_objects.append(ax.add_artist(w1))
            #added_objects.append(ax.text(pointmarkers12[6][0] -0.5, pointmarkers12[6][1] - 0.8, '$P_b$'.format(11), fontsize=24, color='green'))
            
            '''
            w1 = Wedge((round(point[0]), round(point[1])), 2.6, 320, 50, fc='none', ec='green')
            added_objects.append(ax.add_artist(w1))
            added_objects.append(ax.text(pointmarkers12[2][0] + 0.8, pointmarkers12[2][1] + 0.5, '$P_b$'.format(11), fontsize=24, color='green'))
            
            w1 = Wedge((round(point[0]), round(point[1])), 2.6, 105, 255, fc='none', ec='green')
            added_objects.append(ax.add_artist(w1))
            added_objects.append(ax.text(pointmarkers12[8][0] - 0.8, pointmarkers12[8][1] - 0.5, '$P^\prime_b$'.format(11), fontsize=24, color='green'))
            '''
            
            
            if showFast8 and True:
                pointmarkers8 = np.asarray(fastOffsets8)
                pointmarkers8[:, 0] += int(round(point[0]))
                pointmarkers8[:, 1] += int(round(point[1]))
                diff8 = ( img[pointmarkers8[:, 1], pointmarkers8[:, 0]].astype(np.int) - (img[point[1], point[0]])).astype(np.int)
                print('diff8: {0} ({1})'.format(diff8, threshold)) 
                masksame = diff8 <= threshold
                if (point[1] - point[6]) == 2:
                    if ( point[0] - point[5] ) == 0:
                        added_objects.append( ax.scatter(pointmarkers8[0,0], pointmarkers8[0,1],c='magenta', s=s ) )
                        added_objects.append( ax.scatter(pointmarkers8[7,0], pointmarkers8[7,1],c='magenta', s=s ) )
                if (point[1] - point[6]) == -1:
                    if ( point[0] - point[5] ) == -2:
                        added_objects.append( ax.scatter(pointmarkers8[2,0], pointmarkers8[2,1],c='magenta', s=s ) )
                        added_objects.append( ax.scatter(pointmarkers8[3,0], pointmarkers8[3,1],c='magenta', s=s ) )
                        
                        added_objects.append( ax.scatter(pointmarkers8[7,0], pointmarkers8[7,1],c='magenta', s=s ) )
                        added_objects.append( ax.scatter(pointmarkers8[0,0], pointmarkers8[0,1],c='magenta', s=s ) )
                        #added_objects.append( ax.scatter(pointmarkers8[7,0], pointmarkers8[7,1],c='red' ) )
                #added_objects.append( ax.scatter(pointmarkers8[masksame,0], pointmarkers8[masksame,1],c='black' ) )
                #masksame = np.invert(masksame)
                added_objects.append( ax.scatter(pointmarkers8[masksame,0], pointmarkers8[masksame,1],c='magenta', s=s ) )
            '''
            pointmarkers12 = np.asarray(fastOffsetsC)
            pointmarkers12[:, 0] += round(point[0])
            pointmarkers12[:, 1] += round(point[1])
            diff12 = ( img[pointmarkers12[:, 1], pointmarkers12[:, 0]].astype(np.int) - (img[point[1], point[0]])).astype(np.int) 
            masksame = diff12 <= threshold
            added_objects.append( ax.scatter(pointmarkers12[masksame,0], pointmarkers12[masksame,1],c='#000044' ) )
            masksame = np.invert(masksame)
            added_objects.append( ax.scatter(pointmarkers12[masksame,0], pointmarkers12[masksame,1],c='#ffffaa' ) )
            '''
            
            
            
        
    else:
        vt = img[round(point[1]), round(point[0])] - threshold
        diff = img[point[1], point[0]] - (img[pointmarkers[:, 1], pointmarkers[:, 0]]).astype(np.int)  
        print(diff)
        masksame = img[pointmarkers[:, 1], pointmarkers[:, 0]] >= vt
        maskDiff = np.invert(masksame)
        if showFast16:
            added_objects.append( ax.scatter(pointmarkers[masksame,0], pointmarkers[masksame,1], c='black') )
            added_objects.append( ax.scatter(pointmarkers[maskDiff,0], pointmarkers[maskDiff,1], c='white') )
        otsu_t = otsu_threshold(img, point, pointmarkers, negative=False)
        print('otsu: {0}'.format(otsu_t))
        print('otsur: {0}'.format( img[point[1], point[0]] - point[3]  ))
        last_selection = (point, otsu_t, greater_then)
        
        if True:
            pointmarkers12 = np.asarray(fastOffsets12)
            pointmarkers12[:, 0] += int(round(point[0]))
            pointmarkers12[:, 1] += int(round(point[1]))
            
            diff12 = ((img[point[1], point[0]]) - img[pointmarkers12[:, 1], pointmarkers12[:, 0]].astype(np.int)).astype(np.int) 
            print('diff12: {0}'.format(diff12))
            masksame = diff12 <= threshold
            added_objects.append( ax.scatter(pointmarkers12[masksame,0], pointmarkers12[masksame,1],c='red' ) )
            masksame = np.invert(masksame)
            added_objects.append( ax.scatter(pointmarkers12[masksame,0], pointmarkers12[masksame,1],c='white' ) )
            
            if showFast8:
                pointmarkers8 = np.asarray(fastOffsets8)
                pointmarkers8[:, 0] += int(round(point[0]))
                pointmarkers8[:, 1] += int(round(point[1]))
                diff8 = ( (img[point[1], point[0]]) - img[pointmarkers8[:, 1], pointmarkers8[:, 0]].astype(np.int)).astype(np.int)
                print('diff8: {0} ({1})'.format(diff8, threshold)) 
                masksame = diff8 <= threshold
                added_objects.append( ax.scatter(pointmarkers8[masksame,0], pointmarkers8[masksame,1],c='black' ) )
                #masksame = np.invert(masksame)
                #added_objects.append( ax.scatter(pointmarkers8[masksame,0], pointmarkers8[masksame,1],c='#aaffff' ) )
            
            '''
            pointmarkers12 = np.asarray(fastOffsetsC)
            pointmarkers12[:, 0] += round(point[0])
            pointmarkers12[:, 1] += round(point[1])
            diff12 = ((img[point[1], point[0]]) - img[pointmarkers12[:, 1], pointmarkers12[:, 0]].astype(np.int)).astype(np.int) 
            print('diff12: {0}'.format(diff12))
            masksame = diff12 <= threshold
            added_objects.append( ax.scatter(pointmarkers12[masksame,0], pointmarkers12[masksame,1],c='#000044' ) )
            masksame = np.invert(masksame)
            added_objects.append( ax.scatter(pointmarkers12[masksame,0], pointmarkers12[masksame,1],c='#ffffaa' ) )
            '''


def draw_keypoints(img, keypoints, threshold, inter = True, color = 1):
    
    scales = ftext.getImageScales()
    #s = 100
    
    octaves = np.unique( keypoints[:, 2])
    if octaves.shape[0] == 0:
        return
    maxOctave = np.max(octaves)
    images = []
    selectors = []
    drawers = []
    for i in range(int(maxOctave) + 1):
        scale = scales[i]
        dst = ftext.getImageAtScale(i)
        cv2.imwrite("/tmp/pycv-{0}.png".format(i), dst)
        if color == 1:
            images.append(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
        else:
            '''
            shape = img.shape
            shapet = ( shape[0] * scale, shape[1] * scale) 
            dst = np.zeros(shapet, dtype=np.uint8)
            dst = cv2.resize(img, (0,0), fx=scale, fy=scale)
            images.append(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
            '''
            images.append(dst)
            
            #cv2.imshow("img", dst)
            #cv2.waitKey(0)
        
    
    ax_tuple = []
       
    for i in range(int(maxOctave) + 1):
        f = plt.figure(num = i)
        ax = f.add_subplot(111)
        ax_tuple.append(ax)
        
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
        plt.margins(0,0)
        
        #tmp = cv2.cvtColor(, cv2.COLOR_GRAY2RGB)
        ax_tuple[i].imshow(images[i], interpolation='nearest', cmap=pylab.gray())
        ax_tuple[i].grid(True)
        ax_tuple[i].axis('off') 
        ax_tuple[i].set_xlim([0, images[i].shape[1]])
        ax_tuple[i].set_ylim([images[i].shape[0], 0])
        ax_tuple[i].axes.get_xaxis().set_ticks([])
        ax_tuple[i].axes.get_yaxis().set_ticks([])
        
        maskOct = keypoints[:, 2] == i
        octIdx = np.nonzero(maskOct)
        octavePoints = keypoints[keypoints[:, 2] == i, :]
        octavePoints[:, 0] *= scales[i]
        octavePoints[:, 1] *= scales[i]
        octavePoints[:, 5] *= scales[i]
        octavePoints[:, 6] *= scales[i]
        octavePoints[:, 7] *= scales[i]
        octavePoints[:, 8] *= scales[i]
        style = 'rx'
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
                
                if drawGrayScale:
                    style = 'wv'
                    if k == 1:
                        style = "wv"
                    if k == 2:
                        style = "wv"
                    if k == 4:
                        style = "wo"
                        c = 'blue'
                    c = style
                
                ax_tuple[i].scatter(octavePoints[maski, 0], octavePoints[maski, 1],c=c, s=s )
                #ax_tuple[i].plot(octavePoints[maski, 0], octavePoints[maski, 1], style, markersize = 10) 
        else:
            ax_tuple[i].plot(octavePoints[:, 0], octavePoints[:, 1], style)
        #ax_tuple[i].plot(octavePoints[:, 6] * scales[i], octavePoints[:, 7] * scales[i], 'bo')
        #for j in range( octavePoints.shape[0] ):
        #    ax_tuple[i].plot([octavePoints[j,0] * scales[i], octavePoints[j,6]* scales[i]], [octavePoints[j,1] * scales[i], octavePoints[j,7]* scales[i]], 'r-')
        if inter:
            rs = RectangleSelector(ax_tuple[i], line_select_callback,
                                       drawtype='box', useblit=True,
                                       button=[1,3], # don't use middle button
                                       minspanx=5, minspany=5,
                                       spancoords='pixels')
            rs.keypoints = octavePoints
            rs.octIdx = octIdx
            rs.image = images[i]
            rs.threshold = threshold
            selectors.append(rs)
            drawer = FastDrawer(ax_tuple[i], images[i], threshold)
            drawers.append(drawer)
    
    if inter:
        toggle_selector.RS = selectors
        plt.show() 
        plt.show(block=False)
    else:
        plt.show()
    

def draw_letter_histogram(letterKeypointHistogram, title = 'Letter FAST Keypoints Histogram', figNo = 101):
    
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
    for i in range(len(indices)):
        if values[0][indices[i]] == 1:
            break
    indices = indices[0:i]
    ticks[0] = ticks[0][0:i]
    ticks[1] = ticks[1][0:i]
    
        
    f = plt.figure(num = figNo, figsize=(16, 8))
    ax = f.add_subplot(111)
    plt.bar(ticks[0], np.asarray(values[0])[np.asarray(indices)], align='center', width=0.25, facecolor='r')
    plt.bar(ticks[1], np.asarray(values[1])[np.asarray(indices)], align='center', width=0.25, facecolor='b')
    #plt.bar(ticks[2], np.asarray(values[2])[np.asarray(indices)], align='center', width=0.15, facecolor='g', alpha=0.5)
    #plt.bar(ticks[3], np.asarray(values[3])[np.asarray(indices)], align='center', width=0.15, facecolor='g')
    plt.xticks(ticks[0], np.asarray(keys)[np.asarray(indices)])
    ax.set_title(title)
    plt.xlabel('Letter')
    plt.ylabel('Keypoints Count')
    
    #plt.show()
    
    
def show_image(image_path = '/datagrid/personal/TextSpotter/evaluation-sets/icdar2013-Train/196.jpg'):
      
    img = cv2.imread(image_path, 0)
    imgc = cv2.imread(image_path)
    
    scaleFactor = 2.0
    nleves = -1
    edgeThreshold = 12
    keypointTypes = 3
    kMin = 9
    kMax = 15
    charClsModelFile = '/home/busta/workspace/TextSpotter/BoostCharacterDetector.xml'
    
    ft.init('/home/busta/git/SmsReader/charclassifier.xml', 1, scaleFactor, nleves, edgeThreshold, keypointTypes, kMin, kMax, 2, charClsModelFile)
    
    inputDir = '/datagrid/personal/TextSpotter/evaluation-sets/icdar2013-Train'
    segmDir = '{0}/segmentations'.format(inputDir)
    
    baseName = os.path.basename(image_path)
    baseName = baseName[:-4]
    
    segmGt = '{0}/{1}_GT.txt'.format(segmDir, baseName)
    segmentations = ft.getCharSegmentations(img, outputDir, baseName)
    for i in range(segmentations.shape[0]):
            rectn = segmentations[i, :]
            rectn[2] += rectn[0]
            rectn[3] += rectn[1] 
    
    if os.path.exists(segmGt):
        gt_rects = utls.read_icdar2013_segm_gt(segmGt)
            
        rcurrent = 0
        rden = 0
        for gt_rect in gt_rects:
            best_match = 0 
            for detId in range(segmentations.shape[0]):
                rectn = segmentations[detId, :]
                rect_int =  utils.intersect( rectn, gt_rect )
                int_area = utils.area(rect_int)
                union_area = utils.area(utils.union(rectn, gt_rect))
            
                ratio = int_area / float(union_area)
            
                if ratio > best_match:
                    best_match = ratio
                    
            thickness = 1
            color = (0, 0, 255)
            if best_match > 0.5:
                color = (0, 255, 0)
            if best_match > 0.7:
                thickness = 2
            cv2.rectangle(imgc, (gt_rect[0], gt_rect[1]), (gt_rect[2], gt_rect[3]), color, thickness)
                    
            rcurrent += best_match   
            rden += 1
            
        pden = 0
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
            
            pcurrent += best_match   
            pden += 1
    
    ax0 = plt.imshow(img, cmap=pylab.gray())
    keypoints = ft.findKeyPoints(img, scaleFactor, nleves, edgeThreshold, keypointTypes, kMin, kMax)
    tz = plt.plot(keypoints[:, 0], keypoints[:, 1], 'ro')
    plt.show(block=False)
    if imgc.shape[0] > 1024:
        shape = imgc.shape
        shapet = ( shape[0] / 2, shape[1] / 2, shape[2]) 
        dst = np.zeros(shapet, dtype=np.uint8)
        dst = cv2.resize(imgc, (0,0), fx=0.5, fy=0.5)
        imgc = dst
        
    cv2.imshow('imgc', imgc)
    cv2.waitKey(0)
    