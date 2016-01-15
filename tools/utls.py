'''
Created on Oct 2, 2013

@author: busta
'''

import math
import numpy as np

from xml.dom import minidom
from csv import reader

def intersect(a, b):
    ''' returns the intersection of two lists '''
    
    ind_dict = dict((k,i) for i,k in enumerate(a))
    inter = set( ind_dict.keys() ).intersection(b)
    
    indices = [ ind_dict[x] for x in inter ]
    
    return (inter, indices)

def compupte_area(geo):
    ''' Computes area from the feature geometry '''
    a = geo[3]
    b = geo[4]
    # Calculating the Area of the Visual Words detected in the image
    area =  math.pi * a * b
    return area

def get_area_scale_index(area):
    ''' 
    The Scale of feature is defined as sqrt(area) and its values ranges from
    8 to 67. All values < 9 are taken as 8 and > 67 are taken as 67 
    '''
    
    scale = np.around( np.sqrt( area ), 0).astype(np.int)
    scale = np.clip(scale, 8, 67)
    scale = scale - 8;
    return scale

def get_y_relative_index(featurey, yline, line_height):
    ''' returns the relative position of feature against the line'''
    rel_position = np.round(10*(featurey - yline) / line_height).astype(np.int)
    rel_position = np.clip(rel_position, -20, 20)                                                        
    rel_position = rel_position + 20
    return rel_position

def y_index_2relative(index, height):
    index = index - 20
    return index * height / 10.0
    

def reproject_feature(y0, height, relposition):
    
    rel_pos = y_index_2relative(relposition, height)
    pos_low = np.round( y0 - rel_pos )
    pos_up = np.round( y0 + height - rel_pos )
    
    return (pos_low, pos_up)
    
def get_inbox_mask(bbox, geo, epsilon = 0):
    ''' return the mask of features inside the rotated rectangle '''
    slope = (bbox[7] - bbox[5]) / float( bbox[6] - bbox[4])
    geoyT = geo[1, :] + ( bbox[6] - geo[0, :] ) * slope
    
    maskX = np.logical_and( geo[0, :] >= (bbox[0] - epsilon), geo[0, :] <= (bbox[2] + epsilon) )
    maskY = np.logical_and( geoyT >= (bbox[7] - epsilon), geoyT <= (bbox[9] + epsilon) )
    
    return np.logical_and(maskX, maskY)     


def textContent(node):
    ''' Returns text content of xml node '''
    
    if node.nodeType in (node.TEXT_NODE, node.CDATA_SECTION_NODE):
        return node.nodeValue
    else:
        return ''.join(textContent(n) for n in node.childNodes)

def read_icdar2011_gt(gt_file):
    
    doc = minidom.parse(gt_file)
    gtimages = {}
    noOfGtImages = 0
    noGt = 0
    for node in doc.getElementsByTagName('image'):
        
        image_name = textContent(node.getElementsByTagName('imageName')[0])  
        gtimages[image_name] = []
        noOfGtImages += 1
        
        for rect in node.getElementsByTagName('taggedRectangle'):
            
            x = float(rect.getAttributeNode('x').nodeValue)
            y = float(rect.getAttributeNode('y').nodeValue)
            width = float(rect.getAttributeNode('width').nodeValue)
            height = float(rect.getAttributeNode('height').nodeValue)
            
            gtimages[image_name].append( [x, y, x + width, y + height] )
            noGt += 1
            
    return (gtimages, noOfGtImages, noGt)

def read_icdar2013_segm_gt(gt_file, separator = ' '):
    
    f = open( gt_file, "r")
    lines = f.readlines()
    gt_rectangles = []
    groups = []
    group = []
    objId = 0
    for line in lines:
        if line[0] == '#':
            continue
        splitLine = line.split(separator);
        if len(splitLine) < 5:
            if len(group) > 0:
                groups.append(group)
                group = []
            continue
        
        xline = '{0}'.format(line.strip()) 
        
        for splitLine in reader([xline], skipinitialspace=True, quotechar='"', delimiter=separator):
            break
        
        minX = min(int(float(splitLine[5])), int(float(splitLine[7])))
        maxX = max(int(float(splitLine[5])), int(float(splitLine[7])))    
        minY = min(int(float(splitLine[6])), int(float(splitLine[8])))
        maxY = max(int(float(splitLine[6])), int(float(splitLine[8])))
        
        gt_rectangles.append([minX, minY, maxX, maxY, splitLine[9], 0, 0] )
        group.append(objId)
        objId += 1
            
    
    return (gt_rectangles, groups)

def read_icdar2013_txt_gt(gt_file, separator = ' '):
    
    f = open( gt_file, "r")
    lines = f.readlines()
    gt_rectangles = []
    for line in lines:
        if line[0] == '#':
            continue
        splitLine = line.split(separator);
        if len(splitLine) < 5:
            continue
        xline = '{0}'.format(line.strip()) 
        
        for splitLine in reader([xline], skipinitialspace=True, quotechar='"', delimiter=separator):
            break
        
        
        minX = min(int(float(splitLine[0].strip())), int(float(splitLine[2].strip())))
        maxX = max(int(float(splitLine[0].strip())), int(float(splitLine[2].strip())))    
        minY = min(int(float(splitLine[1].strip())), int(float(splitLine[3].strip())))
        maxY = max(int(float(splitLine[1].strip())), int(float(splitLine[3].strip())))    
        
        gt_rectangles.append( (minX, minY, maxX, maxY, splitLine[4]) )
            
    
    return gt_rectangles      
    
def read_mock_segm_gt(gt_file, separator = ' '):
    
    f = open( gt_file, "r")
    lines = f.readlines()
    gt_rectangles = []
    groups = []
    group = []
    stage = 0
    for line in lines:
        if line[0] == '#':
            continue
        
        if line.startswith('RECTANGLES:'):
            stage = 1
        elif line.startswith('LINES:'):
            stage = 2
        elif stage == 1: 
            
            splitLine = line.split(separator);
            xline = '{0}'.format(line.strip()) 
            for splitLine in reader([xline], skipinitialspace=True, quotechar='"', delimiter=separator):
                break
        
            x = float(splitLine[1])
            y = float(splitLine[2])
            width = float(splitLine[3])
            height = float(splitLine[4])
            
            if len( splitLine) > 6:
                gt_rectangles.append( (x, y, x + width, y + height, splitLine[6]) )
            else:
                gt_rectangles.append( (x, y, x + width, y + height, " ") )
        elif stage == 2:
            splitLine = line.split(separator);
            xline = '{0}'.format(line.strip()) 
            for splitLine in reader([xline], skipinitialspace=True, quotechar='"', delimiter=separator):
                break
            
            group = []
            for i in range(len(splitLine)):
                group.append(int(splitLine[i]))
            groups.append(group)
        
            
    
    return (gt_rectangles, groups)
            
    