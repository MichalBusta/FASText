'''
Created on Nov 25, 2014

@author: busta
'''

def intersect(a, b):
    '''Determine the intersection of two rectangles'''
    rect = (0,0,0,0)
    r0 = max(a[0],b[0])
    c0 = max(a[1],b[1])
    r1 = min(a[2],b[2])
    c1 = min(a[3],b[3])
    # Do we have a valid intersection?
    if r1 > r0 and  c1 > c0: 
        rect = (r0,c0,r1,c1)
    return rect

def union(a, b):
    r0 = min(a[0],b[0])
    c0 = min(a[1],b[1])
    r1 = max(a[2],b[2])
    c1 = max(a[3],b[3])
    return (r0,c0,r1,c1)

def area(a):
    '''Computes rectangle area'''
    width = a[2] - a[0]
    height = a[3] - a[1]
    return width * height