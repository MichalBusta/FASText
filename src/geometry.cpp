/*
 * geometry.cpp
 *
 *  Created on: Feb 11, 2015
 *      Author: Michal Busta
 */

#include <set>
#include <opencv2/imgproc/imgproc.hpp>

#include "geometry.h"

namespace cmp
{

double angleDiff(double a,double b){
    double dif = fmod(b - a + M_PI , 2*M_PI);
    if (dif < 0)
        dif += 2*M_PI;
    return dif - M_PI;
}

double distance_to_line( cv::Point begin, cv::Point end, cv::Point x, int& sign )
{
   //translate the begin to the origin
   end -= begin;
   x -= begin;

   //¿do you see the triangle?
   double area = x.cross(end);
   sign = (area > 0) - (area < 0);
   return fabs(area / cv::norm(end));
}

double distance_to_line( const cv::Vec4f& line, cv::Point x, int& sign )
{
   //translate the begin to the origin
   cv::Point2f end = cv::Point2f(line.val[0], line.val[1]);
   cv::Point xf = x - cv::Point(line.val[2], line.val[3]);

   //¿do you see the triangle?
   double area = xf.cross(end);
   sign = (area > 0) - (area < 0);
   return fabs(area / cv::norm(end));
}


double innerAngle(cv::Vec4i& line1, cv::Vec4i& line2, bool invert)
{
	cv::Point p11(line1.val[0],line1.val[1]);
	cv::Point p12(line1.val[2], line1.val[3]);
	cv::Point p21(line2.val[0], line2.val[1]);
	cv::Point p22(line2.val[2], line2.val[3]);

	cv::Point v1 = p12 - p11;
	cv::Point v2 = p21 - p22;
	if(invert)
	{
		v1 = p11 - p12;
		v2 = p21 - p22;
	}

	double dot = v1.dot(v2);
	double cross = v1.cross(v2);
	double angle = atan2(cross, dot);
	return angle;
}

bool isLeft(cv::Point a, cv::Point b, cv::Point c)
{
     return ((b.x - a.x)*(c.y - a.y) - (b.y - a.y)*(c.x - a.x)) > 0;
}

bool isBetween(cv::Vec4f bottomLine, cv::Vec4f line2, cv::Point point, double& mindist )
{

	cv::Point a(bottomLine[2], bottomLine[3]);
	cv::Point b(bottomLine[2] + 100 * bottomLine[0] , bottomLine[3] + 100 * bottomLine[1]);
	cv::Point aa(line2[2], line2[3]);
	cv::Point bb(line2[2] + 100 * bottomLine[0] , line2[3] + 100 * bottomLine[1]);
	int s1 = 0;
	double d1 = distance_to_line( a, b, aa, s1 );
	int s2 = 0;
	double d2 = distance_to_line( a, b, point, s2 );
	int s3 = 0;
	double d3 = distance_to_line( aa, bb, point, s3 );
	mindist = MIN(d2, d3);
	if(d2 > d1)
		return false;
	if(d3 > d1)
		return false;
	return true;
}

double innerAngle(const cv::Point& line1, const cv::Point& line2)
{
	double dot = line1.x*line2.x + line1.y*line2.y;//dot product
	double det = line1.x*line2.y -  line1.y*line2.x; //determinant
	return M_PI - atan2(det, dot); //  # atan2(y, x) or atan2(sin, cos)
}

int rotatedRectangleIntersection( const cv::RotatedRect& rect1, const cv::RotatedRect& rect2, cv::OutputArray intersectingRegion )
{
    const float samePointEps = 0.00001f; // used to test if two points are the same

    cv::Point2f vec1[4], vec2[4];
    cv::Point2f pts1[4], pts2[4];

    std::vector <cv::Point2f> intersection;

    rect1.points(pts1);
    rect2.points(pts2);

    int ret = INTERSECT_FULL;

    // Specical case of rect1 == rect2
    {
        bool same = true;

        for( int i = 0; i < 4; i++ )
        {
            if( fabs(pts1[i].x - pts2[i].x) > samePointEps || (fabs(pts1[i].y - pts2[i].y) > samePointEps) )
            {
                same = false;
                break;
            }
        }

        if(same)
        {
            intersection.resize(4);

            for( int i = 0; i < 4; i++ )
            {
                intersection[i] = pts1[i];
            }

            cv::Mat(intersection).copyTo(intersectingRegion);

            return INTERSECT_FULL;
        }
    }

    // Line vector
    // A line from p1 to p2 is: p1 + (p2-p1)*t, t=[0,1]
    for( int i = 0; i < 4; i++ )
    {
        vec1[i].x = pts1[(i+1)%4].x - pts1[i].x;
        vec1[i].y = pts1[(i+1)%4].y - pts1[i].y;

        vec2[i].x = pts2[(i+1)%4].x - pts2[i].x;
        vec2[i].y = pts2[(i+1)%4].y - pts2[i].y;
    }

    // Line test - test all line combos for intersection
    for( int i = 0; i < 4; i++ )
    {
        for( int j = 0; j < 4; j++ )
        {
            // Solve for 2x2 Ax=b
            float x21 = pts2[j].x - pts1[i].x;
            float y21 = pts2[j].y - pts1[i].y;

            float vx1 = vec1[i].x;
            float vy1 = vec1[i].y;

            float vx2 = vec2[j].x;
            float vy2 = vec2[j].y;

            float det = vx2*vy1 - vx1*vy2;

            float t1 = (vx2*y21 - vy2*x21) / det;
            float t2 = (vx1*y21 - vy1*x21) / det;

            // This takes care of parallel lines
            if( cvIsInf(t1) || cvIsInf(t2) || cvIsNaN(t1) || cvIsNaN(t2) )
            {
                continue;
            }

            if( t1 >= 0.0f && t1 <= 1.0f && t2 >= 0.0f && t2 <= 1.0f )
            {
                float xi = pts1[i].x + vec1[i].x*t1;
                float yi = pts1[i].y + vec1[i].y*t1;

                intersection.push_back(cv::Point2f(xi,yi));
            }
        }
    }

    if( !intersection.empty() )
    {
        ret = INTERSECT_PARTIAL;
    }

    // Check for vertices from rect1 inside recct2
    for( int i = 0; i < 4; i++ )
    {
        // We do a sign test to see which side the point lies.
        // If the point all lie on the same sign for all 4 sides of the rect,
        // then there's an intersection
        int posSign = 0;
        int negSign = 0;

        float x = pts1[i].x;
        float y = pts1[i].y;

        for( int j = 0; j < 4; j++ )
        {
            // line equation: Ax + By + C = 0
            // see which side of the line this point is at
            float A = -vec2[j].y;
            float B = vec2[j].x;
            float C = -(A*pts2[j].x + B*pts2[j].y);

            float s = A*x+ B*y+ C;

            if( s >= 0 )
            {
                posSign++;
            }
            else
            {
                negSign++;
            }
        }

        if( posSign == 4 || negSign == 4 )
        {
            intersection.push_back(pts1[i]);
        }
    }

    // Reverse the check - check for vertices from rect2 inside recct1
    for( int i = 0; i < 4; i++ )
    {
        // We do a sign test to see which side the point lies.
        // If the point all lie on the same sign for all 4 sides of the rect,
        // then there's an intersection
        int posSign = 0;
        int negSign = 0;

        float x = pts2[i].x;
        float y = pts2[i].y;

        for( int j = 0; j < 4; j++ )
        {
            // line equation: Ax + By + C = 0
            // see which side of the line this point is at
            float A = -vec1[j].y;
            float B = vec1[j].x;
            float C = -(A*pts1[j].x + B*pts1[j].y);

            float s = A*x + B*y + C;

            if( s >= 0 )
            {
                posSign++;
            }
            else
            {
                negSign++;
            }
        }

        if( posSign == 4 || negSign == 4 )
        {
            intersection.push_back(pts2[i]);
        }
    }

    // Get rid of dupes
    for( int i = 0; i < (int)intersection.size()-1; i++ )
    {
        for( size_t j = i+1; j < intersection.size(); j++ )
        {
            float dx = intersection[i].x - intersection[j].x;
            float dy = intersection[i].y - intersection[j].y;
            double d2 = dx*dx + dy*dy; // can be a really small number, need double here

            if( d2 < samePointEps*samePointEps )
            {
                // Found a dupe, remove it
                std::swap(intersection[j], intersection.back());
                intersection.pop_back();
                j--; // restart check
            }
        }
    }

    if( intersection.empty() )
    {
        return INTERSECT_NONE ;
    }

    // If this check fails then it means we're getting dupes, increase samePointEps
    //CV_Assert( intersection.size() <= 8 );

    cv::Mat(intersection).copyTo(intersectingRegion);

    return ret;
}

void getConvexHullLines(std::vector<cv::Point>& cHullPoints1, std::vector<cv::Point>& cHullPoints2, const cv::Mat& img, std::vector<cv::Vec4i>& convexLines, std::vector<cv::Point>& chull, double& dist)
{
	std::vector<cv::Point> allHullPoins;

	allHullPoins.reserve(cHullPoints1.size() + cHullPoints2.size());
	std::set<long> index1;
	for(std::vector<cv::Point>::iterator it = cHullPoints1.begin(); it < cHullPoints1.end(); it++ )
	{
		allHullPoins.push_back( *it );
		index1.insert( it->x +  it->y * img.cols );
	}
	std::set<long> index2;
	for(std::vector<cv::Point>::iterator it = cHullPoints2.begin(); it < cHullPoints2.end(); it++ )
	{
		allHullPoins.push_back(*it );
		index2.insert( it->x +  it->y * img.cols );
	}

	std::vector<int> hull;
	convexHull(allHullPoins, hull, false, false);
	if(hull.size() <= 3)
		return;

	chull.resize(hull.size());
	for(size_t i = 0; i < hull.size(); i++)
		chull[i] = allHullPoins[hull[i]];


	std::vector<cv::Vec4i> defects;
	//convexityDefects(allHullPoins, hull, defects);

	for( size_t i = 1; i < chull.size() + 1; i++ )
	{
		cv::Point p1 = chull[i - 1];
		size_t index = i;
		if( index >= chull.size())
			index = 0;
		cv::Point p2 = chull[index];
		if( index1.find( p1.x +  p1.y * img.cols ) != index1.end() && index1.find( p2.x +  p2.y * img.cols ) != index1.end())
			continue;
		if( index2.find( p1.x +  p1.y * img.cols ) != index2.end() && index2.find( p2.x +  p2.y * img.cols ) != index2.end())
			continue;
		defects.push_back(cv::Vec4i(i - 1, index, 0, 0));
	}

	std::vector<cv::Vec4i> defectsCross;

	for( size_t i = 0; i < defects.size(); i++ )
	{
		cv::Point p1 = chull[defects[i].val[0]];
		cv::Point p2 = chull[defects[i].val[1]];
		if( index1.find( p1.x +  p1.y * img.cols ) != index1.end() && index1.find( p2.x +  p2.y * img.cols ) != index1.end())
			continue;
		if( index2.find( p1.x +  p1.y * img.cols ) != index2.end() && index2.find( p2.x +  p2.y * img.cols ) != index2.end())
			continue;
		defectsCross.push_back(defects[i]);
		if(index1.find( p1.x +  p1.y * img.cols ) != index1.end())
		{
			convexLines.push_back( cv::Vec4i(chull[defects[i].val[0]].x, chull[defects[i].val[0]].y, chull[defects[i].val[1]].x, chull[defects[i].val[1]].y) );
		}else
		{
			convexLines.push_back( cv::Vec4i(chull[defects[i].val[1]].x, chull[defects[i].val[1]].y, chull[defects[i].val[0]].x, chull[defects[i].val[0]].y) );
		}
	}
	if(defectsCross.size() == 2)
	{
		dist = INT_MAX;
		for( size_t i = 0; i < cHullPoints1.size(); i++)
		{
			for( size_t j = 0; j < cHullPoints2.size(); j++)
			{
				cv::Point d =  cHullPoints1[i] - cHullPoints2[j];
				double distc = d.x * d.x + d.y * d.y;
				if( distc < dist )
					dist = distc;
			}
		}
	}
}

} /* namespace cmp */
