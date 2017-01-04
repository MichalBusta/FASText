/*
 * geometry.h
 *
 *  Created on: Feb 11, 2015
 *      Author: Michal Busta
 */
#ifndef GEOMETRY_H_
#define GEOMETRY_H_

#include <opencv2/core/core.hpp>
#include <limits>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace cmp
{

double angleDiff(double a, double b);

double distance_to_line( cv::Point begin, cv::Point end, cv::Point x, int& sign );

double distance_to_line( const cv::Vec4f& line, cv::Point x, int& sign );

double innerAngle(cv::Vec4i& line1, cv::Vec4i& line2, bool invert = false);

double innerAngle(const cv::Point& line1, const cv::Point& line2);

bool isBetween(cv::Vec4f bottomLine, cv::Vec4f line2, cv::Point point, double& mindist );

enum RectanglesIntersectTypes {
    INTERSECT_NONE = 0, //!< No intersection
    INTERSECT_PARTIAL  = 1, //!< There is a partial intersection
    INTERSECT_FULL  = 2 //!< One of the rectangle is fully enclosed in the other
};

int rotatedRectangleIntersection( const cv::RotatedRect& rect1, const cv::RotatedRect& rect2, cv::OutputArray intersectingRegion );

void getConvexHullLines(std::vector<cv::Point>& cHullPoints1, std::vector<cv::Point>& cHullPoints2, const cv::Mat& img, std::vector<cv::Vec4i>& convexLines, std::vector<cv::Point>& chull, double& dist);

/**
 * @param img
 * @return The bounding box of non-zero image pixels
 */
inline cv::Rect getNonZeroBBox(const cv::Mat& img, int thresh = 0)
{
	int minX = std::numeric_limits<int>::max();
	int maxX = 0;
	int minY = std::numeric_limits<int>::max();
	int maxY = 0;

	for (int y=0; y<img.rows; y++)
	{
		const uchar* pRow = img.ptr(y);
		for (int x=0; x < img.cols; x++)
		{
			if (*(pRow++) > thresh)
			{
				minX = MIN(minX, x);
				maxX = MAX(maxX, x);
				minY = MIN(minY, y);
				maxY = MAX(maxY, y);
			}

		}
	}
	if(minX == std::numeric_limits<int>::max())
		minX = 0;
	if(minY == std::numeric_limits<int>::max())
		minY = 0;
	return cv::Rect( minX, minY, maxX - minX + 1, maxY - minY + 1);
}

} /* namespace cmp */

#endif /* GEOMETRY_H_ */
