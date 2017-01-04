/*
 * HoughTLDetector.cpp
 *
 *  Created on: Jun 17, 2015
 *      Author: Michal Busta
 */
#include <opencv2/imgproc/imgproc.hpp>
#ifdef OPENCV_2
#	include <opencv2/contrib/contrib.hpp>
#endif

#include <map>
#include <unordered_map>
#include <mutex>

#include "HoughTLDetector.h"
#include "geometry.h"

//#define VERBOSE 1

namespace cmp
{

#define NUM_ANGLE 16

class LineAccumulator{
public:

	LineAccumulator(const cv::Mat& img, float rho, float theta)
{
		int width = img.cols;
		int height = img.rows;
		theta_sampling_step = theta;
		numangle = roundf((M_PI) / theta);
		numangle_2 = numangle / 2;
		numrho = roundf(((width + height) * 2 + 1) / rho);
		this->rho = rho;
		float rho2 = 0.5 * rho;
		float irho = 1 / rho;
		float irho2 = 1 / (2 * rho2);

		sizes[0] = numangle + 2 * voting_offset;
		sizes[1] = numrho + 2 * voting_offset;
		idx2 = sizes[1];
		acc =  cv::Mat(2, sizes, CV_32SC1, cv::Scalar(0));

		tabSin.resize(numangle + 2);
		tabCos.resize(numangle + 2);
		tabSin2.resize(numangle + 2);
		tabCos2.resize(numangle + 2);
		float ang = 0;
		for( int i = 0; i < numangle + 2; ang += theta, i++)
		{
			tabSin[i] = (float)(sin((double)ang) * irho);
			tabCos[i] = (float)(cos((double)ang) * irho);

			tabSin2[i] = (float)(sin((double)ang - M_PI_2) * irho2);
			tabCos2[i] = (float)(cos((double)ang - M_PI_2) * irho2);
		}
}

	void addRegion(cv::Point& center, int regId, std::vector<LetterCandidate>& letterCandidates)
	{
		LetterCandidate& ref = letterCandidates[regId];

		//check for duplicate
		int r = cvRound( center.x * tabCos[0] + center.y * tabSin[0] );
		r += (numrho - 1) / 2;
		int index =  r;
		for(auto& rid : regions2d[index] ){
			LetterCandidate& ref2 = letterCandidates[rid];
			if( ref2.isWord != ref.isWord )
				continue;
			cv::Rect int_box = ref.bbox & ref2.bbox;
			cv::Rect or_box = ref.bbox | ref2.bbox;
			if( int_box.area() / (float) or_box.area() > 0.7 ){
				return;
			}
		}


		if( ref.isWord){
			ref.rotatedRect.angle = fabs(ref.rotatedRect.angle);
			if( ref.rotatedRect.size.width <  ref.rotatedRect.size.height ){
				ref.rotatedRect.angle += 90;
				int swp = ref.rotatedRect.size.width;
				ref.rotatedRect.size.width = ref.rotatedRect.size.height;
				ref.rotatedRect.size.height = swp;
			}
			float theta =  ref.rotatedRect.angle / 180.0 * M_PI - M_PI_2;
			while( theta < 0)
				theta += M_PI;
			while( theta > M_PI)
				theta -= M_PI;
			//theta = M_PI_2;
			assert(theta >= 0);
			assert(theta <  M_PI);
			int n = theta / theta_sampling_step;
			assert(n >= 0);
			assert(n < numangle);
			center.x = ref.bbox.x + ref.bbox.width / 2;
			center.y = ref.bbox.y + ref.bbox.height / 2;
			int r = cvRound( center.x * tabCos[n] + center.y * tabSin[n] );
			r += (numrho - 1) / 2;
			acc.at<int>(n + voting_offset, r + voting_offset) += this->min_value;
			int index =  n * idx2 +  r;
			regions2d[index].insert(regId);

			/*
            acc.at<int>(n + 1 + voting_offset, r + voting_offset) += this->min_value;
            index =  (n  + 1)  * idx2 +  r;
            regions2d[index].insert(regId);

            acc.at<int>(n - 1 + voting_offset, r + voting_offset) += this->min_value;
            index =  (n  - 1)  * idx2 +  r;
            regions2d[index].insert(regId);
			 */
			return;
		}

		for( int n = 0; n < numangle; n++)
		{
			int r = cvRound( center.x * tabCos[n] + center.y * tabSin[n] );
			r += (numrho - 1) / 2;
			int index =  n * idx2 +  r;
			if( ref.quality > 0.3)
				acc.at<int>(n + voting_offset, r + voting_offset) += 1;
			regions2d[index].insert(regId);
		}
	}

	void findMaxima(std::vector<cv::Vec4d>& lines, std::vector<LetterCandidate>& letterCandidates)
	{
		std::unordered_multimap<int,int> reg_to_line;
		std::vector<int> lineIx;
		for (int x = voting_offset; x < acc.cols - voting_offset; x++)
		{
			double min, maxVal;
			cv::minMaxLoc(acc(cv::Rect(x, 0, 1, acc.rows - 1)), &min, &maxVal);
			for (int n = voting_offset; n < acc.rows - voting_offset; n++)
			{
				int value = acc.at<int>(n, x);
				if (value < min_value)
				{
					continue;
				}

				if(value < acc.at<int>(n, x + 1) || value < acc.at<int>(n, x - 1)){
					continue;
				}

				int sumVal = value + acc.at<int>(n, x - 1) + acc.at<int>(n, x + 1);
				if( sumVal < maxVal){
					continue;
				}

				int index =  (n - voting_offset) * idx2 +  (x - voting_offset);
				bool is_maxima = true;
				for( auto& rid: regions2d[index] ){
					if(!is_maxima)
						break;
					LetterCandidate& ref = letterCandidates[rid];
					cv::Point center = ref.getConvexCentroid();

					for( int n2 = 0; n2 < numangle; n2++){
						int r = cvRound( center.x * tabCos[n2] + center.y * tabSin[n2] );
						r += (numrho - 1) / 2;
						if( acc.at<int>(n2 + voting_offset , r + voting_offset) > sumVal){
							is_maxima = false;
							break;
						}
					}
				}

				if(!is_maxima){
					continue;
				}


				//if( (n - voting_offset) != numangle / 2)
					//    continue;

				//double line_rho23 = ((x - 1) - (numrho - 1)*0.5f) * rho;
				std::multimap<int, std::pair<float, int> > line_rho2;
				for( auto& rid: regions2d[index] ){
					LetterCandidate& ref = letterCandidates[rid];
					cv::Point center = ref.bbox.tl();
					float r201 = center.x * tabCos2[n - voting_offset] + center.y * tabSin2[n - voting_offset];
					center = ref.bbox.br();
					int r202 = center.x * tabCos2[n - voting_offset] + center.y * tabSin2[n - voting_offset];
					line_rho2.insert( {MIN(r201, r202), std::pair<int, int>(MAX(r201, r202), rid ) });

				}
				if( line_rho2.size() == 1 ){
					continue;
				}

				std::vector<float> spacing;
				spacing.reserve(line_rho2.size());
				std::map<int, std::pair<float, int>>::iterator itp = line_rho2.begin();
				std::map<int, std::pair<float, int>>::iterator itn = itp;
				itn++;
				do{
					spacing.push_back(MAX(0.0f, itn->first - itp->second.first));
					if( itp->second.first > itn->second.first ){
						std::map<int, std::pair<float, int>>::iterator itc = itn;
						while( itc->second.first < itp->second.first){
							itc->second.first = itp->second.first;
						}

					}
					itp++;
					itn++;
				}while(itn != line_rho2.end());

				if( spacing.size() == 0 )
					continue;

				int r = (x - voting_offset);
				double line_rho = (r - (numrho - 1)*0.5f) * rho;
				int lineId = lines.size();
				lines.push_back(cv::Vec4d(line_rho, (n - voting_offset) * theta_sampling_step, value, lineId));
				itp = line_rho2.begin();
				itn = itp;
				itn++;
				int s = 0;
				while( itn != line_rho2.end()  ){
					regionsMap[lineId].insert(itp->second.second);
					if( ((spacing[s] > 3.8f ) ) ) {
						if( regionsMap[lineId].size() >= 2 ) {
							double maxQuality = 0;
							for(auto& rid : regionsMap[lineId] )
								maxQuality = MAX(maxQuality, letterCandidates[rid].quality);
							if( maxQuality < 0.5 ){
								regionsMap[lineId].clear();
							}else{
								lines.back().val[2] = regionsMap[lineId].size();
								lineId = lines.size();
								lines.push_back(cv::Vec4d(line_rho, (n - voting_offset) * theta_sampling_step, value, lineId));
							}
						}else{
							regionsMap[lineId].clear();
						}
					}
					itp++;
					itn++;
					s++;
				}
				if( regionsMap[lineId].size() > 0 ) {
					double maxQuality = 0;
					for(auto& rid : regionsMap[lineId] )
						maxQuality = MAX(maxQuality, letterCandidates[rid].quality);
					if( maxQuality < 0.5 ){
						regionsMap[lineId].clear();
					}
					regionsMap[lineId].insert(itp->second.second);
					lines.back().val[2] = regionsMap[lineId].size();
				}
			}
		}
	}

	cv::Mat acc;
	std::vector<float> tabSin;
	std::vector<float> tabCos;
	std::vector<float> tabSin2;
	std::vector<float> tabCos2;

	int numrho;
	float rho;
	float theta_sampling_step;
	int numangle;
	int numangle_2;

	int sizes[3];
	int idx2;

	int min_value = 3;

	int voting_offset = 2;

	std::unordered_map<int, std::set<int> > regionsMap;
	std::unordered_map<int, std::set<int> > regions2d;
};


inline void drawLine(cv::Mat& cdst, double rho, double theta, cv::Scalar color)
{
	cv::Point pt1, pt2;
	double a = cos(theta), b = sin(theta);
	double x0 = a*rho, y0 = b*rho;
	pt1.x = cvRound(x0 + 2000 * (-b));
	pt1.y = cvRound(y0 + 2000 * (a));
	pt2.x = cvRound(x0 - 2000 * (-b));
	pt2.y = cvRound(y0 - 2000 * (a));
	line(cdst, pt1, pt2, color, 2, CV_AA);
}

void HoughTLDetector::findTextLines(std::vector<LetterCandidate>& letterCandidates, const cv::Mat& originalImage, double letterHeight,  std::vector<LineGroup>& lineGroups, int type) {

#ifdef VERBOSE
	double t_g = (double) cv::getTickCount();
	double maxRho = round(sqrt(originalImage.rows * originalImage.rows + originalImage.cols * originalImage.cols));
#endif

	///LineAccumulator acc(originalImage.rows, originalImage.cols, letterHeight / 2);
	LineAccumulator acc(originalImage, letterHeight / 2, M_PI / 16);

#ifdef VERBOSE
	std::cout << "Letter Height: " << letterHeight << " - " << letterHeight / 2 <<   std::endl;
	cv::Mat cdst = originalImage.clone();
	cv::cvtColor(cdst, cdst, cv::COLOR_GRAY2BGR);

	double mr =  (int)maxRho;
	//int rhoIx = (int)round(rho / rhoSamplingStep) + _rhoOffset;
	for(double i = 0; i < mr; i += acc.rho){
		drawLine(cdst, i, M_PI_2, cv::Scalar(128, 128, 128));
	}

	for (size_t i = 0; i < letterCandidates.size(); i++)
	{
		if( letterCandidates[i].keyPoint.type != type || letterCandidates[i].duplicate != -1)
			continue;
		double size = letterCandidates[i].bbox.height;
		if( letterCandidates[i].isWord )
			size = MIN(letterCandidates[i].bbox.height, letterCandidates[i].bbox.width);
		letterCandidates[i].angleScore = 0;
		double hr = MIN(size, letterHeight) / MAX(size, letterHeight);
		if( hr < 0.5){
			continue;
		}
		if(letterCandidates[i].isWord)
			cv::rectangle(cdst, letterCandidates[i].bbox, cv::Scalar(0, 255, 0));
		else
			cv::rectangle(cdst, letterCandidates[i].bbox, cv::Scalar(0, 0, 255));
	}

	cv::imshow("voting step", cdst);
	cv::waitKey(0);

#endif

	for (size_t i = 0; i < letterCandidates.size(); i++)
	{
		if( letterCandidates[i].keyPoint.type != type || letterCandidates[i].duplicate != -1){
			continue;
		}
		double size = (letterCandidates[i].bbox.height);
		if( letterCandidates[i].isWord )
			size = MIN(letterCandidates[i].bbox.height, letterCandidates[i].bbox.width);
		double hr = MIN(size, letterHeight) / MAX(size, letterHeight);
		if( hr < 0.5){
			continue;
		}

		cv::Point center = letterCandidates[i].getConvexCentroid();
		acc.addRegion(center, i, letterCandidates);
	}


	std::vector<cv::Vec4d> initialLines;
	acc.min_value = 6;
	acc.findMaxima(initialLines, letterCandidates);

#ifdef VERBOSE
	std::cout << "Hough maxima in " << (cv::getTickCount() - t_g) / (cv::getTickFrequency()) * 1000 << ", " << initialLines.size() <<  "\n";
	sort(initialLines.begin(), initialLines.end(),
			[&](const cv::Vec4d & a, const cv::Vec4d & b) -> bool {
		return a.val[2] > b.val[2];
	});
#endif

	for (size_t i = 0; i < initialLines.size(); i++)
	{
		if( acc.regionsMap[initialLines[i].val[3]].size() < 1)
			continue;
		double rho = initialLines[i][0], theta = initialLines[i][1];
#ifdef VERBOSE



		cv::Mat cdst = originalImage.clone();
		cv::cvtColor(cdst, cdst, cv::COLOR_GRAY2BGR);
		drawLine(cdst, rho, theta, cv::Scalar(128, 128, 128));
#endif
		lineGroups.push_back(LineGroup( initialLines[i].val[2], rho, theta, 1 ));
		lineGroups.back().regionIds = acc.regionsMap[initialLines[i].val[3]];

#ifdef VERBOSE
		for( auto rid :  lineGroups.back().regionIds){
			assert(rid < (int) letterCandidates.size());
			if( letterCandidates[rid].isWord ){
				cv::rectangle(cdst, letterCandidates[rid].bbox, cv::Scalar(0, 255, 0));
			}
			else
				cv::rectangle(cdst, letterCandidates[rid].bbox, cv::Scalar(0, 0, 255));
		}
		std::cout << "Group: " << initialLines[i].val[0] << '/' << initialLines[i].val[1] * 180 / M_PI << "/" << initialLines[i].val[2] << "/" << lineGroups.back().regionIds.size() << std::endl;
		cv::imshow("ts", cdst);
		cv::waitKey(0);
#endif
	}
	//std::cout << "Hough To groups " << (cv::getTickCount() - t_g) / (cv::getTickFrequency()) * 1000 << "\n";
	//std::cout << "Tuples groups " << (cv::getTickCount() - t_g) / (cv::getTickFrequency()) * 1000 << "\n";
}

} /* namespace cmp */

