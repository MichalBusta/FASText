/*
 * TimeUtils.cpp
 *
 *  Created on: Nov 20, 2014
 *      Author: Michal.Busta at gmail.com
 *
 */
#include "TimeUtils.h"

#include <opencv2/core/core.hpp>

#ifdef _WIN32
#include <windows.h>
#include <winbase.h>
#endif

namespace cmp
{

long long TimeUtils::MiliseconsNow()
{
#ifdef _WIN32
	static LARGE_INTEGER s_frequency;
		static BOOL s_use_qpc = QueryPerformanceFrequency(&s_frequency);
		if (s_use_qpc) {
			LARGE_INTEGER now;
			QueryPerformanceCounter(&now);
			return (1000LL * now.QuadPart) / s_frequency.QuadPart;
		}
		else {
			return GetTickCount();
		}
#else
	return cv::getTickCount() / (cv::getTickFrequency()) * 1000;
#endif
}

#ifdef _WIN32
#include <Windows.h>
double get_wall_time(){
    LARGE_INTEGER time,freq;
    if (!QueryPerformanceFrequency(&freq)){
        //  Handle error
        return 0;
    }
    if (!QueryPerformanceCounter(&time)){
        //  Handle error
        return 0;
    }
    return (double)time.QuadPart / freq.QuadPart;
}
double get_cpu_time(){
    FILETIME a,b,c,d;
    if (GetProcessTimes(GetCurrentProcess(),&a,&b,&c,&d) != 0){
        //  Returns total user time.
        //  Can be tweaked to include kernel times as well.
        return
            (double)(d.dwLowDateTime |
            ((unsigned long long)d.dwHighDateTime << 32)) * 0.0001;
    }else{
        //  Handle error
        return 0;
    }
}

//  Posix/Linux
#else
#include <sys/time.h>
double get_wall_time(){
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        //  Handle error
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .001;
}
double get_cpu_time(){
    return (double)clock() / CLOCKS_PER_SEC;
}
#endif

} /* namespace cmp */
