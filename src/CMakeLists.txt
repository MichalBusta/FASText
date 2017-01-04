

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")

include_directories ("${PROJECT_SOURCE_DIR}/src")

add_library(FTreader STATIC 
    "FTPyramid.cpp"
    "FT_common.cpp" 
    "FASTex.cpp"
    "KeyPoints.cpp"    
    "segm/segmentation.cpp"
    "segm/flood_fill.cpp"
    "IOUtils.cpp"
    "TimeUtils.cpp" 
    "Segmenter.cpp"
    "CharClassifier.cpp" 
    "detectors.cpp"
    "vis/componentsVis.cpp"
    "HoughTLDetector.cpp"
    "FastTextLine.cpp"
    "FastTextLineDetector.cpp"
    "geometry.cpp"
)

add_executable(process_dir 
    "process_dir.cpp"
)

target_link_libraries (process_dir 
	FTreader
	${EXTRA_LIBS}
    ${OpenCV_LIBS}
)

if(NOT WIN32 AND NOT ANDROID)
	add_subdirectory(Python)
endif(NOT WIN32 AND NOT ANDROID)


