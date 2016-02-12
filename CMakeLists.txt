cmake_minimum_required(VERSION 2.6)
project(FastText)

SET(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
set(EXECUTABLE_OUTPUT_PATH      "${CMAKE_BINARY_DIR}/bin") 

option (BUILD_WITH_DEBUG_INFO   "Build with debugging information" ON)
option(BUILD_SHARED_LIBS        "Build shared libraries (.dll/.so) instead of static ones (.lib/.a)" OFF )
option(BUILD_PARALLEL           "With OpenMP" On )

set(MODULES_DIR "${PROJECT_SOURCE_DIR}")

if (MSVC)
    add_definitions( -DNOMINMAX )
    add_definitions( -D_VARIADIC_MAX=10 )
endif()

#-------------------Extra libraries -----------------------------------
#extra flags for OpenMP for flann
if(NOT WIN32)
    if(BUILD_PARALLEL)
    find_package(OpenMP)
    if(OPENMP_FOUND)
        if(NOT WIN32)
            set(EXTRA_PERF_LIBS gomp)
            set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
            set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
            add_definitions( -DPARALLEL ) 
        endif(NOT WIN32)
    endif(OPENMP_FOUND)
    else(BUILD_PARALLEL)
    message( STATUS "ANDROID BUILD")
    endif(BUILD_PARALLEL)
endif(NOT WIN32)
#-------------------Extra libraries -----------------------------------

if(ANDROID)
    add_definitions("-std=gnu++0x")
    add_definitions("-DANDROID_LOG")
    SET(CMAKE_CXX_FLAGS_RELEASE  "${CMAKE_CXX_FLAGS_RELEASE} -ftree-vectorize -ffast-math")
else(ANDROID)
    #add support for new c++ standard
    if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
        set(CMAKE_CXX_FLAGS "-std=c++11")
    endif()

    if(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)
        add_definitions("-std=c++11")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-long-long -Wno-reorder -ftree-vectorize")
    endif(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)
endif(ANDROID)

# ----------------------------------------------------------------------------
# Output directories 
# ----------------------------------------------------------------------------
if(ANDROID)
    set(LIBRARY_OUTPUT_PATH         "${CMAKE_BINARY_DIR}/lib/${ANDROID_NDK_ABI_NAME}")
    set(EXECUTABLE_OUTPUT_PATH      "${CMAKE_BINARY_DIR}/bin")
    
else()
    set(LIBRARY_OUTPUT_PATH         "${CMAKE_BINARY_DIR}/lib")
    set(EXECUTABLE_OUTPUT_PATH      "${CMAKE_BINARY_DIR}") 
    set(3P_LIBRARY_OUTPUT_PATH      "${CMAKE_BINARY_DIR}/3rdparty/lib${LIB_SUFFIX}")
endif()

find_package(OpenCV)
if(OpenCV_DIR)
    message( STATUS "Using OpenCV from ${OpenCV_DIR}, include dirs: ${OpenCV_INCLUDE_DIRS} ${OpenCV_VERSION_MAJOR}")
    include_directories(${OpenCV_INCLUDE_DIRS})
endif(OpenCV_DIR)

if(${OpenCV_VERSION_MAJOR} EQUAL "2")
    add_definitions("-DOPENCV_24")
endif(${OpenCV_VERSION_MAJOR} EQUAL "2")

if(WITH_SKEW_DETECTION)
    find_package(skewDetection)
    add_definitions("-DWITH_SKEW_DETECTION")
    set(EXTRA_LIBS "skewDetection")
endif(WITH_SKEW_DETECTION)

add_subdirectory(src)


if(ANDROID)
  include(cmake/OpenCVDetectAndroidSDK.cmake REQUIRED)

  if(NOT ANDROID_TOOLS_Pkg_Revision GREATER 13)
    message(WARNING "OpenCV requires Android SDK tools revision 14 or newer. Otherwise tests and samples will no be compiled.")
  endif()
  
  # Generate FastText.mk for ndk-build (Android build tool)
  include(cmake/FastTextGenAndroidMK.cmake REQUIRED)
endif(ANDROID)

# ---------------------------------------------------------------------------
# Export FastTextConfig.cmake for using in other projects 
# ---------------------------------------------------------------------------
export(TARGETS FTreader FILE "${PROJECT_BINARY_DIR}/FastTextLibraryDepends.cmake")
configure_file(FastTextConfig.cmake.in
  "${PROJECT_BINARY_DIR}/FastTextConfig.cmake" @ONLY)


# ---------------------------------------------------------------------------
# Build the documentation
# ---------------------------------------------------------------------------
if(BUILD_DOC)
    message( STATUS "Creating documentation ...")
    find_package(Doxygen)
    if(DOXYGEN_FOUND)
        configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in ${CMAKE_BINARY_DIR}/Doxyfile @ONLY)
        add_custom_target(
            doc ${DOXYGEN_EXECUTABLE} ${CMAKE_BINARY_DIR}/Doxyfile WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
            COMMENT "Generating API documentation with Doxygen" VERBATIM
        )
    endif(DOXYGEN_FOUND)
endif (BUILD_DOC)

