
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/src/Python)

find_package(PythonLibs 2.7 REQUIRED)

find_package(NumPy REQUIRED)

include_directories(${PYTHON_INCLUDE_DIRS})
include_directories("/usr/include/python2.7")

include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${NUMPY_INCLUDE_DIRS})

include_directories(${PROJECT_SOURCE_DIR}/src)

add_library(fasttext_py 
	"pyFastTextG.cpp"
)

add_library(ftext SHARED
	"pyFastTextAPIG.c"
)

target_link_libraries(ftext  fasttext_py FTreader  ${OpenCV_LIBS} ${PYTHON_LIBRARIES})

set_target_properties( ftext
        PROPERTIES
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
        PREFIX ""
    ) 
