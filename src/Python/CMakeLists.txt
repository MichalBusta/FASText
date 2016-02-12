set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/src/Python)

find_package(PythonLibs REQUIRED)

find_package(NumPy REQUIRED)

include_directories("/usr/include/Cellar/python/2.7.10")

include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${NUMPY_INCLUDE_DIRS})

include_directories(${PROJECT_SOURCE_DIR}/src)

link_directories(/usr/local/Cellar/python/2.7.10/Frameworks/Python.framework/Versions/2.7/lib)

add_library(fasttext_py 
	"pyFastTextG.cpp"
)

add_library(ftext SHARED
	"pyFastTextAPIG.c"
)


message(system "Python Libs: ${PYTHON_LIBRARIES}")

target_link_libraries(ftext  fasttext_py FTreader  ${OpenCV_LIBS} python2.7)


set_target_properties( ftext
        PROPERTIES
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
        PREFIX ""
    ) 
