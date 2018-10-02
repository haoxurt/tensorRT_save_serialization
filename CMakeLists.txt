CMAKE_MINIMUM_REQUIRED(VERSION 3.0)
PROJECT(tensorRT_save_engine C CXX)

set(CMAKE_BUILD_TYPE "Release")



SET(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)

SET(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -ffast-math -Wall -pthread -fopenmp -std=c++11")

SET(INCLUDES_DIRS /usr/local/include ../../include)
SET(LIBS_DIRS /usr/local/lib ../../lib)
SET(LIBS nvinfer_plugin nvinfer nvcaffe_parser pthread)

INCLUDE_DIRECTORIES(${INCLUDES_DIRS} ${CMAKE_SOURCE_DIR}/include)
LINK_DIRECTORIES(${LIBS_DIRS} ${CMAKE_SOURCE_DIR}/lib)
LINK_LIBRARIES(${LIBS})
SET(LIBRARY_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/lib)


add_executable(tensorRT_save_engine tensorRT_save_engine.cpp)
#target_link_libraries(tensorRT_save_engine ${LIBS}) 



