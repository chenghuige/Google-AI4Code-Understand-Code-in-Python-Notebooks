#!/usr/bin/env python
# -*- coding: gbk -*-
#this is for helping creating c++ standard .cc files
#especially for small testing programs or acm programs 
#for single .cc file

import sys,os,datetime

def run(argv):
    file_name = 'CMakeLists.txt' 
    
    if file_name in os.listdir('./'):
        print("The file you want to create already exsits")
        return
    
    #create file
    command = "touch " + file_name
    os.system(command)
    
    #open file
    f = open(file_name, 'w')
    
    #----------------------------------------------------    
    content = """CMAKE_MINIMUM_REQUIRED(VERSION 2.6)
CMAKE_POLICY (SET CMP0015 OLD)
PROJECT(%s)

SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -D_FILE_OFFSET_BITS=64  -DCHG_DEBUG -D_LARGE_FILE -O0  -w -pg -g -fPIC -DP_NEEDS_GNU_CXX_NAMESPACE=1 -fpermissive")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -D_FILE_OFFSET_BITS=64 -D_LARGE_FILE -O3 -fPIC -DP_NEEDS_GNU_CXX_NAMESPACE=1 -DNDEBUG -fpermissive")
SET(CMAKE_CXX_FLAGS_RELEASEDEBUG "$ENV{CXXFLAGS} -DCHG_DEBUG -D_FILE_OFFSET_BITS=64 -D_LARGE_FILE -O3 -fPIC -DP_NEEDS_GNU_CXX_NAMESPACE=1 -fpermissive")

IF (CMAKE_BUILD_TYPE STREQUAL "Debug")
    MESSAGE(STATUS "Debug Mode")
    MESSAGE(STATUS "compiler: " ${CXXFLAGS})
    MESSAGE(STATUS "Flags:" ${CMAKE_CXX_FLAGS_DEBUG} )
ELSEIF (CMAKE_BUILD_TYPE STREQUAL "ReleaseDebug")
    MESSAGE(STATUS "compiler: " ${CXXFLAGS})
    MESSAGE(STATUS "Release Mode without NDEBUG defined")
    MESSAGE(STATUS "Flags:" ${CMAKE_CXX_FLAGS_RELEASEDEBUG})
ELSEIF (CMAKE_BUILD_TYPE STREQUAL "Release")
    MESSAGE(STATUS "Release Mode with NDEBUG defined")
    MESSAGE (STATUS "ABI ${CMAKE_C_COMPILER_ABI}")
    MESSAGE ("CMAKE_PLATFORM_USES_PATH_WHEN_NO_SONAME=[${CMAKE_PLATFORM_USES_PATH_WHEN_NO_SONAME}]")
    MESSAGE(STATUS "compiler: " ${CXXFLAGS})
    MESSAGE(STATUS "Flags:" ${CMAKE_CXX_FLAGS_RELEASE})
ELSE ()
    MESSAGE(STATUS ${CMAKE_BUILD_TYPE}) 
    MESSAGE(STATUS "compiler: " ${CXXFLAGS})
    MESSAGE(STATUS "Release Mode without NDEBUG defined")
    MESSAGE(STATUS "Flags:" ${CMAKE_CXX_FLAGS_RELEASEDEBUG})
ENDIF ()

SET(WORKROOT /home/work/chenghuige/) #may need modify
SET(PROJECTROOT ${WORKROOT}/project)  #may need modify
SET(LIB2 ${WORKROOT}/lib2-64)
SET(ULIB ${LIB2}/ullib)
SET(PUBLIC ${WORKROOT}/public)
SET(THIRD ${WORKROOT}/third-64)
SET(THIRDSRC ${WORKROOT}/thirdsrc)
SET(MYTHIRD ${WORKROOT}/third_chg)

AUX_SOURCE_DIRECTORY(${%s_SOURCE_DIR}/src PROJECT_SRCS)

INCLUDE_DIRECTORIES(
${%s_SOURCE_DIR}/include
${%s_SOURCE_DIR}/utils
${PROJECTROOT}/utils
${THIRD}/boost/include
${THIRD}/glog/include
${THIRD}/gflags/include
${THIRD}/gtest/include
${THIRD}/boost/include
${ULIB}/include
)

LINK_DIRECTORIES(
${THIRD}/boost/lib
${THIRD}/glog/lib
${THIRD}/gflags/lib
${THIRD}/gtest/lib
${THIRD}/boost/lib
${ULIB}/lib
)

SET(LIBS
gflags glog gtest
ullib
boost_filesystem boost_system
)

SET(CMAKE_EXE_LINKER_FLAGS -static)
SET(CMAKE_EXE_LINKER_FLAGS  "-lpthread")


SET(EXECUTABLE_OUTPUT_PATH ./bin)
SET(LIBRARY_OUTPUT_PATH ./bin)

"""%(argv[0], argv[0], argv[0], argv[0])
    f.write(content)
    f.close()

#----------------------------------------------------------
if __name__ == '__main__':
    run(sys.argv[1:]) 
