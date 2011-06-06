project(GMACSamples)
cmake_minimum_required(VERSION 2.8)

if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE "Release")
endif(NOT CMAKE_BUILD_TYPE)

if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
	set(OS_DIR "posix")
elseif(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
	set(OS_DIR "posix")
elseif(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
	set(OS_DIR "windows")
endif(${CMAKE_SYSTEM_NAME} MATCHES "Linux")

set(GMAC_HPE ../../lib/${CMAKE_IMPORT_LIBRARY_PREFIX}gmac-hpe)
set(GMAC_CL ../../lib/${CMAKE_IMPORT_LIBRARY_PREFIX}gmac-cl$)


include_directories(../include)
include_directories(common)

add_executable(eclVecAdd
	${CMAKE_SOURCE_DIR}/ecl/eclVecAdd.cpp
)
target_link_libraries(eclVecAdd ${GMAC_HPE})

add_executable(eclThreadVecAdd
	${CMAKE_SOURCE_DIR}/ecl/eclThreadVecAdd.cpp
	${CMAKE_SOURCE_DIR}/common/utils.h
	${CMAKE_SOURCE_DIR}/common/${OS_DIR}/utils.cpp
)
target_link_libraries(eclThreadVecAdd ${GMAC_HPE})
