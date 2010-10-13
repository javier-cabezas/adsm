project(gmac)
cmake_minimum_required(VERSION 2.8)

include(CheckFunctionExists)
include(FindThreads)
include(gmac.cmake)

if (CMAKE_USE_PTHREADS_INIT)
    add_definitions(-DHAVE_PTHREADS)
    set(gmac_LIBS ${gmac_LIBS} ${CMAKE_THREAD_LIBS_INIT})
    set(THREAD_DIR "pthreads")
endif (CMAKE_USE_PTHREADS_INIT)

if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    message(STATUS "Compiling for GNU/Linux")
    set(OS_DIR "posix")
    add_definitions(-DLINUX)
    set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--no-undefined")
endif(${CMAKE_SYSTEM_NAME} MATCHES "Linux")

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    message(STATUS "Compiling for Mac OS X")
    set(OS_DIR "posix")
    add_definitions(-DDARWIN)
endif(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")

if(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
    message(STATUS "Compiling for Windows")
    set(OS_DIR "windows")
    add_definitions(-DWINDOWS)
endif(${CMAKE_SYSTEM_NAME} MATCHES "Windows")

if(USE_OPENCL)
    message(FATAL_ERROR "OpenCL still not supported")
else()
    # Default compilation for CUDA
    find_path(NVCC nvcc)
    if(EXISTS ${NVCC})
        message(STATUS "Compiling for NVIDIA CUDA: ${NVCC}")
        set(API_DIR "cuda")
        set(CUDA_INCLUDE "${NVCC}/../include")
        include_directories(${CUDA_INCLUDE})
        find_library(cuda_LIB "libcuda.so")
        set(gmac_LIBS ${gmac_LIBS} ${cuda_LIB})
    else(EXISTS ${NVCC})
        message(FATAL_ERROR "NVIDIA CUDA Compiler is required")
    endif(EXISTS ${NVCC})
endif(USE_OPENCL)

if(USE_VM)
    message(STATUS "Compiling with VM support")
    add_definitions(-DUSE_VM)
    if(VM_BITMAP MATCHES "byte")
        message(STATUS "Using 'byte' bitmap implementation")
        add_definitions(-DBITMAP_BYTE)
    else(VM_BITMAP MATCHES "byte")
        if(VM_BITMAP MATCHES "bit")
            message(STATUS "Using 'bit' bitmap implementation")
            add_definitions(-DBITMAP_BIT)
        else(VM_BITMAP MATCHES "bit")
            if(VM_BITMAP MATCHES "word")
                message(STATUS "Using 'word' bitmap implementation")
            else(VM_BITMAP MATCHES "word")
                message(STATUS "Using 'byte' bitmap implementation by default")
            endif(VM_BITMAP MATCHES "word")
        endif(VM_BITMAP MATCHES "bit")
    endif(VM_BITMAP MATCHES "byte")
endif(USE_VM)


if(${OS_DIR} MATCHES "posix")
    message(STATUS "Compiling for POSIX")
    add_definitions(-DPOSIX)
    find_library(dl_LIB "libdl.so")
    set(gmac_LIBS ${gmac_LIBS} ${dl_LIB})
    check_function_exists(posix_memalign HAVE_POSIX_MEMALIGN)
endif(${OS_DIR} MATCHES "posix")

# Set default configuration to RELEASE
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TPYE Release)
endif(NOT CMAKE_BUILD_TYPE)

# Configure compiler flags for different configurations
set(CMAKE_C_FLAGS "-Wall -Werror -Wno-unused-variable")
set(CMAKE_CXX_FLAGS ${CMAKE_C_FLAGS})

set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS} -O3 -fomit-frame-pointer")
set(CMAKE_CXX_FLAGS_RELEASE ${CMAKE_C_FLAGS_RELEASE})

set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS} -g -O0")
set(CMAKE_CXX_FLAGS_DEBUG ${CMAKE_C_FLAGS_DEBUG})

if(USE_PARAVER)
    message(STATUS "Compiling with Paraver support")
    find_library(PARAVER paraver)
    if(EXISTS ${PARAVER})
        message(STATUS "Using Paraver library at ${PARAVER}")
    else()
        message(STATUS "Using Paraver library from source code")
        set(COMPILE_PARAVER true)
    endif()
endif(USE_PARAVER)


# Add subdirectories
add_subdirectory(lib)
add_subdirectory(src)


# Build GMAC library
include_directories(${CMAKE_SOURCE_DIR}/src ${CMAKE_BINARY_DIR}/src)
get_property(gmac_SRC GLOBAL PROPERTY gmac_SRC)
add_library(gmac SHARED ${gmac_SRC})
target_link_libraries(gmac ${gmac_LIBS})
set_target_properties(gmac PROPERTIES VERSION 0.2.0 SOVERSION 1)

install(TARGETS gmac LIBRARY DESTINATION lib)
install(FILES  gmac.h gmac/vm.h DESTINATION include)

#add_subdirectory(doc)
if(MAKE_TESTS)
    message(STATUS "Compiling GMAC tests")
    add_subdirectory(tests)
endif(MAKE_TESTS)


