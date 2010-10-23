project(gmac)
cmake_minimum_required(VERSION 2.8)

include(gmac.cmake)
include(CheckIncludeFile)
include(CheckFunctionExists)
include(FindThreads)

# Identify thread library being used
if(CMAKE_USE_PTHREADS_INIT)
    message(STATUS "Using POSIX threads")
    add_definitions(-DHAVE_PTHREADS)
    set(gmac_LIBS ${gmac_LIBS} ${CMAKE_THREAD_LIBS_INIT})
    set(THREAD_DIR "pthreads")
elseif(CMAKE_USE_WIN32_THREADS_INIT)
    message(STATUS "Using Windows threads")
    add_definitions(/DHAVE_WINTHREADS)
endif(CMAKE_USE_PTHREADS_INIT)

# Identify operating system being used
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


# Identify API being used
option(USE_OPENCL "Enable OpenCL backend" OFF)
if(USE_OPENCL)
    message(FATAL_ERROR "OpenCL still not supported")
else(USE_OPENCL)
    # Default compilation for CUDA
    find_path(NVCC nvcc${CMAKE_EXECUTABLE_SUFFIX}
            HINTS ENV "CUDA_BIN_PATH")
    if(EXISTS ${NVCC})
        message(STATUS "Compiling for NVIDIA CUDA: ${NVCC}")
        set(API_DIR "cuda")
        set(CUDA_INCLUDE "${NVCC}/../include")
        include_directories(${CUDA_INCLUDE})
        find_library(cuda_LIB
            ${CMAKE_IMPORT_LIBRARY_PREFIX}cuda${CMAKE_IMPORT_LIBRARY_SUFFIX}
            HINTS ENV "CUDA_LIB_PATH")
        set(gmac_LIBS ${gmac_LIBS} ${cuda_LIB})
    else(EXISTS ${NVCC})
        message(FATAL_ERROR "NVIDIA CUDA Compiler is required")
    endif(EXISTS ${NVCC})
endif(USE_OPENCL)



# Add subdirectories
add_subdirectory(src)

# Setup GMAC as main target
include_directories(${CMAKE_SOURCE_DIR}/src ${CMAKE_BINARY_DIR}/src)
get_property(gmac_SRC GLOBAL PROPERTY gmac_SRC)
add_library(gmac SHARED ${gmac_SRC})
set_property(TARGET gmac PROPERTY PUBLIC_HEADER ${CMAKE_SOURCE_DIR}/src/gmac/gmac.h)
set_property(TARGET gmac PROPERTY PRIVATE_HEADER ${CMAKE_SOURCE_DIR}/src/gmac/gmac/vm.h)

# Setup configuration options
option(USE_VM "Enable software GPU virtual memory" OFF)
if(USE_VM)
    message(STATUS "Compiling with VM support")
    add_definitions(-DUSE_VM)
    if(VM_BITMAP MATCHES "bit")
        message(STATUS "Using 'bit' bitmap implementation")
        add_definitions(-DBITMAP_BIT)
    elseif(VM_BITMAP MATCHES "word")
        message(STATUS "Using 'word' bitmap implementation")
        add_definitions(-DBITMAP_WORD)
    else(VM_BITMAP MATCHES "bit")
        message(STATUS "Using 'byte' bitmap implementation")
        add_definitions(-DBITMAP_BYTE)
    endif(VM_BITMAP MATCHES "bit")
endif(USE_VM)


# Check if GMAC is buid with Paraver support
option(USE_PARAVER "Enable production of Paraver traces" OFF)
if(USE_PARAVER)
    message(STATUS "Compiling with Paraver support")
    add_definitions(-DPARAVER)
    find_library(PARAVER paraver)
    if(EXISTS ${PARAVER})
        message(STATUS "Using Paraver library at ${PARAVER}")
        set(gmac_LIBS ${gmac_LIBS} paraver)
    else()
        message(STATUS "Using Paraver library from source code")
        set(EP_paraver_SRC ${CMAKE_SOURCE_DIR}/lib/libparaver)
        set(EP_paraver_BIN ${CMAKE_BINARY_DIR}/lib/libparaver)
        if(CMAKE_BUILD_TYPE MATCHES "Debug")
            set(PARAVER_CMAKE_ARGS -DCMAKE_BUILD_TYPE=Debug)
        endif(CMAKE_BUILD_TYPE MATCHES "Debug")
        include(ExternalProject)
        ExternalProject_Add(EP_paraver
            DOWNLOAD_COMMAND ""
            SOURCE_DIR ${EP_paraver_SRC}
            CMAKE_ARGS ${PARAVER_CMAKE_ARGS} 
            BINARY_DIR ${EP_paraver_BIN}
            INSTALL_COMMAND ""
            )
        add_dependencies(gmac EP_paraver)
        include_directories(${EP_paraver_SRC} ${EP_paraver_BIN})
        set(gmac_LIBS ${gmac_LIBS}
            ${EP_paraver_BIN}/paraver/${CMAKE_SHARED_LIBRARY_PREFIX}paraver${CMAKE_SHARED_LIBRARY_SUFFIX})
        install(CODE "execute_process(COMMAND make install WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/lib/libparaver)")
    endif()
endif(USE_PARAVER)

# Check if tests must be compiled
option(MAKE_TESTS "Compile GMAC tests" OFF)
if(MAKE_TESTS)
    message(STATUS "Compile GMAC tests")
    add_subdirectory(tests)
endif(MAKE_TESTS)

# Check if doc must be generated
if(MAKE_DOC)
    message(STATUS "Generate GMAC documentation")
    add_subdirectory(doc)
endif(MAKE_DOC)


# Set default configuration to RELEASE
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TPYE Release)
endif(NOT CMAKE_BUILD_TYPE)

# Configure compiler flags for different configurations
if(CMAKE_COMPILER_IS_GNUCC)
    set(CMAKE_C_FLAGS "-Wall -Werror -Wno-unused-variable")
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS} -O3 -fomit-frame-pointer")
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS} -DDEBUG -g -O0")
elseif(MSVC)
        set(CMAKE_C_FLAGS " /W4 /WX /EHsc")
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS} /Zi /O2 /MD")
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS} /DDEBUG /Zi /Od /MDd")
endif(CMAKE_COMPILER_IS_GNUCC)

set(CMAKE_CXX_FLAGS_RELEASE ${CMAKE_C_FLAGS_RELEASE})
set(CMAKE_CXX_FLAGS ${CMAKE_C_FLAGS})
set(CMAKE_CXX_FLAGS_DEBUG ${CMAKE_C_FLAGS_DEBUG})

# Check for header files
check_include_file("stdint.h" HAVE_STDINT)

# Setup extra libraries needed by GMAC
if(${OS_DIR} MATCHES "posix")
    message(STATUS "Compiling for POSIX")
    add_definitions(-DPOSIX)
    find_library(dl_LIB "${CMAKE_SHARED_LIBRARY_PREFIX}dl${CMAKE_SHARED_LIBRARY_SUFFIX}")
    set(gmac_LIBS ${gmac_LIBS} ${dl_LIB})
    check_function_exists(posix_memalign HAVE_POSIX_MEMALIGN)
endif(${OS_DIR} MATCHES "posix")

# GMAC is a DLL
add_definitions(-DGMAC_DLL)
# Compiling GMAC
add_definitions(-DGMAC_DLL_EXPORTS)

target_link_libraries(gmac ${gmac_LIBS})
link_directories(${gmac_LIBDIR})
set_target_properties(gmac PROPERTIES VERSION 0.2.0 SOVERSION 1)

install(TARGETS gmac 
     LIBRARY DESTINATION lib
     ARCHIVE DESTINATION lib
     RUNTIME DESTINATION bin
     PUBLIC_HEADER DESTINATION include
     PRIVATE_HEADER DESTINATION include/gmac)

configure_gmac_groups()

