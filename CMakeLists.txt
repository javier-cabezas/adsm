project(gmac)
cmake_minimum_required(VERSION 2.8)

include(gmac.cmake)
include(CheckIncludeFile)
include(CheckFunctionExists)
include(FindThreads)

# Set the configuration being built
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif(NOT CMAKE_BUILD_TYPE)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

# Identify thread library being used
if(CMAKE_USE_PTHREADS_INIT)
    message(STATUS "Using POSIX threads")
    add_definitions(-DHAVE_PTHREADS)
    set(gmac_LIBS ${gmac_LIBS} ${CMAKE_THREAD_LIBS_INIT})
    add_gmac_test_library(${CMAKE_THREAD_LIBS_INIT})
    set(THREAD_DIR "pthreads")
elseif(CMAKE_USE_WIN32_THREADS_INIT)
    message(STATUS "Using Windows threads")
else(CMAKE_USE_PTHREADS_INIT)
    message(FATAL "Your thread package (if any) is not supported")
endif(CMAKE_USE_PTHREADS_INIT)

# Identify operating system being used
if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    message(STATUS "Compiling for GNU/Linux")
    set(OS_DIR "posix")
    set(gmac_static_FLAGS "-fPIC")
    add_definitions(-DLINUX)
    set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--no-undefined")
endif(${CMAKE_SYSTEM_NAME} MATCHES "Linux")

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    message(STATUS "Compiling for Mac OS X")
    set(OS_DIR "posix")
    set(gmac_static_FLAGS "-fPIC")
    add_definitions(-DDARWIN)
endif(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")

include("CMakeLists-windows.txt" OPTIONAL)
include("CMakeLists-opencl.txt" OPTIONAL)
include("CMakeLists-cuda.txt")

# Check for Tracing
option(USE_TRACE_CONSOLE "Enable producing traces in the console" OFF)
if(USE_TRACE_CONSOLE)
    add_definitions(-DUSE_TRACE_CONSOLE -DUSE_TRACE)
endif(USE_TRACE_CONSOLE)

# Setup configuration options
option(USE_MPI "Enable MPI interposition" OFF)
if(USE_MPI)
    message(STATUS "Using MPI")
    add_definitions(-DUSE_MPI)
endif(USE_MPI)


# Setup configuration options
option(USE_VM "Enable software GPU virtual memory" OFF)
if(USE_VM)
    message(STATUS "Compiling with VM support")
    add_definitions(-DUSE_VM)
    if(VM_BITMAP MATCHES "bit")
        message(STATUS "Using 'bit' bitmap implementation")
        add_definitions(-DBITMAP_BIT)
    else(VM_BITMAP MATCHES "bit")
        message(STATUS "Using 'byte' bitmap implementation")
        add_definitions(-DBITMAP_BYTE)
    endif(VM_BITMAP MATCHES "bit")

    if(USE_HOSTMAP_VM)
        add_definitions(-DUSE_HOSTMAP_VM)
    endif(USE_HOSTMAP_VM)
endif(USE_VM)


# Check if GMAC is buid with Paraver support
#option(USE_PARAVER "Enable production of Paraver traces" OFF)
#if(USE_PARAVER)
#    message(STATUS "Compiling with Paraver support")
#    add_definitions(-DPARAVER)
#    find_library(PARAVER paraver)
#    if(EXISTS ${PARAVER})
#        message(STATUS "Using Paraver library at ${PARAVER}")
#        set(gmac_LIBS ${gmac_LIBS} paraver)
#    else()
#        message(STATUS "Using Paraver library from source code")
#        set(EP_paraver_SRC ${CMAKE_SOURCE_DIR}/lib/libparaver)
#        set(EP_paraver_BIN ${CMAKE_BINARY_DIR}/lib/libparaver)
#        if(CMAKE_BUILD_TYPE MATCHES "Debug")
#            set(PARAVER_CMAKE_ARGS -DCMAKE_BUILD_TYPE=Debug)
#        endif(CMAKE_BUILD_TYPE MATCHES "Debug")
#        include(ExternalProject)
#        ExternalProject_Add(EP_paraver
#            DOWNLOAD_COMMAND ""
#            SOURCE_DIR ${EP_paraver_SRC}
#            CMAKE_ARGS ${PARAVER_CMAKE_ARGS} 
#            BINARY_DIR ${EP_paraver_BIN}
#            INSTALL_COMMAND ""
#            )
#        add_dependencies(gmac EP_paraver)
#        include_directories(${EP_paraver_SRC} ${EP_paraver_BIN})
#        set(gmac_LIBS ${gmac_LIBS}
#            ${EP_paraver_BIN}/paraver/${CMAKE_SHARED_LIBRARY_PREFIX}paraver${CMAKE_SHARED_LIBRARY_SUFFIX})
#        install(CODE "execute_process(COMMAND make install WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/lib/libparaver)")
#    endif()
#endif(USE_PARAVER)

# Make sure we compile the necessary libraries
add_subdirectory(lib)
import_gmac_libraries()

# Configure compiler flags for different configurations
if(CMAKE_COMPILER_IS_GNUCC)
    set(CMAKE_C_FLAGS "-Wall -Werror -Wno-unused-variable")
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS} -O3 -g -fomit-frame-pointer")
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS} -g -DDEBUG -O0")
elseif(MSVC)
    set(CMAKE_C_FLAGS " /W3 /WX /EHsc")
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS} /Zi /O2 /MD")
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS} /DDEBUG /Zi /Od /MDd")
endif(CMAKE_COMPILER_IS_GNUCC)

set(CMAKE_CXX_FLAGS_RELEASE ${CMAKE_C_FLAGS_RELEASE})
set(CMAKE_CXX_FLAGS ${CMAKE_C_FLAGS})
set(CMAKE_CXX_FLAGS_DEBUG ${CMAKE_C_FLAGS_DEBUG})

# Check for header files
check_function_exists(llabs HAVE_LLABS)

# Setup extra libraries needed by GMAC
if(${OS_DIR} MATCHES "posix")
    message(STATUS "Compiling for POSIX")
    add_definitions(-DPOSIX)
    set(dl_NAME "${CMAKE_SHARED_LIBRARY_PREFIX}dl${CMAKE_SHARED_LIBRARY_SUFFIX}")
    find_library(dl_LIB ${dl_NAME})
    if(NOT dl_LIB)
        message(FATAL "${dl_NAME} not found")
    endif(NOT dl_LIB)

    set(gmac_LIBS ${gmac_LIBS} ${dl_LIB})
    check_function_exists(posix_memalign HAVE_POSIX_MEMALIGN)
elseif(${OS_DIR} MATCHES "windows")
endif(${OS_DIR} MATCHES "posix")

# Add subdirectories
add_subdirectory(src)
add_gmac_groups(${gmac_SRC})


# Check if tests must be compiled
option(MAKE_TESTS "Compile GMAC tests" OFF)
if(MAKE_TESTS)
    message(STATUS "Compile GMAC tests")
    add_subdirectory(tests)
    # Add libraries used in testing
endif(MAKE_TESTS)

# Check if doc must be generated
option(MAKE_DOC "Generate GMAC documentation" OFF)
if(MAKE_DOC)
    message(STATUS "Generate GMAC documentation")
    add_subdirectory(doc)
endif(MAKE_DOC)

# Setup GMAC as main target
include_directories(${CMAKE_SOURCE_DIR}/src
                    ${CMAKE_BINARY_DIR}/src
                    ${CMAKE_BINARY_DIR}/src/include)
# GMAC is a DLL
add_definitions(-DGMAC_DLL)
# Compiling GMAC
add_definitions(-DGMAC_DLL_EXPORTS)

add_gmac_groups(${gmac_SRC} ${gmac-lib_SRC} ${gmac-dbc_SRC})
add_library(gmac-base STATIC ${gmac_SRC} ${gmac-dbc_SRC})
set_target_properties(gmac-base PROPERTIES
    COMPILE_FLAGS ${gmac_static_FLAGS}
    COMPILE_DEFINITIONS_DEBUG USE_DBC)
 

add_library(gmac SHARED ${gmac-lib_SRC})
set_property(TARGET gmac PROPERTY PUBLIC_HEADER
    ${CMAKE_SOURCE_DIR}/src/include/gmac/api.h
    ${CMAKE_SOURCE_DIR}/src/include/gmac/types.h
    ${CMAKE_SOURCE_DIR}/src/include/gmac/cuda.h
    ${CMAKE_SOURCE_DIR}/src/include/gmac/cuda_types.h
    ${CMAKE_SOURCE_DIR}/src/include/gmac/opencl.h
    ${CMAKE_SOURCE_DIR}/src/include/gmac/opencl_types.h
    ${CMAKE_SOURCE_DIR}/src/include/gmac/visibility.h
    ${CMAKE_SOURCE_DIR}/src/include/gmac/vm.h
)


target_link_libraries(gmac gmac-base ${gmac_LIBS})
link_directories(${gmac_LIBDIR})

set_target_properties(gmac PROPERTIES
    COMPILE_DEFINITIONS_DEBUG USE_DBC
    VERSION 0.2.0 SOVERSION 1)

install(TARGETS gmac 
     LIBRARY DESTINATION lib
     ARCHIVE DESTINATION lib
     RUNTIME DESTINATION bin
     PUBLIC_HEADER DESTINATION include/gmac)

include(CPack)
