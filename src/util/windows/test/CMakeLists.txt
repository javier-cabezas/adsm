if(CMAKE_BUILD_TYPE MATCHES "Debug")
    message(STATUS "Using test lock")
    set(util_os_test_SRC
        Lock.h Lock.cpp)

    add_gmac_sources(${util_os_test_SRC})
endif(CMAKE_BUILD_TYPE MATCHES "Debug")
