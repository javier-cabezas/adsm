if(CMAKE_BUILD_TYPE MATCHES "Debug")
    message(STATUS "Using test lock")
    set(util_os_test_SRC
        Lock.h Lock.cpp)

    add_gmac_sources(gmac ${util_os_test_SRC})
else(CMAKE_BUILD_TYPE MATCHES "Debug")
    message(STATUS "Not using test lock")
endif(CMAKE_BUILD_TYPE MATCHES "Debug")
