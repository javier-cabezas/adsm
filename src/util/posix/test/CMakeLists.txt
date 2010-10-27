if(CMAKE_BUILD_TYPE MATCHES "Debug")
    set(util_os_test_SRC
        Lock.h Lock.cpp)

    add_gmac_sources(gmac ${util_os_test_SRC})
endif(CMAKE_BUILD_TYPE MATCHES "Debug")
