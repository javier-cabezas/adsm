if(CMAKE_BUILD_TYPE MATCHES "Debug")
    set(memory_test_SRC
        SharedObject.h SharedObject.ipp)

    add_gmac_sources(${memory_test_SRC})
endif(CMAKE_BUILD_TYPE MATCHES "Debug")
