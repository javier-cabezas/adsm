if(CMAKE_BUILD_TYPE MATCHES "Debug")
    set(memory_protocol_test_SRC
        Lazy.h Lazy.cpp)

    add_gmac_sources(gmac ${memory_protocol_test_SRC})
endif(CMAKE_BUILD_TYPE MATCHES "Debug")
