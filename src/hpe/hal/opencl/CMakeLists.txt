configure_file(${CMAKE_CURRENT_SOURCE_DIR}/gmac.cpp.in ${CMAKE_CURRENT_BINARY_DIR}/gmac.cpp)

set(hal_opencl_hpe_SRC
    gmac.cpp
)

add_gmac_sources(gmac-hal-api-hpe ${hal_opencl_hpe_SRC})
