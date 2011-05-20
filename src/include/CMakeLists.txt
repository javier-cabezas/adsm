configure_file(${CMAKE_CURRENT_SOURCE_DIR}/gmac/api.h.in ${CMAKE_CURRENT_BINARY_DIR}/gmac/api.h)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/gmac/vm.h.in ${CMAKE_CURRENT_BINARY_DIR}/gmac/vm.h)

set(gmac_include_SRC
    gmac/api.h
    gmac/cuda.h
    gmac/cuda_types.h
    gmac/lite.h
    gmac/lite_types.h
    gmac/opencl
    gmac/opencl.h
    gmac/opencl_types.h
    gmac/types.h
    gmac/visibility.h
    gmac/vm.h
)

add_gmac_sources(gmac-include ${gmac_include_SRC})
