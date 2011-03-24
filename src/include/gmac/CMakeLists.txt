configure_file(${CMAKE_CURRENT_SOURCE_DIR}/vm.h.in ${CMAKE_CURRENT_BINARY_DIR}/vm.h)

set(gmac_config_SRC
    types.h api.h visibility.h vm.h cuda.h opencl.h)

add_gmac_sources(gmac ${gmac_config_SRC})
