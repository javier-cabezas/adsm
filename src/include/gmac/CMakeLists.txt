configure_file(${CMAKE_CURRENT_SOURCE_DIR}/api.h.in ${CMAKE_CURRENT_BINARY_DIR}/api.h)
set(gmac_config_SRC
    types.h api.h visibility.h vm.h)

add_gmac_sources(gmac ${gmac_config_SRC})
