# Generate configuraion file
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/config.h.in ${CMAKE_CURRENT_BINARY_DIR}/config.h)

set(gmac_config_SRC
    config.h threads.h order.h debug.h common.h)

add_gmac_sources(gmac ${gmac_config_SRC})
