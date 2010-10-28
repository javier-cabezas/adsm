add_subdirectory(config)

add_subdirectory(include)

add_subdirectory(test)
add_subdirectory(util)
add_subdirectory(trace)

add_subdirectory(core)
add_subdirectory(memory)

add_subdirectory(os)
add_subdirectory(api)

add_subdirectory(gmac)

add_gmac_sources(gmac)
add_gmac_sources(gmac-lib)
add_gmac_sources(gmac-dbc)
