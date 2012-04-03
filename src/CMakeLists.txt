add_subdirectory(config)
add_subdirectory(include)

add_subdirectory(util)

add_subdirectory(hal)

add_subdirectory(trace)

add_subdirectory(core)
add_subdirectory(memory)

add_subdirectory(dsm)
add_subdirectory(uvas)

# add_subdirectory(api)

add_subdirectory(libs)

# add_subdirectory(hpe)

group_gmac_sources(gmac-common gmac-config gmac-util gmac-dbc gmac-hal gmac-hal-api gmac-hal-cpu)
group_gmac_sources(gmac-trace gmac-trace)
group_gmac_sources(gmac-memory gmac-memory)
group_gmac_sources(gmac-dsm gmac-dsm)
group_gmac_sources(gmac-uvas gmac-uvas)
group_gmac_sources(gmac-core gmac-core gmac-arch)
group_gmac_sources(gmac-libs gmac-libs)
group_gmac_sources(gmac-include gmac-include)

group_gmac_sources(gmac-hpe gmac-hpe-api gmac-arch-hpe-api gmac-hal-api-hpe)

if(MAKE_CL)
    group_gmac_sources(gmac-core-lite gmac-arch-lite)
    group_gmac_sources(gmac-lite gmac-arch-lite-api)
    group_gmac_sources(gmac-lite-libs gmac-libs-lite-api)
endif(MAKE_CL)
