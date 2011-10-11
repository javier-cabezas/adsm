add_subdirectory(${API_DIR})

set(hal_SRC
    types.h
    types-detail.h
    types-impl.h
    Device.h
    Device.cpp
)

add_gmac_sources(gmac-hal ${hal_SRC})
add_gmac_sources(gmac-hal-api)
