add_subdirectory(stdc)
add_subdirectory(${OS_DIR})
add_subdirectory(mpi)

if(DEFINED THREAD_DIR)
    add_subdirectory(${THREAD_DIR})
endif(DEFINED THREAD_DIR)

set(gmac_lib_SRC
    gmac.cpp
    init.h
    init.cpp
    stdc/io.cpp
    stdc/stdc.h
    stdc/stdc.cpp
)

add_gmac_sources(gmac-hpe-api ${gmac_lib_SRC})
