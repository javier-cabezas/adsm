add_subdirectory(stdc)
add_subdirectory(${OS_DIR})
add_subdirectory(${THREAD_DIR})

set(gmac_lib_SRC
    gmac.cpp
    paraver.h
    init.h init.cpp)
add_gmac_sources(${gmac_lib_SRC})


