add_subdirectory(stdc)
add_subdirectory(${OS_DIR})
if(DEFINED THREAD_DIR)
    add_subdirectory(${THREAD_DIR})
endif(DEFINED THREAD_DIR)

set(gmac_lib_SRC
    gmac.cpp
    paraver.h
    init.h init.cpp)
add_gmac_sources(gmac-lib ${gmac_lib_SRC})
add_gmac_sources(gmac ${gmac_SRC})
