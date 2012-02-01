add_subdirectory(hal)

if(DEFINED THREAD_DIR)
    add_subdirectory(${THREAD_DIR})
endif(DEFINED THREAD_DIR)

set(hpe_lib_SRC
    gmac.cpp
    init.h
    init.cpp
    core/address_space.h
    core/address_space-impl.h
    core/address_space.cpp
    core/kernel.h
    core/kernel-impl.h
    core/vdevice.h
    core/vdevice-impl.h
    core/vdevice.cpp
    core/process.h
    core/process-impl.h
    core/process.cpp
    core/resource_manager.h
    core/resource_manager.cpp
    core/thread.h
    core/thread.cpp
    core/thread-impl.h
    core/vdevice_table.h
    core/vdevice_table-impl.h
)

set(hpe_lib_DBC
    core/dbc/process.h
    core/dbc/process.cpp
)

add_gmac_sources(gmac-hpe-api ${hpe_lib_SRC} ${hpe_lib_DBC})
add_gmac_sources(gmac-hal-api-hpe)
