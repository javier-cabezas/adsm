set(core_hpe_SRC
    address_space.h
    address_space-impl.h
    address_space.cpp
    context.h
    context.cpp
    kernel.h
    kernel-impl.h
    vdevice.h
    vdevice-impl.h
    vdevice.cpp
    process.h
    process-impl.h
    process.cpp
    resource_manager.h
    resource_manager.cpp
    thread.h
    thread.cpp
    thread-impl.h
    vdevice_table.h
    vdevice_table-impl.h
)

set(core_hpe_DBC
    dbc/process.h
    dbc/process.cpp
)

# set(core_hpe_DBC
#     dbc/context.h
#     dbc/context.cpp
#     dbc/Kernel.h
#     dbc/Kernel.cpp
#     dbc/vdevice.h
#     dbc/vdevice.cpp
#     dbc/Process.h
#     dbc/Process.cpp
# )

add_gmac_sources(gmac-core-hpe ${core_hpe_SRC} ${core_hpe_DBC})
