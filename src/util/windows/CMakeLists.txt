set(util_os_SRC
    loader.h
    loader.cpp
    lock.h
    lock-impl.h
    lock.cpp
    private.h
    private-impl.h
    semaphore.h
    semaphore-impl.h
    semaphore.cpp
    Thread-impl.h
)

set(util_os_DBC
    dbc/lock.h
    dbc/lock.cpp
)

add_gmac_sources(gmac-util ${util_os_SRC} ${util_os_DBC})
