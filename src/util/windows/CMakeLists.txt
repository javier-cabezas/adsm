set(util_os_SRC
    loader.h
    loader.cpp
    lock.h
    lock-impl.h
    lock.cpp
    Private.h
    Private-impl.h
    Semaphore.h
    Semaphore-impl.h
    Semaphore.cpp
    Thread-impl.h
)

set(util_os_DBC
    dbc/lock.h
    dbc/lock.cpp
)

add_gmac_sources(gmac-util ${util_os_SRC} ${util_os_DBC})
