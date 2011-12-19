set(util_os_SRC
    FileLock.h
    FileLock-impl.h
    FileLock.cpp
    loader.h
    lock.h
    lock-impl.h
    lock.cpp
    private.h
    private-impl.h
    Thread-impl.h
    semaphore.h
    semaphore-impl.h
    semaphore.cpp
)

set(util_os_DBC
    dbc/lock.h
    dbc/lock.cpp
)

add_gmac_sources(gmac-util ${util_os_SRC} ${util_os_DBC})
