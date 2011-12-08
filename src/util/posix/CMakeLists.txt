set(util_os_SRC
    FileLock.h
    FileLock-impl.h
    FileLock.cpp
    loader.h
    lock.h
    lock-impl.h
    lock.cpp
    Thread-impl.h
    Semaphore.h
    Semaphore-impl.h
    Semaphore.cpp
)

set(util_os_DBC
    dbc/lock.h
    dbc/lock.cpp
)

add_gmac_sources(gmac-util ${util_os_SRC} ${util_os_DBC})
