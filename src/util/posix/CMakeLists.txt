add_subdirectory(dbc)
set(util_os_SRC
    Lock.h Lock-impl.h Lock.cpp
    FileLock.h FileLock-impl.h FileLock.cpp
    Semaphore.h Semaphore-impl.h Semaphore.cpp
    Private.h Private-impl.h)

add_gmac_sources(gmac ${util_os_SRC})
add_gmac_sources(gmac-dbc)
