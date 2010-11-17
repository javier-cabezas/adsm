add_subdirectory(${OS_DIR})

set(util_SRC
    Atomics.h Thread.h
    Lock.h Lock-impl.h Lock.cpp
    Semaphore.h
    Private.h
    Parameter.h Parameter-impl.h Parameter.cpp Parameter-def.h
    Logger.h Logger-impl.h Logger.cpp
    ReusableObject.h
    Reference.h Reference-impl.h
    Singleton.h Singleton-impl.h
    FileLock.h)

add_gmac_sources(gmac ${util_SRC})
add_gmac_sources(gmac-dbc)
