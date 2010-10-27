add_subdirectory(${OS_DIR})

set(util_SRC
    Lock.h Lock.ipp Lock.cpp
    Semaphore.h
    Private.h
    Parameter.h Parameter.ipp Parameter.cpp Parameter-def.h
    Logger.h Logger.ipp Logger.cpp
    ReusableObject.h
    Reference.h
    Singleton.h Singleton.ipp
    FileLock.h)

add_gmac_sources(gmac ${util_SRC})
