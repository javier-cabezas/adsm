add_subdirectory(${OS_DIR})

set(util_SRC
    Atomics.h
    FileLock.h
    Lock.h
    Lock-impl.h
    Lock.cpp
    Logger.h
    Logger-impl.h
    Logger.cpp
    NonCopyable.h
    Parameter.h
    Parameter-impl.h
    Parameter.cpp
    Parameter-def.h
    Private.h
    Reference.h
    Reference-impl.h
    ReusableObject.h
    Semaphore.h
    Singleton.h
    Singleton-impl.h
    Thread.h
)

add_gmac_sources(gmac-util ${util_SRC})
