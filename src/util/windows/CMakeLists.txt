add_subdirectory(test)
set(util_os_SRC
    Lock.h Lock.ipp Lock.cpp
    Semaphore.h Semaphore.ipp Semaphore.cpp
    Private.h Private.ipp)

add_gmac_sources(${util_os_SRC})
