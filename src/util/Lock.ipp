#ifndef __UTIL_LOCK_IPP_
#define __UTIL_LOCK_IPP_

inline void
Lock::lock()
{
    enterLock(__name);
    MUTEX_LOCK(__mutex);
    exitLock();
}

inline void
Lock::unlock()
{
    MUTEX_UNLOCK(__mutex);
}

inline void
RWLock::lockRead()
{
    enterLock(__name);
    LOCK_READ(__lock);
    exitLock();
}

inline void
RWLock::lockWrite()
{
    enterLock(__name);
    LOCK_WRITE(__lock);
    exitLock();
}

inline void
RWLock::unlock()
{
    LOCK_RELEASE(__lock);
}

#endif
