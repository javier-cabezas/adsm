#ifndef __UTIL_POSIX_LOCK_IPP_
#define __UTIL_POSIX_LOCK_IPP_

#include <debug.h>
#include <threads.h>

#include <cassert>
#include <cstdio>

namespace gmac { namespace util {
inline 
Owned::Owned() : __owner(0)
{
}

inline void
Owned::acquire()
{
   assert(__owner == 0);
   __owner = pthread_self();
}

inline void
Owned::release()
{
   __owner = 0;
}

inline pthread_t
Owned::owner()
{
   return __owner;
}

inline void
Lock::lock()
{
   enterLock(__name);
   pthread_mutex_lock(&__mutex);
#ifdef DEBUG
   acquire();
#endif
   exitLock();
}

inline void
Lock::unlock()
{
#ifdef DEBUG
   if(owner() != pthread_self())
      fprintf(stderr, "WARNING: Thread "FMT_TID" releases lock owned by "FMT_TID"\n", pthread_self(), owner());
   release();
#endif
   pthread_mutex_unlock(&__mutex);
}

inline bool
Lock::tryLock()
{
   return pthread_mutex_trylock(&__mutex) == 0;
}

inline void
RWLock::lockRead()
{
   enterLock(__name);
   pthread_rwlock_rdlock(&__lock);
   exitLock();
}

inline void
RWLock::lockWrite()
{
    enterLock(__name);
    pthread_rwlock_wrlock(&__lock);
#ifdef DEBUG
    if(owner() == pthread_self())
        fprintf(stderr, "WARNING: Lock %d double-locked by "FMT_TID"\n", __name, owner());
    assert(owner() == 0);
    __write = true;
    acquire();
#ifdef LOCK_LOG
    fprintf(stderr, "LOG: %p locked by "FMT_TID"\n", this, owner());
#endif
#endif
    exitLock();
}

inline void
RWLock::unlock()
{
#ifdef DEBUG
    if(__write == true) {
        assert(owner() == SELF());
        __write = false;
#ifdef LOCK_LOG
        fprintf(stderr, "LOG: %p released by "FMT_TID"\n", this, owner());
#endif
        release();
    }
#endif
    pthread_rwlock_unlock(&__lock);
}

inline bool
RWLock::tryRead()
{
    return pthread_rwlock_tryrdlock(&__lock) == 0;
}

inline bool
RWLock::tryWrite()
{
#ifdef DEBUG
    if(pthread_self() == owner()) return false;
#endif
    return pthread_rwlock_trywrlock(&__lock) == 0;
}

}}

#endif
