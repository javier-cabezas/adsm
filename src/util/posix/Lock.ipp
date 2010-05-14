#ifndef __UTIL_POSIX_LOCK_IPP_
#define __UTIL_POSIX_LOCK_IPP_

inline 
Owned::Owned() : __owner(0), logger("Lock")
{
}

inline void
Owned::acquire()
{
   ASSERT(__owner == 0);
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
      WARNING("Thread "FMT_TID" releases lock owned by "FMT_TID, pthread_self(), owner());
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
        logger.warning("Lock %d double-locked by "FMT_TID, __name, owner());
    logger.assertion(owner() == 0);
    __write = true;
    acquire();
    logger.trace("%p locked by "FMT_TID, this, owner());
#endif
    exitLock();
}

inline void
RWLock::unlock()
{
#ifdef DEBUG
    if(__write == true) {
        logger.assertion(owner() == SELF());
        __write = false;
        logger.trace("%p released by "FMT_TID, this, owner());
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

#endif
