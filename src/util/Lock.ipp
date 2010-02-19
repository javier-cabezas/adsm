#ifndef __UTIL_LOCK_IPP_
#define __UTIL_LOCK_IPP_


inline 
Owned::Owned() : __owner(0)
{
}

inline void
Owned::adquire()
{
   ASSERT(__owner == 0);
   __owner = SELF();
}

inline void
Owned::release()
{
   __owner = 0;
}

inline THREAD_ID
Owned::owner()
{
   return __owner;
}

inline void
Lock::lock()
{
   enterLock(__name);
   MUTEX_LOCK(__mutex);
#ifdef DEBUG
   adquire();
#endif
   exitLock();
}

inline void
Lock::unlock()
{
#ifdef DEBUG
   if(owner() != SELF())
      WARNING("Thread 0x%x releases lock owned by 0x%x", SELF(), owner());
   release();
#endif
   MUTEX_UNLOCK(__mutex);
}

inline bool
Lock::tryLock()
{
   return MUTEX_TRYLOCK(__mutex) == 0;
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
#ifdef DEBUG
   if(owner() == SELF())
      WARNING("Lock %d double-locked by %p", __name, owner());
   ASSERT(owner() == 0);
   __write = true;
   adquire();
   TRACE("%p locked by %p", this, __owner);
#endif
   exitLock();
}

inline void
RWLock::unlock()
{
#ifdef DEBUG
   if(__write == true) {
      ASSERT(owner() == SELF());
      __write = false;
      TRACE("%p released by %p", this, __owner);
      release();
   }
#endif
   LOCK_RELEASE(__lock);
}

inline bool
RWLock::tryRead()
{
   return LOCK_TRYREAD(__lock) == 0;
}

inline bool
RWLock::tryWrite()
{
#ifdef DEBUG
   if(SELF() == owner()) return false;
#endif
   return LOCK_TRYWRITE(__lock) == 0;
}

#endif
