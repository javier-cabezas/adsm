#ifndef __UTIL_LOCK_IPP_
#define __UTIL_LOCK_IPP_


inline 
Owned::Owned() : __owner(0)
{
}

inline void
Owned::adquire()
{
   assert(__owner == 0);
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
   adquire();
   exitLock();
}

inline void
Lock::unlock()
{
   if(owner() != SELF())
      WARNING("Thread 0x%x releases lock owned by 0x%x", SELF(), owner());
   release();
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
   assert(owner() == 0);
   __write = true;
   adquire();
   TRACE("%p locked by %p", this, __owner);
   exitLock();
}

inline void
RWLock::unlock()
{
   if(__write == true) {
      assert(owner() == SELF());
      __write = false;
      TRACE("%p released by %p", this, __owner);
      release();
   }
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
   if(SELF() == owner()) return false;
   return LOCK_TRYWRITE(__lock) == 0;
}

#endif
