#ifndef GMAC_TRACE_PARAVER_POSIX_LOCK_IMPL_H_
#define GMAC_TRACE_PARAVER_POSIX_LOCK_IMPL_H_

namespace __impl { namespace trace { namespace paraver {

inline
Lock::Lock() 
{
    pthread_mutex_init(&mutex_, NULL);
}

inline
Lock::~Lock()
{
    pthread_mutex_destroy(&mutex_);
}


inline void
Lock::lock() const
{
    pthread_mutex_lock(&mutex_);
}

inline void
Lock::unlock() const
{
    pthread_mutex_unlock(&mutex_);
}

} } }

#endif
