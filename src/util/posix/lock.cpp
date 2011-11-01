#include <string>

#include "lock.h"

namespace __impl { namespace util {

#if !defined(__APPLE__)
spinlock::spinlock(const char *name) :
    __impl::util::lock__(name)
{
    pthread_spin_init(&spinlock_, PTHREAD_PROCESS_PRIVATE);
}

spinlock::~spinlock()
{
    pthread_spin_destroy(&spinlock_);
}
#endif

mutex::mutex(const char *name) :
    __impl::util::lock__(name)
{
    pthread_mutex_init(&mutex_, NULL);
}

mutex::~mutex()
{
    pthread_mutex_destroy(&mutex_);
}

lock_rw::lock_rw(const char *name) :
    __impl::util::lock__(name)
{
    pthread_rwlock_init(&lock_, NULL);
}

lock_rw::~lock_rw()
{
    pthread_rwlock_destroy(&lock_);
}

}}
