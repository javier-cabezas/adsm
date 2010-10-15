#include <string>

#include "Lock.h"

namespace gmac { namespace util {

Lock::Lock(const char *name) :
    ParaverLock(name)
{
    pthread_mutex_init(&mutex_, NULL);
}

Lock::~Lock()
{
    pthread_mutex_destroy(&mutex_);
}

RWLock::RWLock(const char *name) :
    ParaverLock(name)
{
    pthread_rwlock_init(&lock_, NULL);
}

RWLock::~RWLock()
{
    pthread_rwlock_destroy(&lock_);
}

}}
