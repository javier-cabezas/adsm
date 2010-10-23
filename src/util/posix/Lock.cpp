#include <string>

#include "Lock.h"

namespace gmac { namespace util {

LockImpl::LockImpl(const char *name) :
    ParaverLock(name)
{
    pthread_mutex_init(&mutex_, NULL);
}

LockImpl::~LockImpl()
{
    pthread_mutex_destroy(&mutex_);
}

RWLockImpl::RWLockImpl(const char *name) :
    ParaverLock(name)
{
    pthread_rwlock_init(&lock_, NULL);
}

RWLockImpl::~RWLockImpl()
{
    pthread_rwlock_destroy(&lock_);
}

}}
