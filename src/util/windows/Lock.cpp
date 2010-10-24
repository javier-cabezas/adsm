#include <string>

#include "Lock.h"

namespace gmac { namespace util {

LockImpl::LockImpl(const char *name) :
    ParaverLock(name)
{
    InitializeCriticalSection(&mutex_);
}

LockImpl::~LockImpl()
{
    DeleteCriticalSection(&mutex_);
}

RWLockImpl::RWLockImpl(const char *name) :
    ParaverLock(name)
{
    InitializeSRWLock(&lock_);
}

RWLockImpl::~RWLockImpl()
{
}

}}
