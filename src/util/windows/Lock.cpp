#include <string>

#include "Lock.h"

namespace __impl { namespace util {

Lock::Lock(const char *name) :
    __Lock(name)
{
    InitializeCriticalSection(&mutex_);
}

Lock::~Lock()
{
    DeleteCriticalSection(&mutex_);
}

RWLock::RWLock(const char *name) :
    __Lock(name),
	owner_(0)
{
    InitializeSRWLock(&lock_);
}

RWLock::~RWLock()
{
}

}}
