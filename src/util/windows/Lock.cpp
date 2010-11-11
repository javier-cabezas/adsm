#include <string>

#include "Lock.h"

namespace gmac { namespace util { namespace __impl {

Lock::Lock(const char *name) :
    ParaverLock(name)
{
    InitializeCriticalSection(&mutex_);
}

Lock::~Lock()
{
    DeleteCriticalSection(&mutex_);
}

RWLock::RWLock(const char *name) :
    ParaverLock(name),
	owner_(0)
{
    InitializeSRWLock(&lock_);
}

RWLock::~RWLock()
{
}

}}}
