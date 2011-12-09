#include <string>

#include "mutex.h"

namespace __impl { namespace util {

spinlock::spinlock(const char *name) :
    __Lock(name), spinlock_(0)
{
}

spinlock::~spinlock()
{
}

mutex::mutex(const char *name) :
    __Lock(name)
{
    InitializeCriticalSection(&mutex_);
}

mutex::~mutex()
{
    DeleteCriticalSection(&mutex_);
}

lock_rw::lock_rw(const char *name) :
    __Lock(name),
	owner_(0)
{
    InitializeSlock_rw(&lock_);
}

lock_rw::~lock_rw()
{
}

}}
