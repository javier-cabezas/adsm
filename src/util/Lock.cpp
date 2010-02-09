#include "Lock.h"

namespace gmac { namespace util {

Lock::Lock(paraver::LockName __name) :
    __name(__name)
{
    MUTEX_INIT(__mutex);
}

Lock::~Lock()
{
    MUTEX_DESTROY(__mutex);
}

RWLock::RWLock(paraver::LockName __name) :
    __name(__name)
{
    LOCK_INIT(__lock);
}

RWLock::~RWLock()
{
	LOCK_DESTROY(__lock);
}

}}
