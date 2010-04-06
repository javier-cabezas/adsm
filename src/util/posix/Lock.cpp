#include "Lock.h"

namespace gmac { namespace util {

Lock::Lock(paraver::LockName __name) :
    __name(__name)
{
    pthread_mutex_init(&__mutex, NULL);
}

Lock::~Lock()
{
    pthread_mutex_destroy(&__mutex);
}

RWLock::RWLock(paraver::LockName __name) :
   __write(false),
   __name(__name)
{
    pthread_rwlock_init(&__lock, NULL);
}

RWLock::~RWLock()
{
    pthread_rwlock_destroy(&__lock);
}

}}
