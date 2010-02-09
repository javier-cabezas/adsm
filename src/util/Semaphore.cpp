#include "Semaphore.h"

namespace gmac { namespace util {

Semaphore::Semaphore(unsigned v)
{
    SEM_INIT(__sem, v);
}

Semaphore::~Semaphore()
{
    SEM_DESTROY(__sem);
}

}}
