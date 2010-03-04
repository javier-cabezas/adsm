#include "Semaphore.h"
#include <cassert>



namespace gmac { namespace util {

Semaphore::Semaphore(unsigned v)
{
    pthread_cond_init(&__cond, NULL);
    pthread_mutex_init(&__mutex, NULL);
    __val = v;
}

Semaphore::~Semaphore()
{
    pthread_mutex_destroy(&__mutex);
    pthread_cond_destroy(&__cond);
}

}}
