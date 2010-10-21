#include "Lock.h"

namespace gmac { namespace util {

LockTest::LockTest(const char *name) :
    LockImpl(name),
    locked_(false),
    owner_(NULL)
{
    pthread_mutex_init(&internal_, NULL);
}

LockTest::~LockTest()
{
    pthread_mutex_destroy(&internal_);
}

void LockTest::lock() const
{
    LockImpl::lock();
    pthread_mutex_lock(&internal_);
    ENSURES(locked_ == false);
    locked_ = true;
    owner_ = pthread_self();
    pthread_mutex_unlock(&internal_);
}

void LockTest::unlock() const
{
    pthread_mutex_lock(&internal_);
    REQUIRES(locked_ == true);
    EXPECTS(owner_ == pthread_self());
    LockImpl::unlock();
    owner_ = NULL;
    locked_ = false;
    pthread_mutex_unlock(&internal_);
}

} }
