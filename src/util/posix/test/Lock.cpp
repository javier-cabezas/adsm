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
    owner_ = NULL;
    locked_ = false;

    LockImpl::unlock();

    pthread_mutex_unlock(&internal_);
}

RWLockTest::RWLockTest(const char *name) :
    RWLockImpl(name),
    state_(Idle),
    writer_(NULL)
{
    pthread_mutex_init(&internal_, NULL);
}

RWLockTest::~RWLockTest()
{
    pthread_mutex_destroy(&internal_);
}

void RWLockTest::lockRead() const
{
    pthread_mutex_lock(&internal_);
    REQUIRES(readers_.find(pthread_self()) == readers_.end());
    pthread_mutex_unlock(&internal_);

    RWLockImpl::lockRead();

    pthread_mutex_lock(&internal_);
    ENSURES(state_ == Idle || state_ == Read);
    state_ = Read;
    readers_.insert(pthread_self());
    pthread_mutex_unlock(&internal_);
}

void RWLockTest::lockWrite() const
{
    pthread_mutex_lock(&internal_);
    REQUIRES(readers_.find(pthread_self()) == readers_.end());
    pthread_mutex_unlock(&internal_);

    RWLockImpl::lockWrite();

    pthread_mutex_lock(&internal_);
    ENSURES(readers_.empty() == true);
    ENSURES(state_ == Idle);
    state_ = Write;
    writer_ = pthread_self();
    pthread_mutex_unlock(&internal_);
}

void RWLockTest::unlock() const
{
    pthread_mutex_lock(&internal_);
    if(writer_ == pthread_self()) {
        REQUIRES(readers_.empty() == true);
        REQUIRES(state_ == Write);
        state_ = Idle;
        writer_ = NULL;
    }
    else {
        REQUIRES(readers_.erase(pthread_self()) == 1);
        REQUIRES(state_ == Read);
        if(readers_.empty() == true) state_ = Idle;
    }

    RWLockImpl::unlock();

    pthread_mutex_unlock(&internal_);
}

} }
