#include "Lock.h"

namespace gmac { namespace util { namespace __dbc {

Lock::Lock(const char *name) :
    __impl::Lock(name),
    locked_(false),
    owner_(NULL)
{
    pthread_mutex_init(&internal_, NULL);
}

Lock::~Lock()
{
    pthread_mutex_destroy(&internal_);
}

void Lock::lock() const
{
    //pthread_mutex_lock(&internal_);
    //REQUIRES(owner_ != pthread_self());
    //pthread_mutex_unlock(&internal_);

    __impl::Lock::lock();

    pthread_mutex_lock(&internal_);
    ENSURES(owner_ == 0);
    ENSURES(locked_ == false);
    locked_ = true;
    owner_ = pthread_self();
    pthread_mutex_unlock(&internal_);
}

void Lock::unlock() const
{
    pthread_mutex_lock(&internal_);
    REQUIRES(locked_ == true);
    REQUIRES(owner_ == pthread_self());
    owner_ = NULL;
    locked_ = false;

    __impl::Lock::unlock();

    pthread_mutex_unlock(&internal_);
}

RWLock::RWLock(const char *name) :
    __impl::RWLock(name),
    state_(Idle),
    writer_(NULL)
{
    ENSURES(pthread_mutex_init(&internal_, NULL) == 0);
}

RWLock::~RWLock()
{
    ENSURES(pthread_mutex_destroy(&internal_) == 0);
}

void RWLock::lockRead() const
{
    pthread_mutex_lock(&internal_);
    REQUIRES(readers_.find(pthread_self()) == readers_.end());
    pthread_mutex_unlock(&internal_);

    __impl::RWLock::lockRead();

    pthread_mutex_lock(&internal_);
    ENSURES(state_ == Idle || state_ == Read);
    state_ = Read;
    readers_.insert(pthread_self());
    pthread_mutex_unlock(&internal_);
}

void RWLock::lockWrite() const
{
    pthread_mutex_lock(&internal_);
    REQUIRES(readers_.find(pthread_self()) == readers_.end());
    REQUIRES(writer_ != pthread_self());
    pthread_mutex_unlock(&internal_);

    __impl::RWLock::lockWrite();

    pthread_mutex_lock(&internal_);
    ENSURES(readers_.empty() == true);
    ENSURES(state_ == Idle);
    state_ = Write;
    writer_ = pthread_self();
    pthread_mutex_unlock(&internal_);
}

void RWLock::unlock() const
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

    __impl::RWLock::unlock();

    pthread_mutex_unlock(&internal_);
}

}}}
