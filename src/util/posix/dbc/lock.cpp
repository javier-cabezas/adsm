#ifdef USE_DBC

#include "lock.h"

namespace __dbc { namespace util {

#if 0
#if !defined(__APPLE__)
spinlock::spinlock(const char *name) :
    __impl::util::spinlock(name),
    locked_(false),
    owner_(0)
{
    pthread_mutex_init(&internal_, NULL);
}

spinlock::~spinlock()
{
    pthread_mutex_destroy(&internal_);
}

void spinlock::lock() const
{
    pthread_mutex_lock(&internal_);
    REQUIRES(owner_ != pthread_self());
    pthread_mutex_unlock(&internal_);

    __impl::util::spinlock::lock();

    pthread_mutex_lock(&internal_);
    ENSURES(owner_ == 0);
    ENSURES(locked_ == false);
    locked_ = true;
    owner_ = pthread_self();
    pthread_mutex_unlock(&internal_);
}

void spinlock::unlock() const
{
    pthread_mutex_lock(&internal_);
    REQUIRES(locked_ == true);
    REQUIRES(owner_ == pthread_self());
    owner_ = 0;
    locked_ = false;

    __impl::util::spinlock::unlock();

    pthread_mutex_unlock(&internal_);
}

#endif

mutex::mutex(const char *name) :
    __impl::util::mutex(name),
    locked_(false),
    owner_(0)
{
    pthread_mutex_init(&internal_, NULL);
}

mutex::~mutex()
{
    pthread_mutex_destroy(&internal_);
}

void mutex::lock() const
{
    pthread_mutex_lock(&internal_);
    REQUIRES(owner_ != pthread_self());
    pthread_mutex_unlock(&internal_);

    __impl::util::mutex::lock();

    pthread_mutex_lock(&internal_);
    ENSURES(owner_ == 0);
    ENSURES(locked_ == false);
    locked_ = true;
    owner_ = pthread_self();
    pthread_mutex_unlock(&internal_);
}

void mutex::unlock() const
{
    pthread_mutex_lock(&internal_);
    REQUIRES(locked_ == true);
    REQUIRES(owner_ == pthread_self());
    owner_ = 0;
    locked_ = false;

    __impl::util::mutex::unlock();

    pthread_mutex_unlock(&internal_);
}

lock_rw::lock_rw(const char *name) :
    __impl::util::lock_rw(name),
    state_(Idle),
    writer_(0)
{
    ENSURES(pthread_mutex_init(&internal_, NULL) == 0);
}

lock_rw::~lock_rw()
{
    ENSURES(pthread_mutex_destroy(&internal_) == 0);
    readers_.clear();
}

void lock_rw::lock_read() const
{
    pthread_mutex_lock(&internal_);
    REQUIRES(readers_.find(pthread_self()) == readers_.end() &&
             writer_ != pthread_self());
    pthread_mutex_unlock(&internal_);

    __impl::util::lock_rw::lock_read();

    pthread_mutex_lock(&internal_);
    ENSURES(state_ == Idle || state_ == Read);
    state_ = Read;
    readers_.insert(pthread_self());
    pthread_mutex_unlock(&internal_);
}

void lock_rw::lock_write() const
{
    pthread_mutex_lock(&internal_);
    REQUIRES(readers_.find(pthread_self()) == readers_.end());
    REQUIRES(writer_ != pthread_self());
    pthread_mutex_unlock(&internal_);

    __impl::util::lock_rw::lock_write();

    pthread_mutex_lock(&internal_);
    ENSURES(readers_.empty() == true);
    ENSURES(state_ == Idle);
    state_ = Write;
    writer_ = pthread_self();
    pthread_mutex_unlock(&internal_);
}

void lock_rw::unlock() const
{
    pthread_mutex_lock(&internal_);
    if(writer_ == pthread_self()) {
        REQUIRES(readers_.empty() == true);
        REQUIRES(state_ == Write);
        state_ = Idle;
        writer_ = 0;
    }
    else {
        REQUIRES(readers_.erase(pthread_self()) == 1);
        REQUIRES(state_ == Read);
        if(readers_.empty() == true) state_ = Idle;
    }

    __impl::util::lock_rw::unlock();

    pthread_mutex_unlock(&internal_);
}

#endif

}}

#endif
