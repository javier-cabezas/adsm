/* Copyright (c) 2009-2011 University of Illinois
                   Universitat Politecnica de Catalunya
                   All rights reserved.

Developed by: IMPACT Research Group / Grup de Sistemes Operatius
              University of Illinois / Universitat Politecnica de Catalunya
              http://impact.crhc.illinois.edu/
              http://gso.ac.upc.edu/

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to
deal with the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
  1. Redistributions of source code must retain the above copyright notice,
     this list of conditions and the following disclaimers.
  2. Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimers in the
     documentation and/or other materials provided with the distribution.
  3. Neither the names of IMPACT Research Group, Grup de Sistemes Operatius,
     University of Illinois, Universitat Politecnica de Catalunya, nor the
     names of its contributors may be used to endorse or promote products
     derived from this Software without specific prior written permission.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
WITH THE SOFTWARE.  */

#ifndef GMAC_UTIL_POSIX_DBC_LOCK_H_
#define GMAC_UTIL_POSIX_DBC_LOCK_H_

#include <pthread.h>

#include <set>

#include "config/config.h"
#include "config/dbc/Contract.h"
#include "config/dbc/types.h"
#include "util/posix/lock.h"

namespace __dbc { namespace util {

#if defined(__APPLE__)
class mutex;
typedef mutex spinlock;
#else
template <typename T>
class GMAC_API spinlock :
    public __impl::util::spinlock<T>,
    public virtual Contract {
    DBC_TESTED(__impl::util::spinlock<T>)

protected:
    mutable pthread_mutex_t internal_;
    mutable bool locked_;
    mutable pthread_t owner_;
public:
    spinlock(const char *name) :
    	__impl::util::spinlock<T>(name),
        locked_(false),
        owner_(0)
    {
        pthread_mutex_init(&internal_, NULL);
    }

    virtual ~spinlock()
    {
    	pthread_mutex_destroy(&internal_);
    }
protected:
    void lock() const
    {
    	pthread_mutex_lock(&internal_);
		REQUIRES(owner_ != pthread_self());
		pthread_mutex_unlock(&internal_);

		__impl::util::spinlock<T>::lock();

		pthread_mutex_lock(&internal_);
		ENSURES(owner_ == 0);
		ENSURES(locked_ == false);
		locked_ = true;
		owner_ = pthread_self();
		pthread_mutex_unlock(&internal_);
	}

    void unlock() const
    {
        pthread_mutex_lock(&internal_);
        REQUIRES(locked_ == true);
        REQUIRES(owner_ == pthread_self());
        owner_ = 0;
        locked_ = false;

        __impl::util::spinlock<T>::unlock();

        pthread_mutex_unlock(&internal_);
    }
};
#endif

template <typename T>
class GMAC_API mutex :
    public __impl::util::mutex<T>,
    public virtual Contract {
    DBC_TESTED(__impl::util::mutex<T>)

protected:
    mutable pthread_mutex_t internal_;
    mutable bool locked_;
    mutable pthread_t owner_;
public:
    mutex(const char *name) :
        __impl::util::mutex<T>(name),
    	locked_(false),
    	owner_(0)
    {
    	pthread_mutex_init(&internal_, NULL);
    }

    virtual ~mutex()
    {
    	pthread_mutex_destroy(&internal_);
    }
protected:
    void lock() const
    {
    	pthread_mutex_lock(&internal_);
		REQUIRES(owner_ != pthread_self());
		pthread_mutex_unlock(&internal_);

		__impl::util::mutex<T>::lock();

		pthread_mutex_lock(&internal_);
		ENSURES(owner_ == 0);
		ENSURES(locked_ == false);
		locked_ = true;
		owner_ = pthread_self();
    	pthread_mutex_unlock(&internal_);
    }

    void unlock() const
    {
    	pthread_mutex_lock(&internal_);
		REQUIRES(locked_ == true);
		REQUIRES(owner_ == pthread_self());
		owner_ = 0;
		locked_ = false;

		__impl::util::mutex<T>::unlock();

		pthread_mutex_unlock(&internal_);
    }
};

template <typename T>
class GMAC_API lock_rw :
    public __impl::util::lock_rw<T>,
    public virtual Contract {
    DBC_TESTED(__impl::util::lock_rw<T>)

protected:
    mutable enum { Idle, Read, Write } state_;
    mutable pthread_mutex_t internal_;
    mutable std::set<pthread_t> readers_;
    mutable pthread_t writer_;

    mutable unsigned magic_;
public:
    lock_rw(const char *name) :
        __impl::util::lock_rw<T>(name),
        state_(Idle),
        writer_(0)
    {
        ENSURES(pthread_mutex_init(&internal_, NULL) == 0);
    }

    virtual ~lock_rw()
    {
        ENSURES(pthread_mutex_destroy(&internal_) == 0);
        readers_.clear();
    }
protected:
    void lock_read() const
    {
    	pthread_mutex_lock(&internal_);
		REQUIRES(readers_.find(pthread_self()) == readers_.end() &&
				 writer_ != pthread_self());
		pthread_mutex_unlock(&internal_);

		__impl::util::lock_rw<T>::lock_read();

		pthread_mutex_lock(&internal_);
		ENSURES(state_ == Idle || state_ == Read);
		state_ = Read;
		readers_.insert(pthread_self());
		pthread_mutex_unlock(&internal_);
    }

    void lock_write() const
    {
        pthread_mutex_lock(&internal_);
        REQUIRES(readers_.find(pthread_self()) == readers_.end());
        REQUIRES(writer_ != pthread_self());
        pthread_mutex_unlock(&internal_);

        __impl::util::lock_rw<T>::lock_write();

        pthread_mutex_lock(&internal_);
        ENSURES(readers_.empty() == true);
        ENSURES(state_ == Idle);
        state_ = Write;
        writer_ = pthread_self();
        pthread_mutex_unlock(&internal_);
    }

    void unlock() const
    {
    	pthread_mutex_lock(&internal_);
		if (writer_ == pthread_self()) {
			REQUIRES(readers_.empty() == true);
			REQUIRES(state_ == Write);
			state_ = Idle;
			writer_ = 0;
		} else {
			REQUIRES(readers_.erase(pthread_self()) == 1);
			REQUIRES(state_ == Read);
			if (readers_.empty() == true)
				state_ = Idle;
		}

		__impl::util::lock_rw<T>::unlock();

		pthread_mutex_unlock(&internal_);
    }
};

}}

#endif
