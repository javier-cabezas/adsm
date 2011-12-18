/* Copyright (c) 2009-2011 Universityrsity of Illinois
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

#ifndef GMAC_UTIL_POSIX_LOCK_H_
#define GMAC_UTIL_POSIX_LOCK_H_

#include <pthread.h>

#include <string>
#include <iostream>
#include <map>

#include "config/config.h"
#include "config/dbc/types.h"
#include "util/lock.h"


namespace __impl { namespace util {

#if defined(__APPLE__)
class mutex;
typedef mutex spinlock;
#else

template <typename T>
class spinlocker;

template <typename T>
class locker;

template <typename T>
class locker_rw;

template <typename T>
class GMAC_API spinlock :
	public __impl::util::lock__ {
    DBC_FORCE_TEST(spinlock)
    friend class spinlocker<T>;

protected:
	mutable pthread_spinlock_t spinlock_;
public:
	typedef spinlocker<T> locker_type;

	spinlock(const char *name);
	VIRTUAL ~spinlock();

protected:
	TESTABLE void lock() const;
	TESTABLE void unlock() const;
};
#endif

template <typename T>
class GMAC_API mutex :
	public __impl::util::lock__ {
    DBC_FORCE_TEST(mutex)
	friend class locker<T>;
protected:
	mutable pthread_mutex_t mutex_;
public:
	typedef locker<T> locker_type;

	mutex(const char *name);
	VIRTUAL ~mutex();

protected:
	TESTABLE void lock() const;
	TESTABLE void unlock() const;
};

template <typename T>
class GMAC_API lock_rw : public __impl::util::lock__ {
    DBC_FORCE_TEST(lock_rw)
	friend class locker_rw<T>;
protected:
	mutable pthread_rwlock_t lock_;
    bool write_;
public:
    typedef locker_rw<T> locker_type;

	lock_rw(const char *name);
	VIRTUAL ~lock_rw();

protected:
	TESTABLE void lock_read() const;
	TESTABLE void lock_write() const;
	TESTABLE void unlock() const;
};

template <typename T>
class spinlocker {
protected:
	void lock(spinlock<T> &l) const
	{
		l.lock();
	}

	void unlock(spinlock<T> &l) const
	{
		l.unlock();
	}
};

template <typename T>
class locker {
protected:
	void lock(mutex<T> &l) const
	{
		l.lock();
	}

	void unlock(mutex<T> &l) const
	{
		l.unlock();
	}
};

template <typename T>
class locker_rw {
protected:
	void lock_read(lock_rw<T> &l) const
	{
		l.lock_read();
	}

	void lock_write(lock_rw<T> &l) const
	{
		l.lock_write();
	}

	void unlock(lock_rw<T> &l) const
	{
		l.unlock();
	}
};

}}

#include "lock-impl.h"

#ifdef USE_DBC
#include "dbc/lock.h"
#endif

#endif
