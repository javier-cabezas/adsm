/* Copyright (c) 2009 University of Illinois
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

#ifndef GMAC_UTIL_POSIX_TEST_LOCK_H_
#define GMAC_UTIL_POSIX_TEST_LOCK_H_

#include <sys/types.h>
#include <pthread.h>

#include <set>

#include "config/common.h"
#include "test/types.h"
#include "util/posix/Lock.h"

namespace gmac { namespace util { 

class GMAC_LOCAL LockTest :
    public gmac::util::LockImpl,
    public virtual gmac::test::Contract {
protected:
    mutable pthread_mutex_t internal_;
    mutable bool locked_;
    mutable pthread_t owner_;

public:
    LockTest(const char *name);
    VIRTUAL ~LockTest();

    TESTABLE void lock() const;
    TESTABLE void unlock() const;
};

class GMAC_LOCAL RWLockTest :
    public gmac::util::RWLockImpl,
    public virtual gmac::test::Contract {
protected:
    mutable enum { Idle, Read, Write } state_;
    mutable pthread_mutex_t internal_;
    mutable std::set<pthread_t> readers_;
    mutable pthread_t writer_;
public:
    RWLockTest(const char *name);
    VIRTUAL ~RWLockTest();

    TESTABLE void lockRead() const;
    TESTABLE void lockWrite() const;
    TESTABLE void unlock() const;
};

}}

#endif
