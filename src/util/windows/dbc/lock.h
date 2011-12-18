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

#ifndef GMAC_UTIL_WINDOWS_DBC_LOCK_H_
#define GMAC_UTIL_WINDOWS_DBC_LOCK_H_

#include "config/config.h"
#include "config/dbc/types.h"
#include "config/dbc/Contract.h"
#include "util/windows/mutex.h"

#include <windows.h>

#include <set>

namespace __dbc { namespace util {

class GMAC_LOCAL spinlock :
    public __impl::util::spinlock,
    public virtual Contract {
    DBC_TESTED(__impl::util::spinlock)

protected:
    mutable CRITICAL_SECTION internal_;
    mutable bool locked_;
    mutable DWORD owner_;

public:
    spinlock(const char *name);
    VIRTUAL ~spinlock();
protected:
    TESTABLE void lock() const;
    TESTABLE void unlock() const;
};

class GMAC_LOCAL mutex : public __impl::util::mutex, public Contract {
    DBC_TESTED(__impl::util::mutex)

protected:
    mutable CRITICAL_SECTION internal_;
    mutable bool locked_;
    mutable DWORD owner_;

public:
    mutex(const char *name);
    VIRTUAL ~mutex();
protected:
    TESTABLE void lock() const;
    TESTABLE void unlock() const;
};

class GMAC_LOCAL lock_rw : public __impl::util::lock_rw, public Contract {
    DBC_TESTED(__impl::util::lock_rw)

protected:
    mutable enum { Idle, Read, Write } state_;
    mutable CRITICAL_SECTION internal_;
    mutable std::set<DWORD> readers_;
    mutable DWORD writer_;
public:
    lock_rw(const char *name);
    VIRTUAL ~lock_rw();
protected:
    TESTABLE void lockRead() const;
    TESTABLE void lockWrite() const;
    TESTABLE void unlock() const;
};

}}

#endif
