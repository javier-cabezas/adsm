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

#ifndef __UTIL_POSIX_LOCK_H_
#define __UTIL_POSIX_LOCK_H_


#include <gmac/paraver.h>

#include <pthread.h>
#include <string>
#include <iostream>
#include <map>

namespace gmac { namespace util {

class ParaverLock {
protected:
#ifdef PARAVER
    static const char *eventName;
    static const char *exclusiveName;

    typedef std::map<std::string, unsigned> LockMap;
    static unsigned count;
    static LockMap *map;
    unsigned id;

    static paraver::EventName *event;
    static paraver::StateName *exclusive;

    void setup();
#endif
public:
    ParaverLock(const char *name);

    void enter();
    void locked();
    void exit();
};

class Lock : public ParaverLock {
protected:
	pthread_mutex_t __mutex;
public:
	Lock(const char *__name);
	~Lock();

protected:
	void lock();
	void unlock();
};

class RWLock : public ParaverLock {
protected:
	pthread_rwlock_t __lock;
    bool __write;
public:
	RWLock(const char *__name);
	~RWLock();

protected:
	void lockRead();
	void lockWrite();
	void unlock();
};

}}

#include "Lock.ipp"

#endif
