/* Copyright (c) 2009, 2010 University of Illinois
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

#ifndef __KERNEL_PROCESS_H_
#define __KERNEL_PROCESS_H_

#include <paraver.h>

#include "Queue.h"
#include <memory/Map.h>
#include <gmac/gmac.h>
#include <util/Logger.h>

#include <vector>
#include <list>
#include <map>

namespace gmac {
class Accelerator;
class Context;
class Process;

void apiInit(void);
void contextInit(void);
void memoryInit(const char *manager = NULL, const char *allocator = NULL);
void memoryFini(void);
}

extern gmac::Process *proc;

namespace gmac {

class ThreadQueue : public util::Lock {
public:
    ThreadQueue();
    ~ThreadQueue();
    Queue * queue;
};

class ContextList : public std::list<Context *>, public util::RWLock
{
public:
    ContextList();
};


class QueueMap : public std::map<THREAD_ID, ThreadQueue *>, public util::RWLock
{
public:
    QueueMap();
};

class Process : public util::RWLock{
protected:
    friend class Accelerator;

	std::vector<Accelerator *> _accs;
	ContextList _contexts;

	QueueMap _queues;
    memory::RegionMap _global;
    memory::RegionMap _shared;

	unsigned current;

	static size_t _totalMemory;

	Process();

public:
	virtual ~Process();

	static void init(const char *manager, const char *allocator);

	void initThread();
#define ACC_AUTO_BIND -1
    Context * create(int acc = ACC_AUTO_BIND);
	void remove(Context *ctx);
	ContextList & contexts();

	void *translate(void *addr) const;
	const void *translate(const void *addr) const;

    /* Context management functions */
    void send(THREAD_ID id);
    void receive();
	void sendReceive(THREAD_ID id);
    void copy(THREAD_ID id);
	gmacError_t migrate(Context *ctx, int acc);

	void addAccelerator(Accelerator *acc);

	static size_t totalMemory();
	size_t nAccelerators() const;

    memory::RegionMap &global();
    const memory::RegionMap &global() const;
    memory::RegionMap &shared();
    const memory::RegionMap &shared() const;
};

}

#include "Process.ipp"

#endif
