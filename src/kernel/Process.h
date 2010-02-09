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

#ifndef __KERNEL_PROCESS_H_
#define __KERNEL_PROCESS_H_

#include <paraver.h>
#include <debug.h>

#include <kernel/Queue.h>
#include <memory/Map.h>

#include <cassert>
#include <vector>
#include <list>

namespace gmac {
class Accelerator;
class Context;
class Process;

void apiInit(void);
void contextInit(void);
void memoryInit(const char *name = NULL);
void memoryFini(void);
}

extern gmac::Process *proc;

namespace gmac {

class SharedMemory {
protected:
	void *_addr;
	size_t _size;
	size_t _count;
public:
	SharedMemory(void *_addr, size_t _size, size_t _count = 1);

	void *start() const;
	size_t size() const;

	void inc();
	size_t dec();
};

class ThreadQueue {
public:
    ThreadQueue();
    util::Lock hasContext;
    Queue * queue;
};

class Process {
public:
	typedef std::list<Context *> ContextList;
	typedef std::map<void *, SharedMemory> SharedMap;
protected:
	std::vector<Accelerator *> _accs;
	ContextList _contexts;
	
	typedef std::map<THREAD_ID, ThreadQueue> QueueMap;
	QueueMap _queues;

	util::Lock mutex;
	unsigned current;

	SharedMap _sharedMem;

	static size_t _totalMemory;

	Process();

public:
	virtual ~Process();

	static void init(const char *name);

	void initThread();
	void create();
#define ACC_AUTO_BIND -1
	void clone(Context *ctx, int acc = ACC_AUTO_BIND);
	void remove(Context *ctx);
	gmacError_t migrate(int acc);
	const ContextList &contexts() const;

	void accelerator(Accelerator *acc);

	void *translate(void *);
	const void *translate(const void *addr);
	void sendReceive(THREAD_ID id);

	SharedMap &sharedMem();
	void addShared(void *addr, size_t size);
	bool removeShared(void *addr);
	bool isShared(void *addr) const;

	static size_t totalMemory();
	size_t accs() const;
};

#include "Process.ipp"

}

#endif
