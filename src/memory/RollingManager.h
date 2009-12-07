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

#ifndef __MEMOMRY_ROLLINGMANAGER_H_
#define __MEMOMRY_ROLLINGMANAGER_H_

#include "Handler.h"
#include "RollingRegion.h"

#include <kernel/Context.h>

#include <threads.h>

#include <map>
#include <list>

namespace gmac { namespace memory {

class RollingBuffer {
private:
	std::list<ProtSubRegion *> buffer;
	MUTEX(mutex);
	size_t _max;
	
	inline void lock() {
		enterLock(rolling);
		MUTEX_LOCK(mutex);
		exitLock();
	}

	inline void unlock() { MUTEX_UNLOCK(mutex); }

public:
	RollingBuffer() : _max(0) {
		MUTEX_INIT(mutex);
	}

	inline bool overflows() const { return buffer.size() >= _max; }
	inline size_t inc(size_t n) { _max += n; }
	inline size_t dec(size_t n) { _max -= n; }
	inline bool empty() const { return buffer.empty(); }

	inline void push(ProtSubRegion *region) {
		lock();
		buffer.push_back(region);
		unlock();
	}

	inline ProtSubRegion *pop() {
		assert(buffer.empty() == false);
		lock();
		ProtSubRegion *ret = buffer.front();
		buffer.pop_front();
		unlock();
		return ret;
	}

	inline ProtSubRegion *front() { return buffer.front(); }

	inline void remove(ProtSubRegion *region) {
		lock();
		buffer.remove(region);
		unlock();
	}
};

class RollingManager : public Handler {
protected:
	size_t lineSize;
	size_t lruDelta;

	std::map<Context *, RollingBuffer> regionRolling;

	inline RollingRegion *get(const void *addr) {
		RollingRegion *reg = NULL;
		if(current()) reg = current()->find<RollingRegion>(addr);
		return reg;
	}

	MUTEX(writeMutex);
	void *writeBuffer;
	size_t writeBufferSize;
	void waitForWrite(void *addr = NULL, size_t size = 0);
	void writeBack();
	void flushToDevice();

	virtual bool read(void *);
	virtual bool write(void *);

#ifdef DEBUG
	void dumpRolling();
#endif

	// Methods used by ProtSubRegion to request flushing and invalidating
	friend class RollingRegion;
	void invalidate(ProtSubRegion *region) {
		regionRolling[Context::current()].remove(region);
	}
	void flush(ProtSubRegion *region) {
		regionRolling[Context::current()].remove(region);
		assert(region->copyToDevice() == gmacSuccess);
	}

public:
	RollingManager();
	void *alloc(void *addr, size_t size);
	void release(void *addr);
	void flush(void);
	void sync(void) {};

	Context *owner(const void *addr);
	void invalidate(const void *addr, size_t size);
	void flush(const void *addr, size_t size);
};

} };

#endif
