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

#ifndef __CACHEMANAGER_H_
#define __CACHEMANAGER_H_

#include "MemManager.h"
#include "MemRegion.h"
#include "threads.h"

#include <vector>

namespace gmac {

class ProtSubRegion;
class CacheRegion : public MemRegion {
protected:
	typedef std::list<ProtSubRegion *> Set;
	Set set;
	Set present;
	size_t cacheLine;

	void deleteSet(Set &set);

public:
	CacheRegion(MemHandler &, void *, size_t, size_t);
	~CacheRegion();

	void invalidate();

	inline void invalid(ProtSubRegion *region) { present.remove(region); }
};

class ProtSubRegion : public ProtRegion {
protected:
	CacheRegion *parent;
public:
	ProtSubRegion(CacheRegion *parent, MemHandler &memHandler,
			void *addr, size_t size) :
		ProtRegion(memHandler, addr, size),
		parent(parent)
	{ }

	virtual inline void noAccess() { 
		parent->invalid(this);
		ProtRegion::noAccess();
	}
};


class CacheManager : public MemManager, public MemHandler {
protected:
	static const size_t lineSize = 64;
	static const size_t lruSize = 2;
	size_t pageSize;

	typedef HASH_MAP<void *, CacheRegion *> Map;
	MUTEX(memMutex);
	Map memMap;

	typedef std::list<ProtSubRegion *> Cache;
	HASH_MAP<pthread_t, Cache> regionCache;

	void writeBack(pthread_t tid);
	void flushToDevice(pthread_t tid);
	
public:
	CacheManager();
	virtual bool alloc(void *addr, size_t size);
	virtual void *safeAlloc(void *addr, size_t size);
	virtual void release(void *addr);
	virtual void execute(void);
	virtual void sync(void);

	virtual void read(ProtRegion *region, void *addr);
	virtual void write(ProtRegion *region, void *addr);
};

};

#endif
