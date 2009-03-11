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

#ifndef __REGION_H_
#define __REGION_H_

#include <common/config.h>
#include <common/threads.h>

#include <unistd.h>
#include <stdint.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/syscall.h>

#include <assert.h>

#include <list>

namespace gmac {

//! Generic Memory Region Descriptor
class MemRegion {
protected:
	//! Starting memory address for the region
	void *addr;
	//! Size in bytes of the region
	size_t size;
	//! CPU thread owning the region
	pthread_t owner;
public:
	//! Constructor
	//! \param addr Start memory address
	//! \param size Size in bytes
	MemRegion(void *addr, size_t size) :
		addr(addr),
		size(size),
		owner(gettid())
	{}

	//! Comparision operator
	bool operator==(const void *p) const {
		return (uint8_t *)p >= (uint8_t *)addr &&
				(uint8_t *)p < ((uint8_t *)addr + size);
	}

	bool operator==(const MemRegion &m) const {
		return addr == m.addr;
	}

	//! Returns the size (in bytes) of the Region
	inline size_t getSize() const { return size; }
	//! Sets the size (in bytes) of the Region
	inline void setSize(size_t size) { this->size = size; }
	//! Returns the address of the Region
	inline void *getAddress() const { return addr; }
	//! Sets the address of the Region
	inline void setAddress(void *addr) { this->addr = addr; }
	//! Checks if the current thread is the owner for the region
	inline bool isOwner() const { return owner == gettid(); }
};


//! Functor to locate MemRegions inside pointer lists
class FindMem {
protected:
	void *addr;
public:
	FindMem(void *addr) : addr(addr) {};
	bool operator()(const MemRegion *r) const {
		return (*r) == addr;
	}
};


//! Functor to order MemRegions inside pointer multisets
class LessMem {
public:
	bool operator()(const MemRegion *a, const MemRegion *b) const {
		return a->getAddress() < b->getAddress();
	}
};


class ProtRegion;

//! Handler for Read/Write faults
class MemHandler {
public:
	virtual void read(ProtRegion *, void *) = 0;
	virtual void write(ProtRegion *, void *) = 0;
};


//! Protected Memory Region
class ProtRegion : public MemRegion {
protected:
	MemHandler &memHandler;
	bool dirty;

	static struct sigaction defaultAction;
	static MUTEX(regionMutex);
	static std::list<ProtRegion *> regionList;
	static void setHandler(void);
	static void restoreHandler(void);
	static void segvHandler(int, siginfo_t *, void *);
public:
	ProtRegion(MemHandler &memHandler, void *addr, size_t size);
	~ProtRegion();

	inline void read(void *addr) { memHandler.read(this, addr); }
	inline void write(void *addr) { memHandler.write(this, addr); }

	inline void noAccess(void) {
		mprotect(addr, size, PROT_NONE);
	}
	inline void readOnly(void) {
		mprotect(addr, size, PROT_READ);
	}
	inline void readWrite(void) {
		mprotect(addr, size, PROT_READ | PROT_WRITE);
	}

	inline void clear() { dirty = false; }
	inline void setDirty() { dirty = true; }
	inline bool isDirty() const { return dirty; }
};
};

#endif
