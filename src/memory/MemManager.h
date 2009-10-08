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

#ifndef __MEMORY_MEMMANAGER_H_
#define __MEMORY_MEMMANAGER_H_

#include <memory/Map.h>
#include <memory/PageTable.h>
#include <memory/os/Memory.h>

#include <config.h>
#include <threads.h>

#include <kernel/Context.h>
#include <kernel/Process.h>

#include <stdint.h>

#include <iostream>

namespace gmac {
class MemRegion;

//! Memory Manager Interface

//! Memory Managers implement a policy to move data from/to
//! the CPU memory to/from the accelerator memory.
class MemManager {
private:
	MUTEX(mutex);

protected:
//	memory::PageTable pageTable;

	inline void insert(MemRegion *r) {
		Context::current()->mm().insert(r);
	}

	inline MemRegion *remove(void *addr) {
		MemRegion *ret = Context::current()->mm().remove(addr);
		return ret;
	}

	inline memory::Map *current() {
		if(Context::current() == NULL) return NULL;
		return &Context::current()->mm();
	}

	inline unsigned long align(void *addr) const {
		unsigned long a = (unsigned long)addr;
		unsigned n = 0x1;
		while((a & n) == 0) { n = n << 1; }
		TRACE("Align 0x%x", n);
		return n;
	}

	void insertVirtual(void *cpuPtr, void *devPtr, size_t count);

	inline const memory::PageTable &pageTable() const {
		return Context::current()->mm().pageTable();
	}

	//! This gets memory from the CPU address space
	//! \param addr accelerator address
	//! \param count Size (in bytes) of the mapping
	//! \param prot Protection flags for the mapping
	void *map(void *addr, size_t count, int prot = PROT_READ | PROT_WRITE);

	//! This method upmaps a accelerator address from the CPU address space
	//! \param addr accelerator address
	//! \param count Size (in bytes) to unmap
	void unmap(void *addr, size_t count);

public:
	MemManager() {
		MUTEX_INIT(mutex);
		TRACE("Memory manager starts");
	}
	//! Virtual Destructor. It does nothing
	virtual ~MemManager() {
		TRACE("Memory manager finishes");
		MUTEX_DESTROY(mutex);
	}
	
	//! This method is called whenever the user
	//! requests memory to be used by the accelerator
	//! \param devPtr Allocated memory address. This address
	//! is the same for both, the CPU and the accelerator
	//! \param count Size in bytes of the allocated memory
	virtual void *alloc(void *addr, size_t count) = 0;

	//! This method is called whenever the user
	//! releases accelerator memory
	//! \param devPtr Memory address that has been released
	virtual void release(void *addr) = 0;

	//! This method is called whenever the user invokes
	//! a kernel to be executed at the accelerator
	virtual void flush(void) = 0;

	//! This method is called just after the user requests
	//! waiting for the accelerator to finish
	virtual void sync(void) = 0;

	//! This method is called when a CPU to accelerator translation is
	//! requiered
	//! \param addr Memory address at the CPU
	virtual inline const void *safe(const void *addr) {
		memory::PageTable &pageTable =
			gmac::Context::current()->mm().pageTable();
		const void *ret = (const void *)pageTable.translate(addr);
		if(ret == NULL) ret = proc->translate(addr);
		return ret;
	}

	virtual inline void *safe(void *addr) {
		memory::PageTable &pageTable =
			gmac::Context::current()->mm().pageTable();
		void *ret = (void *)pageTable.translate(addr);
		if(ret == NULL) ret = proc->translate(addr);
		return ret;
	}

	virtual Context *owner(const void *addr) = 0;
	virtual void invalidate(const void *addr, size_t) = 0;
	virtual void flush(const void *addr, size_t) = 0;

};


//! Gets a Memory Manager based on a string name
//! \param managerName Name of the memory manager
MemManager *getManager(const char *managerName);

};
#endif
