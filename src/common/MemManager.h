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

#ifndef __MEMMANAGER_H_
#define __MEMMANAGER_H_

#include <common/config.h>
#include <common/threads.h>

#include <stdint.h>
#include <unistd.h>
#include <sys/mman.h>

#include <cuda.h>

namespace gmac {
//! Memory Manager Interface

//! Memory Managers implement a policy to move data from/to
//! the CPU memory to/from the GPU memory.
class MemManager {
private:
	MUTEX(virtMutex);
	HASH_MAP<void *, void *> virtTable;
	size_t pageSize;
	CUcontext ctx;

	void insertVirtual(void *cpuPtr, void *devPtr, size_t count);

protected:
	//! This method maps a GPU address into the CPU address space
	//! \param addr GPU address
	//! \param count Size (in bytes) of the mapping
	//! \param prot Protection flags for the mapping
	void *map(void *addr, size_t count, int prot = PROT_READ | PROT_WRITE);

	//! This gets memory from the CPU address space
	//! \param addr GPU address
	//! \param count Size (in bytes) of the mapping
	//! \param prot Protection flags for the mapping
	void *safeMap(void *addr, size_t count, int prot = PROT_READ | PROT_WRITE);

	//! This method upmaps a GPU address from the CPU address space
	//! \param addr GPU address
	//! \param count Size (in bytes) to unmap
	void unmap(void *addr, size_t count);

public:
	MemManager();
	//! Virtual Destructor. It does nothing
	virtual ~MemManager() { MUTEX_DESTROY(virtMutex); }

	//! This methods attaches a manager to an execution thread
	virtual inline bool attach(void) {
		cuCtxAttach(&ctx, 0);
	}

	//! This method is called whenever the user
	//! requests memory to be used by the GPU
	//! \param devPtr Allocated memory address. This address
	//! is the same for both, the CPU and the GPU.
	//! \param count Size in bytes of the allocated memory
	virtual bool alloc(void *addr, size_t count) = 0;

	//! This method is called whenever the user
	//! requests memory to be used by the GPU that
	//! might use different memory addresses for the CPU
	//! and the GPU.
	//! \param devPtr Allocated memory address. This address
	//! is the same for both, the CPU and the GPU.
	//! \param count Size in bytes of the allocated memory
	virtual void *safeAlloc(void *addr, size_t count) = 0;

	//! This method is called whenever the user
	//! releases GPU memory
	//! \param devPtr Memory address that has been released
	virtual void release(void *addr) = 0;

	//! This method is called whenever the user invokes
	//! a kernel to be executed at the GPU
	virtual void execute(void) = 0;

	//! This method is called just after the user requests
	//! waiting for the GPU to finish
	virtual void sync(void) = 0;

	//! This method is called when a CPU to GPU translation is
	//! requiered
	//! \param addr Memory address at the CPU
	virtual inline void *safe(void *addr) {
		HASH_MAP<void *, void *>::const_iterator e;
		void *baseAddr = (void *)((unsigned long)addr & ~(pageSize -1));
		size_t off = (unsigned long)addr & (pageSize - 1);
		void *devAddr = NULL;
		MUTEX_LOCK(virtMutex);
		if((e = virtTable.find(baseAddr)) != virtTable.end())
			devAddr = (void *)((uint8_t *)e->second + off);
		MUTEX_UNLOCK(virtMutex);
		return devAddr;
	}
};


//! Gets a Memory Manager based on a string name
//! \param managerName Name of the memory manager
MemManager *getManager(const char *managerName);

};
#endif
