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

#ifndef __MEMORY_MEMMANAGER_H_
#define __MEMORY_MEMMANAGER_H_

#include <memory/Map.h>
#include <memory/PageTable.h>
#include <memory/os/Memory.h>

#include <config.h>

#include <kernel/Context.h>
#include <kernel/Process.h>

#include <stdint.h>

#include <iostream>
#include <iterator>
#include <map>


namespace gmac { namespace memory {

#define GMAC_MALLOC_PINNED 1
#define GMAC_MALLOC_GLOBAL 2

//! Memory Manager Interface

//! Memory Managers implement a policy to move data from/to
//! the CPU memory to/from the accelerator memory.
class Manager {
protected:

#ifdef USE_MMAP
	static const size_t mmSize = 0x100000000;
#endif

	void insert(Region *r);

	Region *remove(void *addr);

	Map *current();

	void insertVirtual(Context *ctx, void *cpuPtr, void *devPtr, size_t count);
	void removeVirtual(Context *ctx, void *cpuPtr, size_t count);
	void insertVirtual(void *cpuPtr, void *devPtr, size_t count);
    void removeVirtual(void *cpuPtr, size_t count);

	const PageTable &pageTable() const;

	//! This gets memory from the CPU address space
	//! \param addr accelerator address
	//! \param count Size (in bytes) of the mapping
	//! \param prot Protection flags for the mapping
	void *mapToHost(void *addr, size_t count, int prot);

    //! This remaps memory in the CPU address space
	//! \param addr  accelerator address
	//! \param hAddr address to remap
	//! \param count Size (in bytes) of the mapping
	void *hostRemap(void *addr, void *hAddr, size_t count);

	//! This method upmaps a accelerator address from the CPU address space
	//! \param addr accelerator address
	//! \param count Size (in bytes) to unmap
	void hostUnmap(void *addr, size_t count);

	virtual void remap(Context *, Region *, void *) = 0;
	void unmap(Context *, Region *);

    virtual Region * newRegion(void * addr, size_t count, bool shared);
    virtual int defaultProt();

public:
	Manager();
	//! Virtual Destructor. It does nothing
	virtual ~Manager();
	
	//! This method is called whenever the user
	//! requests memory to be used by the host
	//! \param addr Allocated memory address. This address
	//! is the same for both, the CPU and the accelerator
	//! \param count Size in bytes of the allocated memory
	gmacError_t malloc(void ** addr, size_t count);

    //! This method is called whenever the user
	//! requests shared memory to be used by all the accelerators
	//! \param devPtr Allocated memory address. This address
	//! is the same for both, the CPU and the accelerator
	//! \param count Size in bytes of the allocated memory
	gmacError_t globalMalloc(void ** addr, size_t count);

#if 0
	//! This methid is called to map accelerator memory to
	//! system memory. Coherence is not maintained for these mappings
	//! \param cpuAddr
	//! \param devPtr
	//! \param count
	virtual void map(void *host, void *dev, size_t count);
#endif

	//! This method is called whenever the user
	//! releases accelerator memory
	//! \param devPtr Memory address that has been released
	gmacError_t free(void *addr);

	//! This method is called whenever the user invokes
	//! a kernel to be executed at the accelerator
	virtual void flush() = 0;
    virtual void flush(const RegionSet & regions) = 0;

    //! This method is called after the user invokes
	//! a kernel to be executed at the accelerator
	virtual void invalidate() = 0;
    virtual void invalidate(const RegionSet & regions) = 0;

#if 0
	//! This method is called just after the user requests
	//! waiting for the accelerator to finish
	virtual void sync(void) = 0;
#endif

	//! This method is called when a CPU to accelerator translation is
	//! requiered
	//! \param addr Memory address at the CPU
	static const void *ptr(Context *ctx, const void *addr);
	static const void *ptr(const void *addr);
	static void *ptr(Context *ctx, void *addr);
	static void *ptr(void *addr);

    void initShared(Context *);

	Context *owner(const void *addr);
	virtual void invalidate(const void *addr, size_t) = 0;
	virtual void flush(const void *addr, size_t) = 0;
};


//! Gets a Memory Manager based on a string name
//! \param managerName Name of the memory manager
Manager *getManager(const char *managerName);

#include "Manager.ipp"

} };
#endif
