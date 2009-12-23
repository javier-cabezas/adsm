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

#ifndef __API_CUDADRV_GPUCONTEXT_H_
#define __API_CUDADRV_GPUCONTEXT_H_

#include <config.h>
#include <debug.h>
#include <paraver.h>

#include "GPU.h"
#include "Module.h"

#include <util/Lock.h>
#include <kernel/Context.h>

#include <stdint.h>
#include <cuda.h>
#include <vector_types.h>

#include <vector>
#include <map>

namespace gmac { namespace gpu {

class Context : public gmac::Context {
protected:
	struct Call {
		Call(dim3 grid, dim3 block, size_t shared, size_t tokens, size_t stack);
		dim3 grid;
		dim3 block;
		size_t shared;
		size_t tokens;
		size_t stack;
	};

	typedef std::vector<Call> CallStack;
	typedef HASH_MAP<Module *, const void *> ModuleMap;

	GPU &gpu;
	ModuleMap modules;

	CallStack _calls;

	static const unsigned USleepLaunch = 100;

	static const unsigned StackSize = 4096;
	uint8_t _stack[StackSize];
	size_t _sp;

	typedef std::map<void *, void *> AddressMap;
	AddressMap hostMem;

#ifdef USE_VM
	static const char *pageTableSymbol;
	const Variable *pageTable;
	struct {
		void *ptr;
		size_t shift;
		size_t size;
		size_t page;
	} devicePageTable;
#elif USE_MMAP
	static const char *baseRegisterSymbol;
#endif

	CUcontext ctx;
	util::Lock *mutex;

	int major;
	int minor;

    CUstream streamLaunch;
    CUstream streamToDevice;
    CUstream streamToHost;
    CUstream streamDevice;

	CUdeviceptr gpuAddr(void *addr) const;
	CUdeviceptr gpuAddr(const void *addr) const;
	void zero(void **addr) const;

	gmacError_t error(CUresult);

	friend class gmac::GPU;

    /*! Auxiliary functions used during Context creation
     */
	void setup();
    void setupStreams();

	Context(GPU &gpu);
	Context(const Context &root, GPU &gpu);

	~Context();
public:
	static Context *current();

	void lock();
    void unlock();

    //
	// Standard Accelerator Interface
    //
	gmacError_t malloc(void **addr, size_t size);
	gmacError_t free(void *addr);

	gmacError_t hostAlloc(void **host, void **device, size_t size);
	gmacError_t hostMemAlign(void **host, void **device, size_t size);
	gmacError_t hostMap(void *host, void **device, size_t size);
	gmacError_t hostFree(void *addr);

	gmacError_t copyToDevice(void *dev, const void *host, size_t size);
	gmacError_t copyToHost(void *host, const void *dev, size_t size);
	gmacError_t copyDevice(void *dst, const void *src, size_t size);

	gmacError_t copyToDeviceAsync(void *dev, const void *host, size_t size);
	gmacError_t copyToHostAsync(void *host, const void *dev, size_t size);
   
#if 0
    gmacError_t copyDeviceAsync(void *src, const void *dest, size_t size);
#endif

	gmacError_t memset(void *dev, int c, size_t size);
	gmacError_t launch(const char *kernel);
	
    gmacError_t sync();
    gmacError_t syncToHost();
    gmacError_t syncToDevice();
    gmacError_t syncDevice();

    //
	// CUDA-related methods
	//
    Module *load(void *fatBin);
	void unload(Module *mod);

	const Function *function(const char *name) const;
	const Variable *constant(const char *name) const;

	void call(dim3 Dg, dim3 Db, size_t shared, int tokens);
	void argument(const void *arg, size_t size, off_t offset);

    bool async() const;

	void flush();
	void invalidate();
};

#include "Context.ipp"

}}

#endif
