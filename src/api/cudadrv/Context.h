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
#include <threads.h>
#include <paraver.h>

#include "GPU.h"
#include "Module.h"

#include <kernel/Context.h>

#include <stdint.h>
#include <cuda.h>
#include <vector_types.h>

#include <vector>
#include <list>

namespace gmac { namespace gpu {


class Context : public gmac::Context {
protected:
	typedef struct Call {
		Call(dim3 grid, dim3 block, size_t shared, size_t tokens,
				size_t stack) : grid(grid),
			block(block),
			shared(shared),
			tokens(tokens),
			stack(stack) {};
		dim3 grid;
		dim3 block;
		size_t shared;
		size_t tokens;
		size_t stack;
	} Call;

protected:
	typedef std::vector<Call> CallStack;
	typedef HASH_MAP<Module *, const void *> ModuleMap;

	GPU &gpu;
	ModuleMap modules;

	CallStack _calls;

	static const unsigned USleepLaunch = 1000;

	static const unsigned StackSize = 4096;
	uint8_t _stack[StackSize];
	size_t _sp;

#ifdef USE_VM
	static const char *pageTableSymbol;
	const Variable *pageTable;
	struct {
		void *ptr;
		size_t shift;
		size_t size;
		size_t page;
	} devicePageTable;
#endif

	CUcontext ctx;
	MUTEX(mutex);

	int major;
	int minor;

    CUstream streamLaunch;

	inline CUdeviceptr gpuAddr(void *addr) const {
		unsigned long a = (unsigned long)addr;
		return (CUdeviceptr)(a & 0xffffffff);
	}

	inline CUdeviceptr gpuAddr(const void *addr) const {
		unsigned long a = (unsigned long)addr;
		return (CUdeviceptr)(a & 0xffffffff);
	}

	inline void zero(void **addr) const {
		memory::addr_t *ptr = (memory::addr_t *)addr;
		*ptr = 0;
	}

	gmacError_t error(CUresult);

	friend class gmac::GPU;

    inline void setupStreams() {
        CUresult ret;
        ret = cuStreamCreate(&streamLaunch, 0);
        if(ret != CUDA_SUCCESS)
            FATAL("Unable to create CUDA stream %d", ret);

        TRACE("Created launch stream: %p", streamLaunch);
    }
	inline void setup() {
		MUTEX_INIT(mutex);
		CUcontext tmp;
		assert(cuDeviceComputeCapability(&major, &minor, gpu.device()) ==
			CUDA_SUCCESS);
		unsigned int flags = 0;
		if(major > 0 && minor > 0) flags |= CU_CTX_MAP_HOST;
		CUresult ret = cuCtxCreate(&ctx, flags, gpu.device());
		if(ret != CUDA_SUCCESS)
			FATAL("Unable to create CUDA context %d", ret);
		assert(cuCtxPopCurrent(&tmp) == CUDA_SUCCESS);

        enable();
	}

	Context(GPU &gpu) :
		gmac::Context(gpu), gpu(gpu), _sp(0)
#ifdef USE_VM
		, pageTable(NULL)
#endif
	{
		setup();

        lock();
        setupStreams();
        unlock();

		TRACE("New GPU context [%p]", this);
	}

	Context(const Context &root, GPU &gpu);

	~Context() {
		TRACE("Remove GPU context [%p]", this);
		MUTEX_DESTROY(mutex);
        cuStreamDestroy(streamLaunch);
		cuCtxDestroy(ctx); 
	}

public:

	inline static Context *current() {
		return static_cast<Context *>(PRIVATE_GET(key));
	}

	inline void lock() {
		enterLock(ctxLocal);
		MUTEX_LOCK(mutex);
		exitLock();
		assert(cuCtxPushCurrent(ctx) == CUDA_SUCCESS);
	}
	inline void unlock() {
		CUcontext tmp;
		assert(cuCtxPopCurrent(&tmp) == CUDA_SUCCESS);
		MUTEX_UNLOCK(mutex);
	}


	// Standard Accelerator Interface
	inline gmacError_t malloc(void **addr, size_t size) {
		zero(addr);
		lock();
		CUresult ret = cuMemAlloc((CUdeviceptr *)addr, size);
		unlock();
		return error(ret);
	}

	inline gmacError_t halloc(void **host, void **device, size_t size) {
		zero(host); zero(device);
		lock();
		CUresult ret = cuMemHostAlloc(host, size, CU_MEMHOSTALLOC_DEVICEMAP);
		if(ret == CUDA_SUCCESS) {
            ret = cuMemHostGetDevicePointer((CUdeviceptr *)device, *host, 0);
			assert(ret == CUDA_SUCCESS);
        }
		unlock();
		return error(ret);
	}

	inline gmacError_t hmap(void *host, void **device) {
		zero(device);
		lock();
		CUresult ret = cuMemHostGetDevicePointer((CUdeviceptr *)device, host, 0);
		unlock();
		return error(ret);
	}

	inline gmacError_t free(void *addr) {
		lock();
		CUresult ret = cuMemFree(gpuAddr(addr));
		unlock();
		return error(ret);
	}

	inline gmacError_t hfree(void *addr) {
		lock();
		CUresult ret = cuMemFreeHost(addr);
		unlock();
		return error(ret);
	}

	inline gmacError_t copyToDevice(void *dev, const void *host, size_t size) {
		lock();
		enterFunction(accHostDeviceCopy);

        CUresult ret;
        ret = cuMemcpyHtoD(gpuAddr(dev), host, size);

		exitFunction();
        unlock();
		return error(ret);
	}

	inline gmacError_t copyToHost(void *host, const void *dev, size_t size) {
		lock();
		enterFunction(accDeviceHostCopy);

        CUresult ret;
		ret = cuMemcpyDtoH(host, gpuAddr(dev), size);

		exitFunction();
		unlock();
		return error(ret);
	}

	inline gmacError_t copyDevice(void *dst, const void *src, size_t size) {
		lock();
		enterFunction(accDeviceDeviceCopy);
		
        CUresult ret;
        ret = cuMemcpyDtoD(gpuAddr(dst), gpuAddr(src), size);

		exitFunction();
		unlock();
		return error(ret);
	}

	inline gmacError_t copyToDeviceAsync(void *dev, const void *host,
			size_t size) {
		lock();
		enterFunction(accHostDeviceCopy);
		CUresult ret = cuMemcpyHtoDAsync(gpuAddr(dev), host, size, 0);
		exitFunction();
		unlock();
		return error(ret);
	}

	inline gmacError_t copyToHostAsync(void *host, const void *dev,
			size_t size) {
		lock();
		enterFunction(accDeviceHostCopy);
		CUresult ret = cuMemcpyDtoHAsync(host, gpuAddr(dev), size, 0);
		exitFunction();
		unlock();
		return error(ret);
	}

	gmacError_t memset(void *dev, int c, size_t size);
	gmacError_t launch(const char *kernel);
	
	inline gmacError_t sync() {
        CUresult ret;
        lock();
        while ((ret = cuStreamQuery(streamLaunch)) == CUDA_ERROR_NOT_READY) {
            unlock();
            usleep(Context::USleepLaunch);
            lock();
        }
        unlock();
		return error(ret);
	}

	// CUDA-related methods
	inline Module *load(void *fatBin) {
		lock();
		Module *module = new Module(fatBin);
		modules.insert(ModuleMap::value_type(module, fatBin));
		unlock();
		return module;
	}

	inline void unload(Module *mod) {
		ModuleMap::iterator m = modules.find(mod);
		assert(m != modules.end());
		lock();
		delete m->first;
		modules.erase(m);
		unlock();
	}

	inline const Function *function(const char *name) const {
		ModuleMap::const_iterator m;	
		for(m = modules.begin(); m != modules.end(); m++) {
			const Function *func = m->first->function(name);
			if(func != NULL) return func;
		}
		return NULL;
	}

	inline const Variable *constant(const char *name) const {
		ModuleMap::const_iterator m;
		for(m = modules.begin(); m != modules.end(); m++) {
			const Variable *var = m->first->constant(name);
			if(var != NULL) return var;
		}
		return NULL;
	}

	inline void call(dim3 Dg, dim3 Db, size_t shared, int tokens) {
		Call c(Dg, Db, shared, tokens, _sp);
		_calls.push_back(c);
	}
	inline void argument(const void *arg, size_t size, off_t offset) {
		memcpy(&_stack[offset], arg, size);
		_sp = (_sp > (offset + size)) ? _sp : offset + size;
	}

	void flush();
	void invalidate();
};


}}

#endif
