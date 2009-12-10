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
        if (gpu.async()) {
            CUresult ret;
            ret = cuStreamCreate(&streamLaunch, 0);
            if(ret != CUDA_SUCCESS)
                FATAL("Unable to create CUDA stream %d", ret);
            ret = cuStreamCreate(&streamToDevice, 0);
            if(ret != CUDA_SUCCESS)
                FATAL("Unable to create CUDA stream %d", ret);
            ret = cuStreamCreate(&streamToHost, 0);
            if(ret != CUDA_SUCCESS)
                FATAL("Unable to create CUDA stream %d", ret);
            ret = cuStreamCreate(&streamDevice, 0);
            if(ret != CUDA_SUCCESS)
                FATAL("Unable to create CUDA stream %d", ret);
        }

        if (gpu.async()) {
            TRACE("Using page locked memory: %zd\n", _bufferPageLockedSize);
            assert(cuMemHostAlloc(&_bufferPageLocked, paramBufferPageLockedSize, CU_MEMHOSTALLOC_PORTABLE) == CUDA_SUCCESS);
            _bufferPageLockedSize = paramBufferPageLockedSize;
        } else {
            _bufferPageLocked     = NULL;
            _bufferPageLockedSize = 0;
        }
    }
	inline void setup() {
		mutex = new util::Lock(paraver::ctxLocal);
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
		delete mutex;
        if (gpu.async()) {
            cuStreamDestroy(streamLaunch);
            cuStreamDestroy(streamToDevice);
            cuStreamDestroy(streamToHost);
            cuStreamDestroy(streamDevice);
        }
		cuCtxDestroy(ctx); 
	}

public:

	inline static Context *current() {
		return static_cast<Context *>(PRIVATE_GET(key));
	}

	inline void lock() {
		mutex->lock();
		assert(cuCtxPushCurrent(ctx) == CUDA_SUCCESS);
	}
	inline void unlock() {
		CUcontext tmp;
		assert(cuCtxPopCurrent(&tmp) == CUDA_SUCCESS);
		mutex->unlock();
	}


	// Standard Accelerator Interface
	inline gmacError_t malloc(void **addr, size_t size) {
		zero(addr);
		lock();
		size += mm().pageTable().getPageSize();
		CUdeviceptr ptr = 0;
		CUresult ret = cuMemAlloc(&ptr, size);
		if(ptr % mm().pageTable().getPageSize()) {
			ptr += mm().pageTable().getPageSize() -
				(ptr % mm().pageTable().getPageSize());
		}
		*addr = (void *)ptr;
		unlock();
		return error(ret);
	}

	inline gmacError_t free(void *addr) {
		lock();
		CUresult ret = cuMemFree(gpuAddr(addr));
		unlock();
		return error(ret);
	}

	gmacError_t hostAlloc(void **host, void **device, size_t size);
	gmacError_t hostMemAlign(void **host, void **device, size_t size);
	gmacError_t hostMap(void *host, void **device, size_t size);
	gmacError_t hostFree(void *addr);

	inline gmacError_t copyToDevice(void *dev, const void *host, size_t size) {
		enterFunction(accHostDeviceCopy);
        gmac::Context *ctx = gmac::Context::current();

        CUresult ret;
        if (gpu.async()) {
            size_t bufferSize = ctx->bufferPageLockedSize();
            void * tmp        = ctx->bufferPageLocked();

            size_t left = size;
            off_t  off  = 0;
            while (left != 0) {
                size_t bytes = left < bufferSize? left: bufferSize;
                memcpy(tmp, ((char *) host) + off, bytes);
                lock();
                ret = cuMemcpyHtoDAsync(gpuAddr(((char *) dev) + off), tmp, bytes, streamToDevice);
                if (ret != CUDA_SUCCESS) { unlock(); goto done; }
                ret = cuStreamSynchronize(streamToDevice);
                unlock();
                if (ret != CUDA_SUCCESS) goto done;

                left -= bytes;
                off  += bytes;
            }           
        } else {
            lock();
            ret = cuMemcpyHtoD(gpuAddr(dev), host, size);
            unlock();
        }

done:
		exitFunction();
		return error(ret);
	}

	inline gmacError_t copyToHost(void *host, const void *dev, size_t size) {
        enterFunction(accDeviceHostCopy);
        gmac::Context *ctx = gmac::Context::current();

        CUresult ret;
        if (gpu.async()) {
            size_t bufferSize = ctx->bufferPageLockedSize();
            void * tmp        = ctx->bufferPageLocked();

            size_t left = size;
            off_t  off  = 0;
            while (left != 0) {
                size_t bytes = left < bufferSize? left: bufferSize;
                lock();
                ret = cuMemcpyDtoHAsync(tmp, gpuAddr(((char *) dev) + off), bytes, streamToHost);
                if (ret != CUDA_SUCCESS) { unlock(); goto done; }
                ret = cuStreamSynchronize(streamToHost);
                unlock();
                if (ret != CUDA_SUCCESS) goto done;
                memcpy(((char *) host) + off, tmp, bytes);

                left -= bytes;
                off  += bytes;
            }           
        } else {
            lock();
            ret = cuMemcpyDtoH(host, gpuAddr(dev), size);
            unlock();
        }

done:
        exitFunction();
		return error(ret);
	}

	inline gmacError_t copyDevice(void *dst, const void *src, size_t size) {
		enterFunction(accDeviceDeviceCopy);
		lock();
		
        CUresult ret;
        ret = cuMemcpyDtoD(gpuAddr(dst), gpuAddr(src), size);

		unlock();
		exitFunction();
		return error(ret);
	}

	inline gmacError_t copyToDeviceAsync(void *dev, const void *host,
			size_t size) {
		lock();
		enterFunction(accHostDeviceCopy);
		CUresult ret = cuMemcpyHtoDAsync(gpuAddr(dev), host, size, streamToDevice);
		exitFunction();
		unlock();
		return error(ret);
	}

	inline gmacError_t copyToHostAsync(void *host, const void *dev,
			size_t size) {
		lock();
		enterFunction(accDeviceHostCopy);
		CUresult ret = cuMemcpyDtoHAsync(host, gpuAddr(dev), size, streamToHost);
		exitFunction();
		unlock();
		return error(ret);
	}
   
#if 0
    inline gmacError_t copyDeviceAsync(void *src, const void *dest,
			size_t size) {
		lock();
		enterFunction(accDeviceCopy);
		CUresult ret = cuMemcpyDtoDAsync(gpuAddr(dest), gpuAddr(src), size, streamDevice);
		exitFunction();
		unlock();
		return error(ret);
	}
#endif

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
        if (ret == CUDA_SUCCESS) {
            TRACE("Sync: success");
            ret = cuStreamSynchronize(streamLaunch);
        } else {
            TRACE("Sync: error: %d", ret);
        }
        unlock();

        return error(ret);
    }

    inline gmacError_t syncToHost() {
        CUresult ret;
        lock();
        if (gpu.async()) {
            ret = cuStreamSynchronize(streamToHost);
        } else {
            ret = cuCtxSynchronize();
        }
        unlock();
        return error(ret);
    }

    inline gmacError_t syncToDevice() {
        CUresult ret;
        lock();
        if (gpu.async()) {
            ret = cuStreamSynchronize(streamToDevice);
        } else {
            ret = cuCtxSynchronize();
        }
        unlock();
        return error(ret);
    }

    inline gmacError_t syncDevice() {
        CUresult ret;
        lock();
        if (gpu.async()) {
            ret = cuStreamSynchronize(streamDevice);
        } else {
            ret = cuCtxSynchronize();
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

    inline bool async() const
    {
        return gpu.async();
    }

	void flush();
	void invalidate();
};


}}

#endif
