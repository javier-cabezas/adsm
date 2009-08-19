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

#include <kernel/Context.h>

#include <stdint.h>
#include <cuda.h>
#include <vector_types.h>

#include <vector>
#include <list>

namespace gmac {

class GPUContext : public Context {
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
public:
	typedef struct Variable {
		Variable(CUdeviceptr ptr, size_t size) : ptr(ptr), size(size) {};
		CUdeviceptr ptr;
		size_t size;
	} Variable;
protected:
	typedef std::vector<Call> CallStack;
	typedef HASH_MAP<const void *, CUfunction> FunctionMap;
	typedef HASH_MAP<const void *, Variable> VariableMap;
	typedef std::list<CUtexref *> Textures;

	GPU &gpu;

	CallStack _calls;
	FunctionMap _functions;
	VariableMap _variables;
	VariableMap _constants;
	Textures _textures;

	static const unsigned StackSize = 4096;
	uint8_t _stack[StackSize];
	size_t _sp;

	CUcontext ctx;
	MUTEX(mutex);


public:
	GPUContext(GPU &gpu) : gpu(gpu), _sp(0) {
		CUcontext tmp;
		MUTEX_INIT(mutex);
		if((cuCtxCreate(&ctx, 0, gpu.device()) != CUDA_SUCCESS))
			FATAL("Unable to create CUDA context");
		assert(cuCtxPopCurrent(&tmp) == CUDA_SUCCESS);
		PRIVATE_SET(key, this);
	}

	~GPUContext() { MUTEX_DESTROY(mutex); }

	inline void lock() {
		pushState(Lock);
		MUTEX_LOCK(mutex);
		assert(cuCtxPushCurrent(ctx) == CUDA_SUCCESS);
		popState();
	}
	inline void release() {
		CUcontext tmp;
		assert(cuCtxPopCurrent(&tmp) == CUDA_SUCCESS);
		MUTEX_UNLOCK(mutex);
	}


	// Standard Accelerator Interface
	inline gmacError_t malloc(void **addr, size_t size) {
		lock();
		gmacError_t ret = gpu.malloc(addr, size);
		release();
		return error(ret);
	}

	inline gmacError_t free(void *addr) {
		lock();
		gmacError_t ret = gpu.free(addr);
		release();
		return error(ret);
	}

	inline gmacError_t copyToDevice(void *dev, const void *host, size_t size) {
		lock();
		gmacError_t ret = gpu.copyToDevice(dev, host, size);
		release();
		return error(ret);
	}

	inline gmacError_t copyToHost(void *host, const void *dev, size_t size) {
		lock();
		gmacError_t ret = gpu.copyToHost(host, dev, size);
		release();
		return error(ret);
	}

	inline gmacError_t copyDevice(void *dst, const void *src, size_t size) {
		lock();
		gmacError_t ret = gpu.copyDevice(dst, src, size);
		release();
		return error(ret);
	}

	inline gmacError_t copyToDeviceAsync(void *dev, const void *host,
			size_t size) {
		lock();
		gmacError_t ret = gpu.copyToDeviceAsync(dev, host, size);
		release();
		return error(ret);
	}

	inline gmacError_t copyToHostAsync(void *host, const void *dev,
			size_t size) {
		lock();
		gmacError_t ret = gpu.copyToHostAsync(host, dev, size);
		release();
		return error(ret);
	}

	inline gmacError_t memset(void *dev, int c, size_t size) {
		lock();
		gmacError_t ret = gpu.memset(dev, c, size);
		release();
		return error(ret);
	}

	gmacError_t launch(const char *kernel);
	
	inline gmacError_t sync() {
		lock();
		gmacError_t ret = gpu.sync();
		release();
		return error(ret);
	}

	// CUDA-related methods
	inline void function(const char *host, CUfunction dev) {
		_functions.insert(FunctionMap::value_type(host, dev));
	}
	inline const CUfunction *function(const char *host) const {
		FunctionMap::const_iterator f;
		if((f = _functions.find(host)) == _functions.end()) return NULL;
		return &f->second;
	}

	inline void variable(const char *host, CUdeviceptr ptr, size_t size) {
		Variable variable(ptr, size);
		_variables.insert(VariableMap::value_type(host, variable));
	}
	inline const Variable *variable(const char *host) const {
		VariableMap::const_iterator v;
		if((v = _variables.find(host)) == _variables.end()) return NULL;
		return &v->second;
	}

	inline void constant(const char *host, CUdeviceptr ptr, size_t size) {
		Variable constant(ptr, size);
		_constants.insert(VariableMap::value_type(host, constant));
	}
	inline const Variable *constant(const char *host) const {
		VariableMap::const_iterator c;
		if((c = _constants.find(host)) == _constants.end()) return NULL;
		return &c->second;
	}

	inline void bind(CUtexref *tex) {
		_textures.push_back(tex);
	}
	inline void unbind(CUtexref *tex) {
		_textures.remove(tex);
	}

	inline void call(dim3 Dg, dim3 Db, size_t shared, int tokens) {
		Call c(Dg, Db, shared, tokens, _sp);
		_calls.push_back(c);
	}
	inline void argument(const void *arg, size_t size, off_t offset) {
		memcpy(&_stack[offset], arg, size);
		_sp = (_sp > (offset + size)) ? _sp : offset + size;
	}
};


}

#endif
