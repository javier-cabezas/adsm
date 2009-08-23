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

#ifndef __API_CUDA_GPUCONTEXT_H_
#define __API_CUDA_GPUCONTEXT_H_

#include <config.h>
#include <debug.h>
#include <threads.h>
#include <paraver.h>

#include "GPU.h"

#include <os/loader.h>
#include <kernel/Context.h>

#include <stdint.h>
#include <cuda.h>
#include <vector_types.h>

#include <vector>
#include <list>


IMPORT_SYM(cudaError_t, __cudaLaunch, const char*);

namespace gmac { namespace gpu {

class Context : public gmac::Context {
protected:
	GPU &gpu;

	inline void check() { assert(current() == this); }

	gmacError_t error(cudaError_t);

	friend class gmac::GPU;

	Context(GPU &gpu) : gpu(gpu) {
		PRIVATE_SET(key, this);
		cudaSetDevice(gpu.device());
		TRACE("New GPU context [%p]", this);
	}

	~Context() {
		TRACE("Remove GPU context [%p]", this);
	}
	
public:

	inline void lock() {};
	inline void release() {};

	// Standard Accelerator Interface
	inline gmacError_t malloc(void **addr, size_t size) {
		check();
		cudaError_t ret = cudaMalloc(addr, size);
		return error(ret);
	}

	inline gmacError_t free(void *addr) {
		check();
		cudaError_t ret = cudaFree(addr);
		return error(ret);
	}

	inline gmacError_t copyToDevice(void *dev, const void *host, size_t size) {
		check();
		cudaError_t ret = cudaMemcpy(dev, host, size, cudaMemcpyHostToDevice);
		return error(ret);
	}

	inline gmacError_t copyToHost(void *host, const void *dev, size_t size) {
		check();
		cudaError_t ret = cudaMemcpy(host, dev, size, cudaMemcpyDeviceToHost);
		return error(ret);
	}

	inline gmacError_t copyDevice(void *dst, const void *src, size_t size) {
		check();
		cudaError_t ret = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice);
		return error(ret);
	}

	inline gmacError_t copyToDeviceAsync(void *dev, const void *host,
			size_t size) {
		check();
		cudaError_t ret = cudaMemcpyAsync(dev, host, size,
			cudaMemcpyHostToDevice, 0);
		return error(ret);
	}

	inline gmacError_t copyToHostAsync(void *host, const void *dev,
			size_t size) {
		check();
		cudaError_t ret = cudaMemcpyAsync(host, dev, size,
				cudaMemcpyDeviceToHost, 0);
		return error(ret);
	}

	inline gmacError_t memset(void *dev, int c, size_t size) {
		check();
		cudaError_t ret = cudaMemset(dev, c, size);
		return error(ret);
	}

	gmacError_t launch(const char *kernel) {
		check();
		cudaError_t ret = __cudaLaunch(kernel);
		return error(ret);
	};
	
	inline gmacError_t sync() {
		check();
		cudaError_t ret = cudaThreadSynchronize();
		return error(ret);
	}
};

}}

#endif
