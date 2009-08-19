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

#include <kernel/Context.h>

#include <stdint.h>
#include <cuda.h>
#include <vector_types.h>

#include <vector>
#include <list>

namespace gmac {

class GPUContext : public Context {
protected:
	GPU &gpu;

	inline void check() { assert(current == this); }

public:
	GPUContext(GPU &gpu) : gpu(gpu) {
		PRIVATE_SET(key, this);
		cudaSetDevice(gpu.device());
	}

	// Standard Accelerator Interface
	inline gmacError_t malloc(void **addr, size_t size) {
		check();
		gmacError_t ret = gpu.malloc(addr, size);
		return error(ret);
	}

	inline gmacError_t free(void *addr) {
		check();
		gmacError_t ret = gpu.free(addr);
		return error(ret);
	}

	inline gmacError_t copyToDevice(void *dev, const void *host, size_t size) {
		check();
		gmacError_t ret = gpu.copyToDevice(dev, host, size);
		return error(ret);
	}

	inline gmacError_t copyToHost(void *host, const void *dev, size_t size) {
		check();
		gmacError_t ret = gpu.copyToHost(host, dev, size);
		return error(ret);
	}

	inline gmacError_t copyDevice(void *dst, const void *src, size_t size) {
		check();
		gmacError_t ret = gpu.copyDevice(dst, src, size);
		return error(ret);
	}

	inline gmacError_t copyToDeviceAsync(void *dev, const void *host,
			size_t size) {
		check();
		gmacError_t ret = gpu.copyToDeviceAsync(dev, host, size);
		return error(ret);
	}

	inline gmacError_t copyToHostAsync(void *host, const void *dev,
			size_t size) {
		check();
		gmacError_t ret = gpu.copyToHostAsync(host, dev, size);
		return error(ret);
	}

	inline gmacError_t memset(void *dev, int c, size_t size) {
		check();
		gmacError_t ret = gpu.memset(dev, c, size);
		return error(ret);
	}

	gmacError_t launch(const char *kernel) {
		check();
		gmacError_t ret = gpu.launch(kernel);
		return error(ret);
	};
	
	inline gmacError_t sync() {
		check();
		gmacError_t ret = gpu.sync();
		return error(ret);
	}
};

}

#endif
